# ECHO — Playlist-Search Discord DJ

ECHO is a local-first Discord DJ bot designed for voice channels. It listens to real-time voice conversations, parses voice commands via regex for near-instant response times without LLMs in the critical path, and walks curated/live YouTube Music and SoundCloud playlists.

## Architecture

ECHO runs as a dual-process system connected via local TCP:

1. **Python Brain (`bot.py`)**:
   - Manages voice command recognition via WebRTC VAD and Faster-Whisper.
   - Parses voice commands with zero-latency regex matching (sub-millisecond parsing, ~2s to TTS acknowledgement, <10s to music).
   - Searches and resolves playlists via YouTube Music (`ytmusicapi`) and SoundCloud (`yt-dlp`).
   - Runs the Playlist Walk Engine: ranked top-3 playlist pick, random start offset, 10–15 track walk before re-roll, and 14-day video ID deduplication.
   - Delivers ambient room sentiment and commentary using a single local 8b LLM (`huihui_ai/qwen3-abliterated:8b-v2` via Ollama) on song transitions.
   - Generates local TTS responses via Kokoro ONNX (`bf_alice`).

2. **Node Voice Bridge (`voice-bridge/bridge.js`)**:
   - Handles Discord voice gateway connections, voice stream reception, Opus decryption, and PCM streaming to Python.
   - Manages playback audio streaming (`@discordjs/voice`) and background prefetching via `yt-dlp`.
   - Python communicates with Node over a local TCP socket (`127.0.0.1:47777`).

---

## Setup & Requirements

### 1. Prerequisites
- Python 3.10+
- Node.js 18+
- [Ollama](https://ollama.com/) running locally
- System binary: `yt-dlp` (must be accessible on system PATH)

### 2. Python Environment
Create virtual environment and install dependencies:
```bash
python -m venv dj-bot
.\dj-bot\Scripts\python.exe -m pip install aiohttp py-cord PyNaCl python-dotenv faster-whisper ytmusicapi kokoro-onnx soundfile webrtcvad-wheels
```

### 3. Node Bridge
```bash
cd voice-bridge
npm install
cd ..
```

### 4. Models
- **Ollama**:
  ```bash
  ollama pull huihui_ai/qwen3-abliterated:8b-v2
  ```
- **Kokoro TTS**: Place `kokoro-v1.0.onnx` and `voices-v1.0.bin` in the `models/` folder.

### 5. Build Fallback Playlists
Generate the offline curated fallback playlist catalog:
```bash
.\dj-bot\Scripts\python.exe build_playlists.py
```

### 6. Configuration (`.env`)
Create a `.env` file in the project root with the following variables (do not commit this file):
- `DISCORD_TOKEN`: Discord Bot Token
- `GUILD_ID`: Target Discord Server ID
- `VOICE_CHANNEL_ID`: Discord Voice Channel ID
- `TEXT_CHANNEL_ID`: Discord Text Channel ID
- `OLLAMA_MODEL`: Ollama model tag (`huihui_ai/qwen3-abliterated:8b-v2`)
- `TTS_VOICE`: Kokoro TTS voice name (`bf_alice`)
- `TTS_ENABLED`: `true` or `false`
- `JOIN_AUTOPLAY`: `true` or `false`
- `WHISPER_MODEL`: Faster-Whisper model size (`small`)
- `SPOKEN_LANGUAGE`: Spoken audio language (`en`)
- `BRIDGE_PORT`: Local TCP bridge port (default: `47777`)
- `DEBUG`: `true` or `false`

---

## Voice Commands

All voice commands are prefixed with the wake word **"Echo"** or **"Hey Echo"**:

- **Play Music**:
  - `"Echo, play some aggressive dubstep"` (searches YouTube Music playlists)
  - `"Echo, put on jazz"`
  - `"Echo, throw on a bit of phonk"`
  - `"Echo, play dubstep on soundcloud"` (searches SoundCloud sets)
- **Playback Controls**:
  - `"Echo, skip"` / `"Echo, next"`
  - `"Echo, back"` / `"Echo, previous"`
  - `"Echo, pause"`
  - `"Echo, resume"` / `"Echo, unpause"`
  - `"Echo, stop"`
- **Info & Modes**:
  - `"Echo, what's playing"` / `"Echo, what is this"`
  - `"Echo, radio"` / `"Echo, play mix"`

---

## Personas

Echo features three personalities switchable at runtime via `/persona` or on channel join:

1. **Unhinged** (Default): Feral gremlin DJ with no filter and main-character syndrome. Roasts the song, room, and self.
2. **Velvet**: Deadpan late-night host who has heard it all and remains quietly, devastatingly unimpressed.
3. **Hype**: Catastrophically over-excited hype-man who treats every song transition like a championship win.

*Note: Persona affects only commentary voice, TTS acknowledgements, and song-change reactions. Playback and search remain unopinionated.*

---

## Playlist Walk Engine

- **Ranked Selection**: Searches for playlists matching the user's query, ranks the top 10 by track count, and picks uniformly from the top 3.
- **Random Offset**: Never starts at track 1; begins at a random index to ensure variety.
- **Walk & Re-roll**: Walks 10–15 tracks within the playlist before automatically re-rolling a fresh search for the same query.
- **14-Day Deduplication**: Checks candidate video IDs against `play_history.jsonl` and skips any track played in the last 14 days.
- **Instant Interrupts**: New play commands interrupt the current walk immediately, acknowledging within ~1-2 seconds and playing within ~10 seconds.
- **Background Prefetch**: The next track in the walk is prefetched and cached locally by the bridge while the current track plays.
