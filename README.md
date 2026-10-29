# ECHO — Playlist-Search Discord DJ

ECHO is a local-first Discord DJ bot designed for voice channels. It listens to real-time voice conversations, parses voice commands via regex for near-instant response times without LLMs in the critical path, and walks curated/live YouTube Music and SoundCloud playlists.

---

> [!WARNING]
> ## Privacy & Voice Transcription Disclosure
> ECHO **transcribes** all audio spoken in the voice channel—not just commands prefixed with the wake word.
> - **Real-time transcription**: All incoming voice audio is transcribed locally via Faster-Whisper.
> - **In-memory buffer**: The last 50 transcript lines are kept in memory for conversational room sentiment.
> - **Terminal logs**: Transcripts are printed along with Discord usernames to the host terminal.
> - **Local LLM processing**: Ambient transcripts are fed to the local Ollama LLM to generate track transition commentary and sentiment.
> - **No remote storage**: All processing runs locally on the host machine.
>
> **Important:** Please inform all members in your Discord server/voice channel that the bot transcribes speech in real time while present in voice.

---

## Architecture

ECHO runs as a dual-process system connected via local TCP:

1. **Python Brain (`bot.py`)**:
   - Manages voice command recognition via WebRTC VAD and Faster-Whisper.
   - Parses voice commands with zero-latency regex matching (pure `parse_voice_command`, sub-millisecond parsing, ~2s to TTS acknowledgement, <10s to music).
   - Searches and resolves playlists via YouTube Music (`ytmusicapi`) and SoundCloud (`yt-dlp`).
   - Runs the Playlist Walk Engine: ranked top-3 playlist pick, random start offset, 10–15 track walk before re-roll, and 14-day video ID deduplication.
   - Delivers ambient room sentiment and commentary using a single local 8b LLM (`huihui_ai/qwen3-abliterated:8b-v2` via Ollama) on song transitions.
   - Generates local TTS responses via Kokoro ONNX (`bf_alice`).

2. **Node Voice Bridge (`voice-bridge/bridge.js`)**:
   - Handles Discord voice gateway connections, voice stream reception, Opus decryption, 48kHz→16kHz downsampling, and PCM streaming to Python.
   - Manages playback audio streaming (`@discordjs/voice`) and background prefetching via `yt-dlp`.
   - Communicates with Python over a local TCP socket (`127.0.0.1:47777`).

---

## Setup & Requirements

### 1. Prerequisites
- Python 3.10+
- Node.js 18+
- [Ollama](https://ollama.com/) running locally
- System binary: `yt-dlp` (tested on version `2026.08.19`+; must be accessible on system PATH)

### 2. Python Environment
Create virtual environment and install pinned dependencies:
```bash
python -m venv dj-bot
.\dj-bot\Scripts\python.exe -m pip install -r requirements.txt
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
Create a `.env` file in the project root with the following variables (never commit this file):
- `DISCORD_TOKEN`: Discord Bot Token
- `GUILD_ID`: Target Discord Server ID
- `VOICE_CHANNEL_ID`: Discord Voice Channel ID
- `TEXT_CHANNEL_ID`: Discord Text Channel ID
- `IGNORE_USER_IDS`: Comma-separated Discord user IDs to ignore from voice listening / transcription (prevents audio feedback loops)
- `OLLAMA_MODEL`: Ollama model tag (`huihui_ai/qwen3-abliterated:8b-v2`)
- `TTS_VOICE`: Kokoro TTS voice name (`bf_alice`)
- `TTS_ENABLED`: `true` or `false`
- `JOIN_AUTOPLAY`: `true` or `false`
- `WHISPER_MODEL`: Faster-Whisper model size (`small`)
- `SPOKEN_LANGUAGE`: Spoken audio language (`en`)
- `BRIDGE_PORT`: Local TCP bridge port (default: `47777`)
- `DEBUG`: `true` or `false`

---

## Slash Commands

ECHO provides a full suite of slash commands for manual control:

- `/join`: Summons Echo to the configured voice channel and initializes the audio bridge.
- `/leave`: Disconnects from voice, shuts down the voice bridge, and releases Ollama model VRAM.
- `/playlist <url>`: Directly plays a YouTube or SoundCloud playlist URL.
- `/queue`: Displays the current playlist queue, playing track, and upcoming tracks.
- `/status`: Displays system status, active persona, voice bridge connection, active walk playlist, and now-playing track.
- `/persona`: Interactively switches Echo's personality via a Discord UI select menu.
- `/skip`: Skips the currently playing track.
- `/back`: Rewinds to the previous track in the active walk.
- `/pause`: Pauses playback.
- `/resume`: Resumes paused playback.
- `/stop`: Stops playback, clears the queue, and sets playback intent to stopped.

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

## Features

### Chat Playlist-Link Auto-Detection
Posting any YouTube or SoundCloud playlist URL directly into the bot's text channel will automatically load and begin walking the playlist without requiring a slash command.

### Instant Lyrics Display
When a song starts playing, Echo automatically queries LRCLIB for synchronized lyrics and posts them into the text channel. If lyrics cannot be found, Echo posts a short polite notification (`🎵 Couldn't find lyrics for <title>.`).

### Personas
Echo features three personalities switchable at runtime via `/persona` or on channel join:
1. **Unhinged** (Default): Feral gremlin DJ with no filter and main-character syndrome. Roasts the song, room, and self.
2. **Velvet**: Deadpan late-night host who has heard it all and remains quietly, devastatingly unimpressed.
3. **Hype**: Catastrophically over-excited hype-man who treats every song transition like a championship win.

---

## Playlist Walk Engine & Lifecycle

- **Ranked Selection**: Searches for playlists matching the user's query, ranks the top 10 by track count, and picks uniformly from the top 3.
- **Random Offset**: Starts at a random index to ensure variety; never begins at track 1 unless the playlist is very short.
- **Keep-One-Ahead Design**: Operates purely event-driven rather than using drift-prone prefetch timers. The Python controller keeps exactly one track ahead queued in the bridge. As soon as a track transition starts (`track_start`), Python queues the next track.
- **10–15 Track Walks & Seamless Re-roll**: After playing 10–15 tracks from the playlist, the walk engine seamlessly re-rolls a fresh search or curated mix without cutting off the currently playing song (`interrupt=False`).
- **Stop Semantics**: Executing stop (voice or slash) clears the queue, halts playback, and sets module playback intent to `stopped`. The bridge suppresses `queue_empty` refills, ensuring music stays stopped until a new play or resume command is received.
- **30-Second Identical-Play Debounce**: If an identical play query is received within 30 seconds of an active walk start (e.g. from acoustic room echo or repeated speech recognition), Echo replies with *"already on it"* and avoids restarting or duplicating the walk.
- **14-Day Deduplication**: Checks candidate video IDs against `play_history.jsonl` and skips any track played in the last 14 days.
- **Instant Interrupts**: Voice or slash play commands interrupt the current walk immediately, acknowledging within ~1-2 seconds and streaming the first track within seconds.

---

## Running Tests

Run the test suite with plain asserts (no external test runner required):
```bash
.\dj-bot\Scripts\python.exe tests/test_parser.py
```
