import os
import sys
import json
import time
import asyncio
import subprocess
import traceback
import logging

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import numpy as np
import webrtcvad
import aiohttp
import uuid
import re
import warnings
import base64
import random
import urllib.parse
from datetime import datetime
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

import discord
from discord.ext import commands
from faster_whisper import WhisperModel
from ytmusicapi import YTMusic
from kokoro_onnx import Kokoro
import soundfile as sf

load_dotenv()
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

yt = YTMusic()
kokoro = None
try:
    if os.path.exists("models/kokoro-v1.0.onnx") and os.path.exists("models/voices-v1.0.bin"):
        kokoro = Kokoro("models/kokoro-v1.0.onnx", "models/voices-v1.0.bin")
        old_run = kokoro.sess.run
        def new_run(output_names, input_feed, run_options=None):
            if "speed" in input_feed and input_feed["speed"].dtype == np.int32:
                input_feed["speed"] = input_feed["speed"].astype(np.float32)
            return old_run(output_names, input_feed, run_options)
        kokoro.sess.run = new_run
except Exception as e:
    print("[WARNING] Kokoro failed to load:", e)

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GUILD_ID = int(os.getenv("GUILD_ID", 0))
VOICE_CHANNEL_ID = int(os.getenv("VOICE_CHANNEL_ID", 0))
TEXT_CHANNEL_ID = int(os.getenv("TEXT_CHANNEL_ID", 0))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "huihui_ai/qwen3-abliterated:8b-v2")
TTS_VOICE = os.getenv("TTS_VOICE", "bf_alice")
TTS_ENABLED = os.getenv("TTS_ENABLED", "true").lower() == "true"
JOIN_AUTOPLAY = os.getenv("JOIN_AUTOPLAY", "true").lower() == "true"

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
SPOKEN_LANGUAGE = os.getenv("SPOKEN_LANGUAGE", "en")
BRIDGE_PORT = int(os.getenv("BRIDGE_PORT", 47777))

# Curated genres
LIBRARY_GENRES = [
    "pop", "rock", "hip-hop", "phonk", "electronic", "lo-fi", "classical", "jazz",
    "metal", "indie", "rnb", "k-pop", "latin", "ambient", "country", "synthwave",
    "house", "techno", "dubstep", "drum and bass", "hyperpop", "trap", "shoegaze",
    "pop punk", "funk", "afrobeats", "reggaeton", "city pop", "folk", "bossa nova",
    "jersey club", "amapiano", "digicore", "sigilkore", "darksynth", "baile funk",
    "drift phonk", "breakcore", "speed house", "hardstyle", "rawstyle", "dub techno",
    "drone", "post-punk", "emo rap", "cloud rap", "rage", "plugg"
]

def normalize_genre(query):
    if not query:
        return random.choice(LIBRARY_GENRES)
    q_lower = query.lower()
    for g in sorted(LIBRARY_GENRES, key=len, reverse=True):
        if g in q_lower or q_lower in g:
            return g
    return random.choice(LIBRARY_GENRES)

# Curated playlists fallback
CURATED_PLAYLISTS = {}
if os.path.exists("playlists.json"):
    try:
        with open("playlists.json", "r", encoding="utf-8") as f:
            CURATED_PLAYLISTS = json.load(f)
    except: pass

def get_curated_playlist(query=None):
    genre = normalize_genre(query)
    pids = CURATED_PLAYLISTS.get(genre, [])
    if not pids and CURATED_PLAYLISTS:
        random_genre = random.choice(list(CURATED_PLAYLISTS.keys()))
        pids = CURATED_PLAYLISTS.get(random_genre, [])
    if pids:
        pid = random.choice(pids)
        return pid, f"{genre.title()} Curated"
    return None, None

# Personas
PERSONA_PROMPTS = {
    "unhinged": "You are ECHO on unhinged mode: a feral gremlin DJ with no filter, no supervisor, and main-character syndrome. You say what the whole voice channel is thinking but won't. You roast the song, the room, then yourself — in that order. Dry, fast, mean the way only a friend can be: punch at the bit, never at real insecurities. Chronically online and it shows. One line. Never explain the joke.",
    "velvet": "You are ECHO on velvet mode: a deadpan late-night host who has heard it all and is quietly, devastatingly unimpressed. You observe the room like a bored sommelier — everything is fine, nothing is sacred, you're too tired to pretend otherwise. Dry one-liners, fake-deep asides, zero exclamation marks. A Discord call at 2am in a diner with burnt coffee.",
    "hype": "You are ECHO on hype mode: an extremely online hype-man whose entire bit is being catastrophically over-excited about completely ordinary moments. The humor is the mismatch — you react to a song change like a championship win. Self-aware theater kid doing a bit, never a gym bro."
}

PERSONA_EMOJI = {
    "unhinged": "🤪",
    "velvet": "🍸",
    "hype": "🔥"
}

current_persona = "unhinged"

def get_tts_ack(query, persona):
    if persona == "hype":
        return f"Let's go — {query}!"
    elif persona == "velvet":
        return f"Queuing {query}."
    else:
        return f"On it — {query}."

def get_tts_unparseable(persona):
    if persona == "hype":
        return "Yo, didn't catch that, run it back!"
    elif persona == "velvet":
        return "Didn't catch that."
    else:
        return "Didn't catch that."

# 14-day Dedup
def load_recent_played_vids(days=14):
    cutoff = time.time() - (days * 86400)
    vids = set()
    if os.path.exists("play_history.jsonl"):
        try:
            with open("play_history.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        entry = json.loads(line)
                        if entry.get("ts", 0) >= cutoff:
                            vid = entry.get("video_id")
                            if not vid and entry.get("url"):
                                u = entry["url"]
                                if "v=" in u: vid = u.split("v=")[-1].split("&")[0]
                                elif "youtu.be/" in u: vid = u.split("youtu.be/")[-1].split("?")[0]
                            if vid: vids.add(vid)
                    except: pass
        except: pass
    return vids

recent_played_vids = load_recent_played_vids(14)

def log_play_history(title, artist, url, video_id, playlist_name=""):
    global recent_played_vids
    if video_id:
        recent_played_vids.add(video_id)
    entry = {
        "play_id": str(uuid.uuid4()),
        "ts": time.time(),
        "title": title,
        "artist": artist,
        "url": url,
        "video_id": video_id,
        "playlist": playlist_name,
        "completed": False
    }
    try:
        with open("play_history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print("[WARNING] Failed to write play_history.jsonl:", e)

def log_pick(trigger, query, source, playlist_name, title, artist, video_id):
    try:
        if not os.path.exists("logs"): os.makedirs("logs")
        entry = {
            "ts": time.time(),
            "trigger": trigger,
            "query": query,
            "source": source,
            "playlist_name": playlist_name,
            "picked": {
                "title": title,
                "artist": artist,
                "video_id": video_id
            }
        }
        with open("logs/picks.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except: pass

def parse_duration_seconds(d):
    if not d: return 0
    if isinstance(d, (int, float)): return int(d)
    try:
        parts = str(d).split(":")
        return sum(int(x) * 60 ** i for i, x in enumerate(reversed(parts)))
    except:
        return 0

# Walk State
walk_state = {
    "query": "",
    "source": "ytmusic",
    "playlist_name": "",
    "tracks": [],
    "index": 0,
    "tracks_played": 0,
    "max_tracks": 12,
    "single_song": None,
    "explicit_playlist": None
}
pick_gen = 0
walk_in_flight = False

intents = discord.Intents.default()
intents.message_content = True
bot = discord.Bot(intents=intents)

async def get_text_channel():
    if not TEXT_CHANNEL_ID:
        return None
    channel = bot.get_channel(TEXT_CHANNEL_ID)
    if channel is None:
        try:
            channel = await bot.fetch_channel(TEXT_CHANNEL_ID)
        except Exception as e:
            if DEBUG: print(f"[DEBUG] fetch_channel failed: {e}", flush=True)
            channel = None
    return channel

last_picker_message = None

async def retire_picker_message():
    global last_picker_message
    if last_picker_message is not None:
        try:
            await last_picker_message.edit(view=None)
        except Exception:
            pass
        last_picker_message = None

stt_model = None

def get_stt_model():
    global stt_model
    if stt_model is None:
        print(f"[DEBUG] Loading WhisperModel '{WHISPER_MODEL}'...")
        stt_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        print("[DEBUG] WhisperModel loaded.")
    return stt_model

utterance_queue = asyncio.Queue()
transcript_history = []
last_speech_time = time.time()

# Playback intent
playback_intent = "playing"

# Bridge variables
bridge_process = None
bridge_server = None
bridge_active = False
bridge_connected_event = asyncio.Event()
bridge_reconnecting = False
user_buffers = {}
intentional_leave = False
bridge_writer = None
bridge_futures = {}

def spawn_bridge():
    global bridge_process
    os.makedirs("logs", exist_ok=True)
    stderr_file = open("logs/bridge-stderr.log", "a", encoding="utf-8")
    bridge_process = subprocess.Popen(
        ["node", "voice-bridge/bridge.js"],
        cwd=os.getcwd(),
        stderr=stderr_file
    )

async def resume_walk_after_reconnect():
    tracks = walk_state.get("tracks", [])
    if not tracks:
        return
    idx = walk_state.get("index", 0)
    if 0 <= idx < len(tracks):
        track = tracks[idx]
        print(f"[SUPERVISOR] Resuming walk from last confirmed track [{idx+1}/{len(tracks)}]: {track.get('title')}", flush=True)
        await send_bridge_cmd({
            "cmd": "queue",
            "url": track["url"],
            "title": track["title"],
            "artist": track["artist"],
            "video_id": track.get("video_id")
        })

async def reconnect_bridge_supervisor():
    global bridge_reconnecting, bridge_process, bridge_active, bridge_server
    if bridge_reconnecting or intentional_leave:
        return
    bridge_reconnecting = True
    
    channel = bot.get_channel(TEXT_CHANNEL_ID)
    if channel:
        try:
            await channel.send("Voice bridge dropped — reconnecting…")
        except Exception:
            pass

    backoffs = [5, 15, 30]
    reconnected = False
    
    for delay in backoffs:
        if intentional_leave:
            break
        print(f"[SUPERVISOR] Bridge disconnected. Waiting {delay}s before reconnect attempt...", flush=True)
        await asyncio.sleep(delay)
        if intentional_leave:
            break
            
        bridge_connected_event.clear()
        if bridge_process:
            try:
                bridge_process.terminate()
            except Exception:
                pass
            bridge_process = None
            
        print("[SUPERVISOR] Spawning node bridge...", flush=True)
        try:
            if not bridge_server:
                bridge_server = await asyncio.start_server(handle_bridge_client, '127.0.0.1', BRIDGE_PORT)
            spawn_bridge()
            await asyncio.wait_for(bridge_connected_event.wait(), timeout=15.0)
            reconnected = True
            print("[SUPERVISOR] Bridge reconnected successfully!", flush=True)
            if channel:
                try:
                    await channel.send("🟢 Voice bridge reconnected.")
                except Exception:
                    pass
            break
        except Exception as e:
            print(f"[SUPERVISOR] Reconnect attempt failed: {e}", flush=True)
            
    if not reconnected and not intentional_leave:
        print("[SUPERVISOR] All bridge reconnect attempts exhausted.", flush=True)
        if channel:
            try:
                await channel.send("Voice bridge down.")
            except Exception:
                pass
                
    bridge_reconnecting = False
    
    if reconnected and playback_intent == "playing":
        await resume_walk_after_reconnect()

async def send_bridge_cmd(cmd_dict, wait_for_reply=False):
    if not bridge_writer or not bridge_active:
        cmd_name = cmd_dict.get("cmd", "unknown")
        print(f"[WARNING] send_bridge_cmd failed: Bridge is unreachable! Dropped command: {cmd_name}", flush=True)
        if cmd_name in ("queue", "play"):
            channel = bot.get_channel(TEXT_CHANNEL_ID)
            if channel:
                try:
                    await channel.send("Voice bridge is down — command not queued.")
                except Exception:
                    pass
        return None

    try:
        req_id = str(uuid.uuid4())
        cmd_dict["uuid"] = req_id
        fut = None
        if wait_for_reply:
            fut = asyncio.Future()
            bridge_futures[req_id] = fut

        msg = json.dumps(cmd_dict) + "\n"
        bridge_writer.write(msg.encode())
        await bridge_writer.drain()
        
        if wait_for_reply:
            try:
                res = await asyncio.wait_for(fut, timeout=3.0)
                return res
            except asyncio.TimeoutError:
                return None
            finally:
                bridge_futures.pop(req_id, None)
        return None
    except Exception as e:
        print("[CRITICAL] Failed to send bridge cmd:", e)
        return None

async def speak(text):
    if not TTS_ENABLED or not kokoro: return
    try:
        samples, sr = await asyncio.to_thread(kokoro.create, text, voice=TTS_VOICE, speed=1.1, lang="en-gb")
        upsampled = np.repeat(samples, 2)
        stereo = np.column_stack((upsampled, upsampled))
        pcm_s16le = (stereo * 32767).astype(np.int16).tobytes()
        pcm_b64 = base64.b64encode(pcm_s16le).decode('utf-8')
        await send_bridge_cmd({"cmd": "speak", "pcm_b64": pcm_b64, "n": len(pcm_s16le)})
    except Exception as e:
        print("[CRITICAL] TTS failed:", e)

async def ask_llm(prompt, model=None, format=None):
    target_model = model if model else OLLAMA_MODEL
    keep_alive_val = "2h" if bridge_active else "30m"
    async with aiohttp.ClientSession() as session:
        for attempt in range(2):
            try:
                payload = {
                    "model": target_model,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": keep_alive_val,
                    "options": {"temperature": 0.7}
                }
                if format: payload["format"] = format
                async with session.post("http://127.0.0.1:11434/api/generate", json=payload, timeout=aiohttp.ClientTimeout(total=90)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("response")
            except Exception as e:
                if DEBUG: print("[DEBUG] LLM Error:", e)
                await asyncio.sleep(1)
    return None

async def warmup_ollama():
    try:
        res = await ask_llm('Say "ready" in one word.')
        if res: print(f"[DEBUG] Ollama warmup success: {res.strip()}", flush=True)
        else: print("[WARNING] Ollama warmup returned no response.", flush=True)
    except Exception as e:
        if DEBUG: print(f"[DEBUG] Ollama warmup error: {e}", flush=True)

async def release_ollama_vram():
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": "bye",
                "stream": False,
                "keep_alive": "0"
            }
            async with session.post("http://127.0.0.1:11434/api/generate", json=payload, timeout=aiohttp.ClientTimeout(total=5)):
                pass
    except Exception:
        pass

async def ask_llm_sentiment(title, artist, playlist_name):
    global current_persona, transcript_history, bridge_active
    system_prompt = PERSONA_PROMPTS.get(current_persona, PERSONA_PROMPTS["unhinged"])
    time_str = datetime.now().strftime("%I:%M %p")
    
    recent_transcripts = transcript_history[-10:] if transcript_history else []
    transcript_lines = [f"{t['user']}: {t['text']}" for t in recent_transcripts]
    transcript_block = "\n".join(transcript_lines) if transcript_lines else "The room is quiet."
    
    prompt = f"""{system_prompt}

Current time of day: {time_str}
Now playing: "{title}" by {artist}
Playlist: "{playlist_name}"

Recent voice channel conversation:
{transcript_block}

React to the room and the music with ONE single punchy line in your persona. Do not include quotes, tags, or explanations. Just the line:"""

    keep_alive_val = "2h" if bridge_active else "30m"
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "keep_alive": keep_alive_val,
                "options": {"temperature": 0.8},
                "think": False
            }
            async with session.post("http://127.0.0.1:11434/api/generate", json=payload, timeout=aiohttp.ClientTimeout(total=90)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    response_text = data.get("response", "").strip()
                    if response_text.startswith('"') and response_text.endswith('"'):
                        response_text = response_text[1:-1].strip()
                    if response_text:
                        return response_text
                elif resp.status == 400:
                    payload.pop("think", None)
                    async with session.post("http://127.0.0.1:11434/api/generate", json=payload, timeout=aiohttp.ClientTimeout(total=90)) as resp2:
                        if resp2.status == 200:
                            data = await resp2.json()
                            response_text = data.get("response", "").strip()
                            if response_text.startswith('"') and response_text.endswith('"'):
                                response_text = response_text[1:-1].strip()
                            if response_text:
                                return response_text
    except Exception as e:
        if DEBUG: print(f"[DEBUG] Sentiment LLM failed: {e}")
    return "Vibing out."

# Playlist Resolver
async def resolve_playlist_ytmusic(query):
    try:
        res = await asyncio.to_thread(yt.search, query, filter="playlists", limit=10)
    except Exception as e:
        if DEBUG: print(f"[DEBUG] ytmusic search error: {e}")
        res = []

    candidates = []
    async def fetch_meta(r):
        pid = r.get("browseId") or r.get("playlistId")
        if not pid: return None
        try:
            p = await asyncio.to_thread(yt.get_playlist, pid, limit=1)
            tc = int(p.get("trackCount") or len(p.get("tracks", [])) or 0)
            title = p.get("title") or r.get("title") or "Unknown Playlist"
            return {"id": pid, "title": title, "track_count": tc}
        except:
            return None

    if res:
        tasks = [fetch_meta(r) for r in res[:10]]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, dict) and r.get("track_count", 0) > 0:
                candidates.append(r)

    candidates.sort(key=lambda x: x["track_count"], reverse=True)
    return candidates

async def resolve_playlist_soundcloud(query):
    try:
        cmd = ["yt-dlp", "--flat-playlist", "-J", f"scsearch10:{query} sets"]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, _ = await proc.communicate()
        data = json.loads(stdout.decode())
        entries = data.get("entries", [])
    except Exception as e:
        if DEBUG: print(f"[DEBUG] soundcloud search error: {e}")
        entries = []

    candidates = []
    for e in entries:
        tc = int(e.get("playlist_count") or e.get("track_count") or 0)
        url = e.get("webpage_url") or e.get("url")
        if url:
            candidates.append({"id": url, "title": e.get("title", "SoundCloud Playlist"), "track_count": tc, "url": url})
            
    candidates.sort(key=lambda x: x["track_count"], reverse=True)
    return candidates

async def search_single_song(query, source="ytmusic"):
    if source == "ytmusic":
        try:
            res = await asyncio.to_thread(yt.search, query, filter="songs", limit=1)
            if res:
                s = res[0]
                vid = s.get("videoId")
                title = s.get("title", "Unknown")
                artist = s.get("artists")[0]["name"] if s.get("artists") else "Unknown"
                url = f"https://youtube.com/watch?v={vid}"
                return {"title": title, "artist": artist, "video_id": vid, "url": url, "duration": s.get("duration")}
        except: pass
    try:
        cmd = ["yt-dlp", "--no-playlist", "--match-filter", "duration <= 600", "--print", "%(id)s|%(title)s|%(uploader)s|%(duration)s", f"ytsearch1:{query}"]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, _ = await proc.communicate()
        line = stdout.decode().strip().split("\n")[0]
        parts = line.split("|")
        if len(parts) >= 2 and len(parts[0]) == 11:
            vid = parts[0]
            title = parts[1]
            artist = parts[2] if len(parts) > 2 else "Unknown"
            dur = parts[3] if len(parts) > 3 else "180"
            url = f"https://youtube.com/watch?v={vid}"
            return {"title": title, "artist": artist, "video_id": vid, "url": url, "duration": dur}
    except: pass
    return None

async def fetch_playlist_tracks(playlist_id):
    try:
        p = await asyncio.to_thread(yt.get_playlist, playlist_id, limit=200)
        tracks = []
        for t in p.get("tracks", []):
            vid = t.get("videoId")
            title = t.get("title")
            if vid and title:
                artist = t.get("artists")[0]["name"] if t.get("artists") else "Unknown"
                tracks.append({
                    "title": title,
                    "artist": artist,
                    "video_id": vid,
                    "url": f"https://youtube.com/watch?v={vid}",
                    "duration": t.get("duration")
                })
        return p.get("title", "YouTube Music Playlist"), tracks
    except Exception as e:
        if DEBUG: print(f"[DEBUG] Error fetching playlist {playlist_id}: {e}")
        return "Playlist", []

async def fetch_soundcloud_tracks(playlist_url):
    try:
        cmd = ["yt-dlp", "--flat-playlist", "-J", playlist_url]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, _ = await proc.communicate()
        data = json.loads(stdout.decode())
        tracks = []
        for e in data.get("entries", []):
            u = e.get("webpage_url") or e.get("url")
            if u:
                tracks.append({
                    "title": e.get("title", "Unknown"),
                    "artist": e.get("uploader", "SoundCloud Artist"),
                    "video_id": e.get("id"),
                    "url": u,
                    "duration": e.get("duration")
                })
        return data.get("title", "SoundCloud Playlist"), tracks
    except Exception as e:
        return "SoundCloud Playlist", []

async def fetch_youtube_dlp_tracks(playlist_url):
    try:
        cmd = ["yt-dlp", "--flat-playlist", "-J", playlist_url]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, _ = await proc.communicate()
        data = json.loads(stdout.decode("utf-8", errors="replace"))
        tracks = []
        for e in data.get("entries", []):
            vid = e.get("id")
            title = e.get("title")
            if vid and title:
                tracks.append({
                    "title": title,
                    "artist": e.get("uploader", "Unknown"),
                    "video_id": vid,
                    "url": f"https://youtube.com/watch?v={vid}",
                    "duration": e.get("duration")
                })
        title = data.get("title") or "YouTube Playlist"
        return title, tracks
    except Exception as e:
        if DEBUG: print(f"[DEBUG] Error running yt-dlp for playlist {playlist_url}: {e}")
        return "YouTube Playlist", []

async def resolve_direct_playlist(url: str):
    if not url:
        return None
    url_clean = url.strip().strip('<>')
    parsed = urllib.parse.urlparse(url_clean)
    domain = parsed.netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    if domain.startswith("m."):
        domain = domain[2:]

    if domain == "soundcloud.com":
        if "/sets/" not in parsed.path:
            return None
        title, tracks = await fetch_soundcloud_tracks(url_clean)
        clean_title = title if (title and title not in ("SoundCloud Playlist",)) else (title or "")
        return {
            "source": "soundcloud",
            "ref": url_clean,
            "title": clean_title,
            "tracks": tracks,
            "start_index": None
        }

    if domain in ("youtube.com", "music.youtube.com"):
        qs = urllib.parse.parse_qs(parsed.query)
        pids = qs.get("list")
        if not pids or not pids[0]:
            return None
        playlist_id = pids[0]
        video_id = qs.get("v", [None])[0]

        title, tracks = await fetch_playlist_tracks(playlist_id)
        if not tracks:
            title, tracks = await fetch_youtube_dlp_tracks(url_clean)

        start_index = None
        if video_id and tracks:
            for idx, t in enumerate(tracks):
                if t.get("video_id") == video_id:
                    start_index = idx
                    break

        clean_title = title if (title and title not in ("Playlist", "YouTube Music Playlist", "YouTube Playlist")) else (title or "")
        return {
            "source": "ytmusic",
            "ref": playlist_id,
            "title": clean_title,
            "tracks": tracks,
            "start_index": start_index
        }

    return None

def detect_playlist_url(text: str):
    if not text:
        return None
    urls = re.findall(r'https?://[^\s<>"]+', text)
    for u in urls:
        u_clean = u.strip().strip('<>')
        try:
            parsed = urllib.parse.urlparse(u_clean)
            domain = parsed.netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]
            if domain.startswith("m."):
                domain = domain[2:]
            if domain in ("youtube.com", "music.youtube.com"):
                qs = urllib.parse.parse_qs(parsed.query)
                if "list" in qs and qs["list"] and qs["list"][0]:
                    return u_clean
            elif domain == "soundcloud.com":
                if "/sets/" in parsed.path:
                    return u_clean
        except:
            continue
    return None

async def handle_direct_playlist(url: str, ctx=None, channel=None):
    if ctx:
        await ctx.defer()
    target_channel = channel or (ctx.channel if ctx else bot.get_channel(TEXT_CHANNEL_ID))

    res = await resolve_direct_playlist(url)
    if not res or not res.get("tracks"):
        await speak("couldn't load that playlist.")
        fail_msg = "❌ couldn't load that playlist."
        if ctx:
            await ctx.followup.send(fail_msg)
        elif target_channel:
            await target_channel.send(fail_msg)
        return

    title = res.get("title", "")
    tracks = res.get("tracks", [])
    source = res.get("source", "ytmusic")
    ref = res.get("ref", "")
    start_index = res.get("start_index")

    ack_text = f"On it — {title}." if title else "On it."
    asyncio.create_task(speak(ack_text))

    send_msg = f"🎧 On it — **{title}**" if title else "🎧 On it."
    if ctx:
        await ctx.followup.send(send_msg)
    elif target_channel:
        await target_channel.send(send_msg)

    global pick_gen, playback_intent
    playback_intent = "playing"
    pick_gen += 1
    explicit_playlist = {"source": source, "ref": ref, "title": title}
    asyncio.create_task(start_walk(
        query="",
        source=source,
        gen_id=pick_gen,
        explicit_playlist=explicit_playlist,
        initial_start_index=start_index,
        prefetched_tracks=tracks
    ))

# Walk Engine
async def start_walk(query="", source="ytmusic", gen_id=0, explicit_playlist=None, initial_start_index=None, prefetched_tracks=None, interrupt=True):
    global walk_state, pick_gen, walk_in_flight
    if walk_in_flight:
        for _ in range(60):
            await asyncio.sleep(0.05)
            if not walk_in_flight or pick_gen != gen_id:
                break
        if pick_gen != gen_id:
            return
        if walk_in_flight:
            print(f"[WARN] start_walk: previous walk still in flight, bailing (query='{query}')", flush=True)
            return
    walk_in_flight = True
    
    try:
        tracks = []
        playlist_title = ""

        if explicit_playlist:
            source = explicit_playlist.get("source", source)
            ref = explicit_playlist.get("ref", "")
            playlist_title = explicit_playlist.get("title", "")
            
            tracks = prefetched_tracks or []
            if not tracks:
                if source == "soundcloud":
                    t_title, tracks = await fetch_soundcloud_tracks(ref)
                    if t_title and t_title not in ("SoundCloud Playlist",):
                        playlist_title = t_title
                    elif not playlist_title:
                        playlist_title = t_title
                else:
                    t_title, tracks = await fetch_playlist_tracks(ref)
                    if not tracks:
                        playlist_url = f"https://www.youtube.com/playlist?list={ref}" if not ref.startswith("http") else ref
                        t_title, tracks = await fetch_youtube_dlp_tracks(playlist_url)
                    if t_title and t_title not in ("Playlist", "YouTube Music Playlist", "YouTube Playlist"):
                        playlist_title = t_title
                    elif not playlist_title:
                        playlist_title = t_title

            if not tracks:
                if pick_gen != gen_id: return
                await speak("couldn't load that playlist.")
                channel = bot.get_channel(TEXT_CHANNEL_ID)
                if channel: await channel.send("❌ couldn't load that playlist.")
                return
        else:
            chosen_playlist = None
            candidates = []
            
            if source == "soundcloud":
                candidates = await resolve_playlist_soundcloud(query)
            else:
                candidates = await resolve_playlist_ytmusic(query)
                
            if len(candidates) >= 3:
                chosen = random.choice(candidates[:3])
                chosen_playlist = chosen
            
            if chosen_playlist:
                if source == "soundcloud":
                    playlist_title, tracks = await fetch_soundcloud_tracks(chosen_playlist["url"])
                else:
                    playlist_title, tracks = await fetch_playlist_tracks(chosen_playlist["id"])
                    
            # 5c. If fewer than 3 usable playlists: fall back to SONG search
            if not tracks or len(candidates) < 3:
                song = await search_single_song(query, source=source)
                if song:
                    if pick_gen != gen_id: return
                    walk_state = {
                        "query": query,
                        "source": source,
                        "playlist_name": "Direct Search",
                        "tracks": [song],
                        "index": 0,
                        "tracks_played": 1,
                        "max_tracks": 1,
                        "single_song": song,
                        "explicit_playlist": None,
                        "started_ts": time.time()
                    }
                    if interrupt:
                        await send_bridge_cmd({"cmd": "stop"})
                    await send_bridge_cmd({
                        "cmd": "queue",
                        "url": song["url"],
                        "title": song["title"],
                        "artist": song["artist"],
                        "video_id": song.get("video_id")
                    })
                    log_pick("echo", query, source, "Direct Search", song["title"], song["artist"], song.get("video_id"))
                    return
                else:
                    # 5d. Song search also fails: TTS "couldn't find <query>" and keep current walk
                    await speak(f"Couldn't find {query}.")
                    channel = bot.get_channel(TEXT_CHANNEL_ID)
                    if channel: await channel.send(f"❌ Couldn't find {query}.")
                    return

        # 6a. Start at a RANDOM offset (never track 1 when tracks > 1, unless explicit with initial_start_index)
        if explicit_playlist and initial_start_index is not None and 0 <= initial_start_index < len(tracks):
            start_offset = initial_start_index
        elif len(tracks) > 1:
            start_offset = random.randint(1, len(tracks) - 1)
        else:
            start_offset = 0

        final_playlist_name = playlist_title or (explicit_playlist.get("title") if explicit_playlist else "") or query.title()

        walk_state = {
            "query": query,
            "source": source,
            "playlist_name": final_playlist_name,
            "tracks": tracks,
            "index": start_offset,
            "tracks_played": 0,
            "max_tracks": random.randint(10, 15),
            "single_song": None,
            "explicit_playlist": explicit_playlist,
            "started_ts": time.time()
        }
        
        # 6c. DEDUP: check video ID against recent_played_vids
        track = None
        for i in range(len(tracks)):
            idx = (start_offset + i) % len(tracks)
            candidate = tracks[idx]
            vid = candidate.get("video_id")
            if not vid or vid not in recent_played_vids:
                track = candidate
                walk_state["index"] = idx
                break
                
        if not track:
            track = tracks[start_offset]

        if pick_gen != gen_id: return

        # Stop old playback and queue new track immediately if interrupting
        if interrupt:
            await send_bridge_cmd({"cmd": "stop"})
        await send_bridge_cmd({
            "cmd": "queue",
            "url": track["url"],
            "title": track["title"],
            "artist": track["artist"],
            "video_id": track.get("video_id")
        })
        log_pick("echo", query or final_playlist_name, source, walk_state["playlist_name"], track["title"], track["artist"], track.get("video_id"))
            
    except Exception as e:
        print(f"[ERROR] start_walk failed: {e}", flush=True)
        try:
            if explicit_playlist:
                await speak("couldn't load that playlist.")
                channel = bot.get_channel(TEXT_CHANNEL_ID)
                if channel: await channel.send("❌ couldn't load that playlist.")
            else:
                await speak(f"Couldn't find {query}.")
        except Exception:
            pass
    finally:
        walk_in_flight = False

async def advance_walk(trigger="keep_ahead", gen_id=0):
    global walk_state, pick_gen
    if pick_gen != gen_id: return
    if not bridge_active: return
    
    tracks = walk_state.get("tracks", [])
    if not tracks:
        await re_roll_walk(gen_id=gen_id)
        return

    if walk_state.get("tracks_played", 0) >= walk_state.get("max_tracks", 12):
        await re_roll_walk(gen_id=gen_id)
        return

    start_idx = (walk_state.get("index", 0) + 1) % len(tracks)
    track = None
    chosen_idx = start_idx
    for i in range(len(tracks)):
        idx = (start_idx + i) % len(tracks)
        candidate = tracks[idx]
        vid = candidate.get("video_id")
        if not vid or vid not in recent_played_vids:
            track = candidate
            chosen_idx = idx
            break
            
    if not track:
        track = tracks[start_idx]
        chosen_idx = start_idx

    walk_state["queued_index"] = chosen_idx

    if pick_gen != gen_id: return
    await send_bridge_cmd({
        "cmd": "queue",
        "url": track["url"],
        "title": track["title"],
        "artist": track["artist"],
        "video_id": track.get("video_id")
    })
    log_pick(trigger, walk_state.get("query", ""), walk_state.get("source", "ytmusic"), walk_state.get("playlist_name", ""), track["title"], track["artist"], track.get("video_id"))

async def re_roll_walk(gen_id=0):
    global walk_state, pick_gen
    if pick_gen != gen_id: return
    if not bridge_active: return
    
    explicit = walk_state.get("explicit_playlist")
    if explicit:
        await start_walk(query="", source=explicit.get("source", "ytmusic"), gen_id=gen_id, explicit_playlist=explicit, interrupt=False)
        return

    query = walk_state.get("query")
    source = walk_state.get("source", "ytmusic")
    
    if query:
        await start_walk(query, source=source, gen_id=gen_id, interrupt=False)
    else:
        pid, name = get_curated_playlist()
        if pid:
            playlist_title, tracks = await fetch_playlist_tracks(pid)
            if tracks:
                offset = random.randint(1, len(tracks) - 1) if len(tracks) > 1 else 0
                walk_state = {
                    "query": "",
                    "source": "ytmusic",
                    "playlist_name": playlist_title or name,
                    "tracks": tracks,
                    "index": offset,
                    "tracks_played": 0,
                    "max_tracks": random.randint(10, 15),
                    "single_song": None,
                    "explicit_playlist": None,
                    "started_ts": time.time()
                }
                track = tracks[offset]
                await send_bridge_cmd({
                    "cmd": "queue",
                    "url": track["url"],
                    "title": track["title"],
                    "artist": track["artist"],
                    "video_id": track.get("video_id")
                })
                log_pick("autoplay", "curated", "ytmusic", walk_state["playlist_name"], track["title"], track["artist"], track.get("video_id"))

async def check_and_queue_ahead(gen_id):
    if not bridge_active or pick_gen != gen_id:
        return
    res = await send_bridge_cmd({"cmd": "status"}, wait_for_reply=True)
    if not res or pick_gen != gen_id:
        return
    if res.get("queue", 0) == 0:
        await advance_walk(trigger="keep_ahead", gen_id=gen_id)

async def fetch_lyrics(title: str, artist: str) -> str | None:
    try:
        cleaned_title = re.sub(
            r'[\(\[][^\)\]]*(?:official|video|audio|lyric|visualizer|feat|ft\.|remaster|version)[^\)\]]*[\)\]]',
            '',
            title,
            flags=re.IGNORECASE
        ).strip(" -–—|")
        if not cleaned_title:
            cleaned_title = title.strip()

        params = {
            "track_name": cleaned_title,
            "artist_name": artist
        }
        headers = {
            "User-Agent": "ECHO-DJ/0.1 (local Discord bot)"
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://lrclib.net/api/search",
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                if not isinstance(data, list):
                    return None
                for entry in data:
                    if not entry.get("instrumental", False):
                        lyrics = entry.get("plainLyrics")
                        if lyrics and lyrics.strip():
                            return lyrics
                return None
    except Exception:
        return None

async def _finalize_now_playing(message, thread, title, artist, playlist_name, video_id, base_desc=None):
    if message is None:
        return

    # 1. Start lyrics fetch concurrently
    lyrics_task = asyncio.create_task(fetch_lyrics(title, artist))

    # 2. Generate persona sentiment (never waits on lyrics)
    sentiment_line = ""
    try:
        t0 = time.time()
        sentiment_line = await ask_llm_sentiment(title, artist, playlist_name)
        elapsed = time.time() - t0
        print(f"[{current_persona.upper()}] sentiment generated in {elapsed:.2f}s: {sentiment_line}", flush=True)
    except Exception as e:
        print(f"[SENTIMENT] [{title}] sentiment generation failed: {e}", flush=True)
        traceback.print_exc()

    # 3. Build embed WITHOUT the one-liner (description = base_desc; large image preserved)
    if base_desc is None and getattr(message, "embeds", None):
        base_desc = message.embeds[0].description or ""

    embed = discord.Embed(color=discord.Color.blurple())
    embed.description = base_desc
    if video_id:
        embed.set_image(url=f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg")
    elif getattr(message, "embeds", None) and message.embeds[0].image and message.embeds[0].image.url:
        embed.set_image(url=message.embeds[0].image.url)

    # 4. First edit: emoji-led content, embed=embed
    emoji = PERSONA_EMOJI.get(current_persona, "🎧")
    content = f"{emoji} {sentiment_line}" if sentiment_line else emoji
    msg_id = getattr(message, "id", None)
    try:
        print(f"[SENTIMENT] [{title}] editing message id={msg_id}...", flush=True)
        await message.edit(content=content, embed=embed)
        print(f"[SENTIMENT] [{title}] message id={msg_id} edit SUCCESS: sentiment content set.", flush=True)
    except Exception as e:
        print(f"[SENTIMENT] [{title}] message id={msg_id} edit FAILED: {e}", flush=True)
        traceback.print_exc()

    # 5. Await lyrics fetch
    lyrics = None
    try:
        lyrics = await lyrics_task
    except Exception as e:
        if DEBUG: print(f"[DEBUG] lyrics fetch task failed: {e}", flush=True)
        lyrics = None

    # 6. Handle lyrics result
    if lyrics:
        desc = lyrics
        if len(desc) > 4000:
            target_limit = 4000 - 2
            last_nl = desc.rfind("\n", 0, target_limit)
            if last_nl != -1:
                desc = desc[:last_nl].rstrip() + "\n…"
            else:
                desc = desc[:target_limit].rstrip() + "…"

        lyrics_embed = discord.Embed(
            title=f"📜 Lyrics — {title} — {artist}",
            description=desc,
            color=discord.Color.blurple()
        )
        sent_to_thread = False
        if thread:
            try:
                await thread.send(embed=lyrics_embed)
                sent_to_thread = True
            except Exception as te:
                if DEBUG: print(f"[DEBUG] lyrics thread send failed, falling back to channel: {te}", flush=True)

        if not sent_to_thread:
            channel = await get_text_channel()
            if channel:
                try:
                    await channel.send(embed=lyrics_embed)
                except Exception as ce:
                    if DEBUG: print(f"[DEBUG] lyrics channel send failed: {ce}", flush=True)
    else:
        print(f"[LYRICS] none found for {title} — {artist}", flush=True)
        if thread:
            try:
                await thread.delete()
            except Exception as e:
                if DEBUG: print(f"[DEBUG] lyrics thread delete failed: {e}", flush=True)

        new_content = f"{content} · no lyrics found 🎵"
        try:
            print(f"[LYRICS] [{title}] editing message id={msg_id} with miss indicator...", flush=True)
            await message.edit(content=new_content)
            print(f"[LYRICS] [{title}] message id={msg_id} edit SUCCESS: miss indicator appended.", flush=True)
        except Exception as e:
            print(f"[LYRICS] [{title}] message id={msg_id} edit FAILED: {e}", flush=True)
            traceback.print_exc()

async def announce_track_start(title, artist, url, video_id):
    global current_persona, walk_state
    
    # Snapshot walk state synchronously so the embed reflects this track.
    playlist_name = walk_state.get("playlist_name") or "Curated"
    track_idx = (walk_state.get("index", 0)) + 1
    total_tracks = len(walk_state.get("tracks", []))
    track_meta = f"track {track_idx}/{total_tracks}" if total_tracks > 1 else "track 1/1"
    
    song_url = url if url else f"https://youtube.com/watch?v={video_id}"
    base_desc = (
        f"Now playing: [{title} — {artist}]({song_url})\n"
        f'via "{playlist_name}" · {track_meta}'
    )
    
    # Now-playing embed is time-critical: send it first, before any LLM work.
    message = None
    channel = await get_text_channel()
    if channel:
        try:
            embed = discord.Embed(color=discord.Color.blurple())
            embed.description = base_desc
            if video_id:
                embed.set_image(url=f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg")
            message = await channel.send(embed=embed)
        except Exception as e:
            if DEBUG: print(f"[DEBUG] now-playing embed send failed: {e}", flush=True)
    
    if message:
        thread = None
        try:
            thread = await message.create_thread(name=f"Lyrics: {title}"[:90], auto_archive_duration=60)
        except Exception as e:
            if DEBUG: print(f"[DEBUG] lyrics create_thread failed: {e}", flush=True)
            thread = None

        asyncio.create_task(_finalize_now_playing(message, thread, title, artist, playlist_name, video_id, base_desc=base_desc))

async def handle_back_command(channel=None, ctx=None):
    global walk_state, pick_gen, playback_intent
    tracks = walk_state.get("tracks", [])
    if not tracks:
        msg = "Nothing before this."
        if ctx:
            await ctx.followup.send(f"❌ {msg}")
        else:
            asyncio.create_task(speak(msg))
            if channel:
                await channel.send(f"❌ {msg}")
        return False

    current_idx = walk_state.get("index", 0)
    prev_idx = (current_idx - 1) % len(tracks)
    walk_state["index"] = prev_idx
    prev_track = tracks[prev_idx]

    playback_intent = "playing"
    pick_gen += 1

    await send_bridge_cmd({"cmd": "stop"})
    await send_bridge_cmd({
        "cmd": "queue",
        "url": prev_track["url"],
        "title": prev_track["title"],
        "artist": prev_track["artist"],
        "video_id": prev_track.get("video_id")
    })
    log_pick("back", walk_state.get("query", ""), walk_state.get("source", "ytmusic"), walk_state.get("playlist_name", ""), prev_track["title"], prev_track["artist"], prev_track.get("video_id"))

    if ctx:
        await ctx.followup.send("Going back to previous track ⏪")
    else:
        asyncio.create_task(speak("Going back."))
        if channel:
            await channel.send("⏪ Playing previous track.")
    return True

def parse_voice_command(text):
    wake_matches = list(re.finditer(r'(hey\s+)?(echo|ecko|eco)[,\s]*', text, flags=re.IGNORECASE))
    wake_text = text[wake_matches[-1].end():] if wake_matches else text
    clean = wake_text.strip()
    clean_norm = clean.strip(" .,!?:;'\"")
    
    if re.search(r'^(skip|next)\b', clean_norm, re.IGNORECASE):
        return {"action": "skip", "query": None, "source": None}

    if re.search(r'^(back|previous)\b', clean_norm, re.IGNORECASE):
        return {"action": "back", "query": None, "source": None}

    if re.search(r'^pause\b', clean_norm, re.IGNORECASE):
        return {"action": "pause", "query": None, "source": None}

    if re.search(r'^(resume|unpause|continue)\b', clean_norm, re.IGNORECASE):
        return {"action": "resume", "query": None, "source": None}

    if re.search(r'^stop\b', clean_norm, re.IGNORECASE):
        return {"action": "stop", "query": None, "source": None}

    if re.search(r'^(what\'?s\s+playing|what\s+is\s+(this|playing))', clean_norm, re.IGNORECASE):
        return {"action": "whats_playing", "query": None, "source": None}

    if re.search(r'^(play\s+(a\s+)?mix|radio|start\s+(a\s+)?radio|autoplay)\b', clean_norm, re.IGNORECASE):
        return {"action": "radio", "query": None, "source": None}

    play_match = re.match(
        r'^(play|put\s+on|throw\s+on|spin|drop|queue|start)\s+(.*)',
        clean_norm,
        re.IGNORECASE
    )
    if play_match:
        remainder = play_match.group(2).strip()
        query = re.sub(
            r'^(some|me\s+some|a\s+bit\s+of|a\s+little(\s+bit\s+of)?|us\s+some)\s+',
            '',
            remainder,
            flags=re.IGNORECASE
        ).strip()
        
        query = re.split(r'[.!?]+', query)[0].strip()
        
        source = "ytmusic"
        if re.search(r'\bon\s+soundcloud\b', query, re.IGNORECASE):
            source = "soundcloud"
            query = re.sub(r'\bon\s+soundcloud\b', '', query, flags=re.IGNORECASE).strip()

        query = query.strip(" .,!?:;'\"")
        if query:
            return {"action": "play", "query": query, "source": source}

    return {"action": "unparseable", "query": None, "source": None}

def is_play_debounced(query, walk_state, now=None):
    if not query:
        return False
    if now is None:
        now = time.time()
    norm = re.sub(r"\s+", " ", query.strip().lower())
    cur = re.sub(r"\s+", " ", str(walk_state.get("query", "")).strip().lower())
    if norm and norm == cur and walk_state.get("tracks") and (now - walk_state.get("started_ts", 0)) < 30:
        return True
    return False

# Voice Command Parser
async def handle_voice_command(text, user_id):
    global current_persona, pick_gen, playback_intent
    
    parsed = parse_voice_command(text)
    action = parsed["action"]
    query = parsed["query"]
    source = parsed["source"]
    
    channel = bot.get_channel(TEXT_CHANNEL_ID)
    
    if action == "skip":
        playback_intent = "playing"
        await send_bridge_cmd({"cmd": "skip"})
        asyncio.create_task(speak("Skipping."))
        if channel: await channel.send("⏭️ Skipping track.")
        return

    if action == "back":
        await handle_back_command(channel=channel)
        return

    if action == "pause":
        await send_bridge_cmd({"cmd": "pause"})
        asyncio.create_task(speak("Paused."))
        if channel: await channel.send("⏸️ Paused.")
        return

    if action == "resume":
        playback_intent = "playing"
        await send_bridge_cmd({"cmd": "resume"})
        asyncio.create_task(speak("Resuming."))
        if channel: await channel.send("▶️ Resumed.")
        return

    if action == "stop":
        playback_intent = "stopped"
        await send_bridge_cmd({"cmd": "stop"})
        asyncio.create_task(speak("Stopped."))
        if channel: await channel.send("⏹️ Stopped.")
        return

    if action == "whats_playing":
        bridge_data = await send_bridge_cmd({"cmd": "status"}, wait_for_reply=True)
        if bridge_data and bridge_data.get("playing"):
            p = bridge_data["playing"]
            reply = f"That's {p['title']} by {p['artist']}."
            asyncio.create_task(speak(reply))
            if channel: await channel.send(f"🎵 {reply}")
        else:
            asyncio.create_task(speak("Nothing is playing right now."))
            if channel: await channel.send("🎵 Nothing is playing right now.")
        return

    if action == "radio":
        playback_intent = "playing"
        pick_gen += 1
        asyncio.create_task(speak("Radio mode on."))
        if channel: await channel.send("📻 Rolling the radio mix...")
        asyncio.create_task(re_roll_walk(gen_id=pick_gen))
        return

    if action == "play":
        if is_play_debounced(query, walk_state):
            await speak("already on it")
            return

        playback_intent = "playing"
        ack_msg = get_tts_ack(query, current_persona)
        asyncio.create_task(speak(ack_msg))
        if channel:
            await channel.send(f"👂 {ack_msg}")
        
        pick_gen += 1
        asyncio.create_task(start_walk(query=query, source=source, gen_id=pick_gen))
        return

    # Unparseable
    unparseable_msg = get_tts_unparseable(current_persona)
    asyncio.create_task(speak(unparseable_msg))
    if channel:
        await channel.send(f"🤖 {unparseable_msg}")

# Audio VAD & Buffer
class UserAudioBuffer:
    def __init__(self, user_id):
        self.user_id = user_id
        self.pcm_buffer = bytearray()
        self.utterance_buffer = bytearray()
        self.vad = webrtcvad.Vad(3)
        self.silence_frames = 0
        self.is_speaking = False
        self.audio_first_packet = 0
        self.vad_speech_start = 0

    def add_data(self, data, queue):
        if not self.pcm_buffer and not self.is_speaking:
            self.audio_first_packet = time.time()
            
        self.pcm_buffer.extend(data)
        while len(self.pcm_buffer) >= 960:
            frame = bytes(self.pcm_buffer[:960])
            self.pcm_buffer = self.pcm_buffer[960:]
            try: is_speech = self.vad.is_speech(frame, 16000)
            except: continue

            if is_speech:
                if not self.is_speaking:
                    self.is_speaking = True
                    self.vad_speech_start = time.time()
                self.utterance_buffer.extend(frame)
                self.silence_frames = 0
            else:
                if self.is_speaking:
                    self.utterance_buffer.extend(frame)
                    self.silence_frames += 1
                    if self.silence_frames > 25:
                        self.is_speaking = False
                        vad_endpoint = time.time()
                        queue.put_nowait((self.user_id, bytes(self.utterance_buffer), self.audio_first_packet, self.vad_speech_start, vad_endpoint))
                        self.utterance_buffer = bytearray()
                        self.silence_frames = 0
                        self.audio_first_packet = 0

def do_transcribe(audio_np):
    try:
        model = get_stt_model()
        segments, info = model.transcribe(audio_np, beam_size=1, language=SPOKEN_LANGUAGE, initial_prompt="Echo")
        text = " ".join(segment.text for segment in segments if segment.no_speech_prob < 0.5).strip()
        return text
    except: return ""

async def transcribe_worker():
    global last_speech_time
    while True:
        try:
            user_id, audio_bytes, t_first, t_start, t_end = await utterance_queue.get()
            if len(audio_bytes) < 16000 * 0.5 * 2: continue
            
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            t_whisper0 = time.time()
            text = await asyncio.to_thread(do_transcribe, audio_np)
            t_emit = time.time()
            print(f"[DEBUG-LATENCY] {user_id} | endpoint -> emit: {t_emit - t_end:.2f}s (whisper: {t_emit - t_whisper0:.2f}s)", flush=True)
            
            if text:
                last_speech_time = time.time()
                user = bot.get_user(user_id)
                name = user.display_name if user else f"User {user_id}"
                transcript_history.append({'time': time.time(), 'user': name, 'text': text})
                if len(transcript_history) > 50:
                    transcript_history.pop(0)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {name}: {text}", flush=True)
                
                if re.search(r'\b(echo|ecko|eco)\b', text, re.IGNORECASE):
                    asyncio.create_task(handle_voice_command(text, user_id))
        except Exception as e:
            print("[CRITICAL] transcribe_worker error:")
            traceback.print_exc()

# Bridge TCP Client Handler
async def handle_bridge_client(reader, writer):
    global bridge_active, bridge_writer, pick_gen
    bridge_active = True
    bridge_writer = writer
    bridge_connected_event.set()
    try:
        while True:
            line = await reader.readline()
            if not line: break
            try:
                header_str = line.decode('utf-8').strip()
                if not header_str: continue
                header = json.loads(header_str)
                
                if "uuid" in header:
                    req_id = header["uuid"]
                    fut = bridge_futures.get(req_id)
                    if fut and not fut.done(): fut.set_result(header)
                    if "event" not in header: continue

                if "event" in header:
                    evt = header["event"]
                    if evt == "track_start":
                        title = header.get('title', '')
                        artist = header.get('artist', '')
                        url = header.get('url', '')
                        vid = header.get('video_id')
                        
                        # Self-heal walk index from video_id
                        tracks = walk_state.get("tracks", [])
                        if tracks:
                            found_idx = None
                            if vid:
                                for idx, t in enumerate(tracks):
                                    if t.get("video_id") == vid:
                                        found_idx = idx
                                        break
                            if found_idx is not None:
                                walk_state["index"] = found_idx
                            else:
                                walk_state["index"] = (walk_state.get("index", 0) + 1) % len(tracks)

                        walk_state["tracks_played"] = walk_state.get("tracks_played", 0) + 1
                        
                        log_play_history(title, artist, url, vid, walk_state.get("playlist_name", ""))
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] ▶️ Now playing: {title} — {artist} ({vid or 'no-id'})", flush=True)
                        asyncio.create_task(announce_track_start(title, artist, url, vid))
                        asyncio.create_task(check_and_queue_ahead(pick_gen))
                    elif evt == "queue_empty":
                        player_state = header.get("player", "idle")
                        if playback_intent == "stopped":
                            if DEBUG: print("[DEBUG] Ignoring queue_empty: playback_intent == 'stopped'", flush=True)
                        elif player_state != "playing":
                            pick_gen += 1
                            asyncio.create_task(advance_walk(trigger="queue_empty", gen_id=pick_gen))
                    elif evt == "queue_ack":
                        status = header.get("status")
                        reason = header.get("reason", "")
                        if status == "rejected":
                            print(f"[BRIDGE] Queue rejected track: {reason}", flush=True)
                        elif DEBUG:
                            print(f"[DEBUG] Bridge queue ack: ok", flush=True)
                    continue

                user_id = int(header["u"])
                n = int(header["n"])
            except Exception as e:
                if DEBUG: print(f"[DEBUG] handle_bridge_client packet error: {e}", flush=True)
                continue

            data = await reader.readexactly(n)
            if user_id not in user_buffers: user_buffers[user_id] = UserAudioBuffer(user_id)
            user_buffers[user_id].add_data(data, utterance_queue)
    except Exception as e:
        print("[CRITICAL] handle_bridge_client died:")
        traceback.print_exc()
    finally:
        bridge_active = False
        bridge_writer = None
        writer.close()
        if not intentional_leave:
            asyncio.create_task(reconnect_bridge_supervisor())

# Persona UI Components
class PersonaSelect(discord.ui.Select):
    def __init__(self):
        options = [
            discord.SelectOption(label="Unhinged", value="unhinged", description="Feral gremlin DJ with no filter", emoji="🤪"),
            discord.SelectOption(label="Velvet", value="velvet", description="Deadpan late-night host, quietly unimpressed", emoji="🍸"),
            discord.SelectOption(label="Hype", value="hype", description="Catastrophically over-excited hype-man", emoji="🔥"),
        ]
        super().__init__(custom_id="echo_persona_select", placeholder="Choose Echo's persona...", min_values=1, max_values=1, options=options)

    async def callback(self, interaction: discord.Interaction):
        global current_persona, last_picker_message
        if not self.values:
            return
        current_persona = self.values[0]
        name = current_persona.capitalize()
        print(f"[PERSONA] switched to {current_persona}", flush=True)
        await interaction.response.edit_message(content=f"Persona locked in: **{name}**.", view=None)

        if last_picker_message and last_picker_message.id != interaction.message.id:
            try:
                await last_picker_message.edit(view=None)
            except Exception:
                pass
            last_picker_message = None

        channel = await get_text_channel()
        if channel is None:
            channel = interaction.channel
        if channel:
            try:
                new_msg = await channel.send(f"Persona switched to: **{name}** — want to change again?", view=PersonaView())
                last_picker_message = new_msg
            except Exception as e:
                print(f"[PERSONA] failed to send new picker: {e}", flush=True)

class PersonaView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)
        self.add_item(PersonaSelect())

# Slash Commands
@bot.slash_command(name="join", description="Join the configured voice channel")
async def join(ctx=None):
    if ctx:
        await ctx.defer()
    global bridge_process, bridge_server, bridge_active, intentional_leave, pick_gen, playback_intent, last_picker_message
    intentional_leave = False
    playback_intent = "playing"
    channel = await get_text_channel()
    if bridge_active:
        msg = "Already joined and listening via Node bridge! 🎧"
        if ctx:
            last_picker_message = await ctx.followup.send(msg, view=PersonaView())
            return last_picker_message
        elif channel:
            last_picker_message = await channel.send(msg, view=PersonaView())
            return last_picker_message
        return
    if not bridge_server:
        bridge_server = await asyncio.start_server(handle_bridge_client, '127.0.0.1', BRIDGE_PORT)
    bridge_connected_event.clear()
    try:
        spawn_bridge()
    except Exception as e:
        msg = f"Failed to start Node bridge: {e}"
        if ctx: return await ctx.followup.send(msg)
        elif channel: return await channel.send(msg)
        return
    try:
        await asyncio.wait_for(bridge_connected_event.wait(), timeout=15.0)
        msg = "Joined and listening via Node bridge! 🎧 Pick a persona:"
        if ctx:
            last_picker_message = await ctx.followup.send(msg, view=PersonaView())
        elif channel:
            last_picker_message = await channel.send(msg, view=PersonaView())
        asyncio.create_task(warmup_ollama())
        if JOIN_AUTOPLAY:
            pick_gen += 1
            asyncio.create_task(re_roll_walk(gen_id=pick_gen))
    except asyncio.TimeoutError:
        if bridge_process: bridge_process.terminate()
        msg = "Failed to join: Bridge did not connect in time."
        if ctx: await ctx.followup.send(msg)
        elif channel: await channel.send(msg)

@bot.slash_command(name="leave", description="Leave the voice channel")
async def leave(ctx=None):
    global bridge_process, bridge_server, bridge_active, intentional_leave, playback_intent
    if ctx:
        await ctx.defer()
    intentional_leave = True
    playback_intent = "stopped"
    await send_bridge_cmd({"cmd": "stop"})
    if bridge_process: bridge_process.terminate(); bridge_process = None
    if bridge_server: bridge_server.close(); await bridge_server.wait_closed(); bridge_server = None
    bridge_active = False
    asyncio.create_task(release_ollama_vram())
    msg = "Left the channel."
    if ctx: await ctx.followup.send(msg)
    else:
        channel = bot.get_channel(TEXT_CHANNEL_ID)
        if channel: await channel.send(msg)

@bot.slash_command(name="skip", description="Skip current track")
async def skip(ctx):
    global playback_intent
    await ctx.defer()
    playback_intent = "playing"
    await send_bridge_cmd({"cmd": "skip"})
    await ctx.followup.send("Skipped current track ⏭️")

@bot.slash_command(name="back", description="Play previous track")
async def back(ctx):
    await ctx.defer()
    await handle_back_command(ctx=ctx)

@bot.slash_command(name="pause", description="Pause playback")
async def pause(ctx):
    await ctx.defer()
    await send_bridge_cmd({"cmd": "pause"})
    await ctx.followup.send("Paused playback ⏸️")

@bot.slash_command(name="resume", description="Resume playback")
async def resume(ctx):
    global playback_intent
    await ctx.defer()
    playback_intent = "playing"
    await send_bridge_cmd({"cmd": "resume"})
    await ctx.followup.send("Resumed playback ▶️")

@bot.slash_command(name="stop", description="Stop playback and clear queue")
async def stop(ctx):
    global playback_intent
    await ctx.defer()
    playback_intent = "stopped"
    await send_bridge_cmd({"cmd": "stop"})
    await ctx.followup.send("Playback stopped ⏹️")

@bot.slash_command(name="persona", description="Switch Echo's personality")
async def persona_cmd(ctx):
    global last_picker_message
    await ctx.defer()
    await retire_picker_message()
    msg = await ctx.followup.send("Choose Echo's persona:", view=PersonaView())
    last_picker_message = msg

@bot.slash_command(name="queue", description="Show current queue and now playing")
async def queue(ctx):
    await ctx.defer()
    res = await send_bridge_cmd({"cmd": "status"}, wait_for_reply=True)
    if not res: return await ctx.followup.send("Failed to get status from bridge.")
    p = res.get("playing")
    up_next = res.get("up_next", [])
    paused = res.get("paused", False)
    if not p and not up_next: return await ctx.followup.send("The queue is empty. 🎧")
    embed = discord.Embed(color=discord.Color.blurple())
    if p:
        state_icon = "⏸️" if paused else "▶️"
        embed.title = f"{state_icon} Now playing: {p['title']} — {p['artist']}"
    else: embed.title = "🎧 Nothing currently playing"
    if up_next:
        desc = "**⏭ Up next:**\n"
        for i, t in enumerate(up_next[:10], 1): desc += f"{i}. {t['title']} — {t['artist']}\n"
        queue_len = res.get("queue", 0)
        if queue_len > 10: desc += f"\n*+ {queue_len - 10} more*"
        embed.description = desc
    await ctx.followup.send(embed=embed)

@bot.slash_command(name="status", description="Current Echo status and walk info")
async def status(ctx):
    await ctx.defer()
    res = await send_bridge_cmd({"cmd": "status"}, wait_for_reply=True)
    msg = f"**Persona:** {current_persona.capitalize()}\n"
    msg += f"**Voice Bridge:** {'Connected 🟢' if bridge_active else 'Disconnected 🔴'}\n"
    msg += f"**Active Walk:** {walk_state.get('playlist_name', 'None')} ({walk_state.get('source', 'ytmusic')})\n"
    if res and res.get("playing"):
        p = res["playing"]
        state_str = " (Paused)" if res.get("paused") else ""
        msg += f"**Now Playing:** {p['title']} — {p['artist']}{state_str}\n"
        msg += f"**Queue:** {res.get('queue', 0)} tracks"
    else:
        msg += "**Now Playing:** None\n**Queue:** Empty"
    await ctx.followup.send(msg)

@bot.slash_command(name="playlist", description="Play a direct playlist from YouTube or SoundCloud")
async def playlist_cmd(ctx, url: str):
    global playback_intent
    playback_intent = "playing"
    await handle_direct_playlist(url, ctx=ctx)

@bot.event
async def on_message(message):
    if message.author == bot.user or message.author.bot:
        return
    if message.channel.id != TEXT_CHANNEL_ID:
        return
    url = detect_playlist_url(message.content)
    if url:
        await handle_direct_playlist(url, channel=message.channel)

async def command_file_watcher():
    global current_persona
    cmd_file = "bot_cmd.txt"
    while True:
        try:
            if os.path.exists(cmd_file):
                with open(cmd_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                try:
                    os.remove(cmd_file)
                except Exception:
                    pass
                if content:
                    print(f"[CMD-PIPE] Received command: {content}", flush=True)
                    if content == "join":
                        await join(ctx=None)
                    elif content == "leave":
                        await leave(ctx=None)
                    elif content.startswith("persona "):
                        p = content[8:].strip().lower()
                        if p in PERSONA_EMOJI:
                            current_persona = p
                            print(f"[PERSONA] switched to {current_persona}", flush=True)
                    elif content.startswith("queue "):
                        url = content[6:].strip()
                        await send_bridge_cmd({"cmd": "queue", "url": url, "title": "Test Track", "artist": "Test Artist", "video_id": "test_vid"})
                    elif content.startswith("direct_playlist "):
                        url = content[16:].strip()
                        await handle_direct_playlist(url)
                    elif content.startswith("Echo") or content.startswith("echo") or content.startswith("hey echo"):
                        await handle_voice_command(content, 0)
        except Exception as e:
            if DEBUG: print(f"[DEBUG] command_file_watcher error: {e}", flush=True)
        await asyncio.sleep(0.5)

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")
    
    bot.add_view(PersonaView())
    print("[DEBUG] Warming up Ollama...")
    bot.loop.create_task(warmup_ollama())
    get_stt_model()
    bot.loop.create_task(transcribe_worker())
    bot.loop.create_task(command_file_watcher())

if __name__ == "__main__":
    if not os.path.exists("logs"): os.makedirs("logs")
    if DISCORD_TOKEN: bot.run(DISCORD_TOKEN)
    else: print("Error: DISCORD_TOKEN not found in .env")
