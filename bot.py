import os
import json
import time
import asyncio
import subprocess
import traceback
import logging
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
    "single_song": None
}
pick_gen = 0
walk_in_flight = False

intents = discord.Intents.default()
bot = discord.Bot(intents=intents)

print(f"[DEBUG] Loading WhisperModel '{WHISPER_MODEL}'...")
stt_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
print("[DEBUG] WhisperModel loaded.")

utterance_queue = asyncio.Queue()
transcript_history = []
last_speech_time = time.time()

# Bridge variables
bridge_process = None
bridge_server = None
bridge_active = False
bridge_connected_event = asyncio.Event()
user_buffers = {}
intentional_leave = False
bridge_writer = None
bridge_futures = {}

async def send_bridge_cmd(cmd_dict, wait_for_reply=False):
    if bridge_writer and bridge_active:
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
    async with aiohttp.ClientSession() as session:
        for attempt in range(2):
            try:
                payload = {
                    "model": target_model,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": "30m",
                    "options": {"temperature": 0.7}
                }
                if format: payload["format"] = format
                async with session.post("http://127.0.0.1:11434/api/generate", json=payload, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("response")
            except Exception as e:
                if DEBUG: print("[DEBUG] LLM Error:", e)
                await asyncio.sleep(1)
    return None

async def ask_llm_sentiment(title, artist, playlist_name):
    global current_persona, transcript_history
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

    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "30m",
                "options": {"temperature": 0.8}
            }
            async with session.post("http://127.0.0.1:11434/api/generate", json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
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

# Walk Engine
async def prefetch_task(duration_sec, gen_id):
    global pick_gen
    await asyncio.sleep(max(0, duration_sec - 45))
    if pick_gen == gen_id:
        await advance_walk(trigger="prefetch", gen_id=pick_gen)

async def start_walk(query, source="ytmusic", gen_id=0):
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
        chosen_playlist = None
        candidates = []
        
        if source == "soundcloud":
            candidates = await resolve_playlist_soundcloud(query)
        else:
            candidates = await resolve_playlist_ytmusic(query)
            
        if len(candidates) >= 3:
            chosen = random.choice(candidates[:3])
            chosen_playlist = chosen
        
        tracks = []
        playlist_title = ""
        
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
                    "single_song": song
                }
                await send_bridge_cmd({"cmd": "stop"})
                await send_bridge_cmd({"cmd": "queue", "url": song["url"], "title": song["title"], "artist": song["artist"]})
                log_pick("echo", query, source, "Direct Search", song["title"], song["artist"], song.get("video_id"))
                d_sec = parse_duration_seconds(song.get("duration", 180))
                if d_sec > 45:
                    bot.loop.create_task(prefetch_task(d_sec, pick_gen))
                return
            else:
                # 5d. Song search also fails: TTS "couldn't find <query>" and keep current walk
                await speak(f"Couldn't find {query}.")
                channel = bot.get_channel(TEXT_CHANNEL_ID)
                if channel: await channel.send(f"❌ Couldn't find {query}.")
                return

        # 6a. Start at a RANDOM offset (never track 1 when tracks > 1)
        if len(tracks) > 1:
            start_offset = random.randint(1, len(tracks) - 1)
        else:
            start_offset = 0

        walk_state = {
            "query": query,
            "source": source,
            "playlist_name": playlist_title or query.title(),
            "tracks": tracks,
            "index": start_offset,
            "tracks_played": 0,
            "max_tracks": random.randint(10, 15),
            "single_song": None
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

        # Stop old playback and queue new track immediately
        await send_bridge_cmd({"cmd": "stop"})
        await send_bridge_cmd({"cmd": "queue", "url": track["url"], "title": track["title"], "artist": track["artist"]})
        log_pick("echo", query, source, walk_state["playlist_name"], track["title"], track["artist"], track.get("video_id"))
        
        d_sec = parse_duration_seconds(track.get("duration"))
        if d_sec > 45:
            bot.loop.create_task(prefetch_task(d_sec, pick_gen))
            
    except Exception as e:
        print(f"[ERROR] start_walk failed for query='{query}': {e}", flush=True)
        try:
            await speak(f"Couldn't find {query}.")
        except Exception:
            pass
    finally:
        walk_in_flight = False

async def advance_walk(trigger="queue_empty", gen_id=0):
    global walk_state, pick_gen
    if pick_gen != gen_id: return
    
    tracks = walk_state.get("tracks", [])
    if not tracks:
        await re_roll_walk(gen_id=gen_id)
        return

    walk_state["tracks_played"] += 1
    if walk_state["tracks_played"] >= walk_state.get("max_tracks", 12):
        await re_roll_walk(gen_id=gen_id)
        return

    start_idx = (walk_state["index"] + 1) % len(tracks)
    track = None
    for i in range(len(tracks)):
        idx = (start_idx + i) % len(tracks)
        candidate = tracks[idx]
        vid = candidate.get("video_id")
        if not vid or vid not in recent_played_vids:
            track = candidate
            walk_state["index"] = idx
            break
            
    if not track:
        track = tracks[start_idx]
        walk_state["index"] = start_idx

    if pick_gen != gen_id: return
    await send_bridge_cmd({"cmd": "queue", "url": track["url"], "title": track["title"], "artist": track["artist"]})
    log_pick(trigger, walk_state.get("query", ""), walk_state.get("source", "ytmusic"), walk_state.get("playlist_name", ""), track["title"], track["artist"], track.get("video_id"))
    
    d_sec = parse_duration_seconds(track.get("duration"))
    if d_sec > 45:
        bot.loop.create_task(prefetch_task(d_sec, pick_gen))

async def re_roll_walk(gen_id=0):
    global walk_state, pick_gen
    if pick_gen != gen_id: return
    
    query = walk_state.get("query")
    source = walk_state.get("source", "ytmusic")
    
    if query:
        await start_walk(query, source=source, gen_id=gen_id)
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
                    "single_song": None
                }
                track = tracks[offset]
                await send_bridge_cmd({"cmd": "queue", "url": track["url"], "title": track["title"], "artist": track["artist"]})
                log_pick("autoplay", "curated", "ytmusic", walk_state["playlist_name"], track["title"], track["artist"], track.get("video_id"))
                d_sec = parse_duration_seconds(track.get("duration"))
                if d_sec > 45:
                    bot.loop.create_task(prefetch_task(d_sec, pick_gen))

async def announce_track_start(title, artist, url, video_id):
    global current_persona, walk_state
    
    playlist_name = walk_state.get("playlist_name") or "Curated"
    sentiment_line = await ask_llm_sentiment(title, artist, playlist_name)
    print(f"[{current_persona.upper()}] {sentiment_line}", flush=True)
    
    track_idx = walk_state.get("tracks_played", 1)
    total_tracks = len(walk_state.get("tracks", []))
    track_meta = f"track {track_idx}/{total_tracks}" if total_tracks > 1 else "track 1/1"
    
    song_url = url if url else f"https://youtube.com/watch?v={video_id}"
    
    channel = bot.get_channel(TEXT_CHANNEL_ID)
    if channel:
        embed = discord.Embed(color=discord.Color.blurple())
        embed.description = (
            f"Now playing: [{title} — {artist}]({song_url})\n"
            f'via "{playlist_name}" · {track_meta}\n'
            f"{sentiment_line}"
        )
        await channel.send(embed=embed)

# Voice Command Parser
async def handle_voice_command(text, user_id):
    global current_persona, pick_gen
    
    wake_matches = list(re.finditer(r'(hey\s+)?(echo|ecko|eco)[,\s]*', text, flags=re.IGNORECASE))
    wake_text = text[wake_matches[-1].end():] if wake_matches else text
    clean = wake_text.strip()
    clean_norm = clean.strip(" .,!?:;'\"")
    
    channel = bot.get_channel(TEXT_CHANNEL_ID)
    
    # Control patterns
    if re.search(r'^(skip|next)(\s+song|\s+track)?$', clean_norm, re.IGNORECASE):
        await send_bridge_cmd({"cmd": "skip"})
        asyncio.create_task(speak("Skipping."))
        if channel: await channel.send("⏭️ Skipping track.")
        return

    if re.search(r'^(back|previous)(\s+song|\s+track)?$', clean_norm, re.IGNORECASE):
        res = await send_bridge_cmd({"cmd": "back"}, wait_for_reply=True)
        if res and res.get("event") == "error":
            asyncio.create_task(speak("Nothing before this."))
            if channel: await channel.send(f"❌ {res.get('message', 'Nothing before this.')}")
        else:
            asyncio.create_task(speak("Going back."))
            if channel: await channel.send("⏪ Playing previous track.")
        return

    if re.search(r'^pause(\s+the\s+music|\s+playback)?$', clean_norm, re.IGNORECASE):
        await send_bridge_cmd({"cmd": "pause"})
        asyncio.create_task(speak("Paused."))
        if channel: await channel.send("⏸️ Paused.")
        return

    if re.search(r'^(resume|unpause|continue)(\s+the\s+music|\s+playback)?$', clean_norm, re.IGNORECASE):
        await send_bridge_cmd({"cmd": "resume"})
        asyncio.create_task(speak("Resuming."))
        if channel: await channel.send("▶️ Resumed.")
        return

    if re.search(r'^stop(\s+the\s+music|\s+playback|\s+radio|\s+mix)?$', clean_norm, re.IGNORECASE):
        await send_bridge_cmd({"cmd": "stop"})
        asyncio.create_task(speak("Stopped."))
        if channel: await channel.send("⏹️ Stopped.")
        return

    if re.search(r'^(what\'?s\s+playing|what\s+is\s+(this|playing))', clean_norm, re.IGNORECASE):
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

    if re.search(r'^(play\s+(a\s+)?mix|radio|start\s+(a\s+)?radio|autoplay)$', clean_norm, re.IGNORECASE):
        pick_gen += 1
        asyncio.create_task(speak("Radio mode on."))
        if channel: await channel.send("📻 Rolling the radio mix...")
        asyncio.create_task(re_roll_walk(gen_id=pick_gen))
        return

    # Play pattern
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
        
        source = "ytmusic"
        if re.search(r'\bon\s+soundcloud\b', query, re.IGNORECASE):
            source = "soundcloud"
            query = re.sub(r'\bon\s+soundcloud\b', '', query, flags=re.IGNORECASE).strip()

        query = query.strip(" .,!?:;'\"")
        
        if query:
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
        segments, info = stt_model.transcribe(audio_np, beam_size=5, language=SPOKEN_LANGUAGE)
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
                        vid = None
                        if url:
                            if "v=" in url: vid = url.split("v=")[-1].split("&")[0]
                            elif "youtu.be/" in url: vid = u = url.split("youtu.be/")[-1].split("?")[0]
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] ▶️ Now playing: {title} — {artist}", flush=True)
                        log_play_history(title, artist, url, vid, walk_state.get("playlist_name", ""))
                        asyncio.create_task(announce_track_start(title, artist, url, vid))
                    elif evt == "queue_empty":
                        player_state = header.get("player", "idle")
                        if player_state != "playing":
                            pick_gen += 1
                            asyncio.create_task(advance_walk(trigger="queue_empty", gen_id=pick_gen))
                    continue

                user_id = int(header["u"])
                n = int(header["n"])
            except: continue

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

# Persona UI Components
class PersonaSelect(discord.ui.Select):
    def __init__(self):
        options = [
            discord.SelectOption(label="Unhinged", value="unhinged", description="Feral gremlin DJ with no filter", emoji="🤪"),
            discord.SelectOption(label="Velvet", value="velvet", description="Deadpan late-night host, quietly unimpressed", emoji="🍸"),
            discord.SelectOption(label="Hype", value="hype", description="Catastrophically over-excited hype-man", emoji="🔥"),
        ]
        super().__init__(placeholder="Choose Echo's persona...", min_values=1, max_values=1, options=options)

    async def callback(self, interaction: discord.Interaction):
        global current_persona
        current_persona = self.values[0]
        await interaction.response.edit_message(content=f"Persona locked in: **{current_persona.capitalize()}** 🔒", view=None)

class PersonaView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)
        self.add_item(PersonaSelect())

# Slash Commands
@bot.slash_command(name="join", description="Join the configured voice channel")
async def join(ctx):
    await ctx.defer()
    global bridge_process, bridge_server, bridge_active, intentional_leave, pick_gen
    intentional_leave = False
    if bridge_active:
        return await ctx.followup.send("Already joined and listening via Node bridge! 🎧", view=PersonaView())
    if not bridge_server:
        bridge_server = await asyncio.start_server(handle_bridge_client, '127.0.0.1', BRIDGE_PORT)
    bridge_connected_event.clear()
    try:
        bridge_process = subprocess.Popen(["node", "voice-bridge/bridge.js"], cwd=os.getcwd())
    except Exception as e:
        return await ctx.followup.send(f"Failed to start Node bridge: {e}")
    try:
        await asyncio.wait_for(bridge_connected_event.wait(), timeout=15.0)
        await ctx.followup.send("Joined and listening via Node bridge! 🎧 Pick a persona:", view=PersonaView())
        if JOIN_AUTOPLAY:
            pick_gen += 1
            asyncio.create_task(re_roll_walk(gen_id=pick_gen))
    except asyncio.TimeoutError:
        if bridge_process: bridge_process.terminate()
        await ctx.followup.send("Failed to join: Bridge did not connect in time.")

@bot.slash_command(name="leave", description="Leave the voice channel")
async def leave(ctx):
    global bridge_process, bridge_server, bridge_active, intentional_leave
    await ctx.defer()
    intentional_leave = True
    await send_bridge_cmd({"cmd": "stop"})
    if bridge_process: bridge_process.terminate(); bridge_process = None
    if bridge_server: bridge_server.close(); await bridge_server.wait_closed(); bridge_server = None
    bridge_active = False
    await ctx.followup.send("Left the channel.")

@bot.slash_command(name="skip", description="Skip current track")
async def skip(ctx):
    await ctx.defer()
    await send_bridge_cmd({"cmd": "skip"})
    await ctx.followup.send("Skipped current track ⏭️")

@bot.slash_command(name="back", description="Play previous track")
async def back(ctx):
    await ctx.defer()
    res = await send_bridge_cmd({"cmd": "back"}, wait_for_reply=True)
    if res and res.get("event") == "error": await ctx.followup.send(f"❌ {res.get('message')}")
    else: await ctx.followup.send("Going back to previous track ⏪")

@bot.slash_command(name="pause", description="Pause playback")
async def pause(ctx):
    await ctx.defer()
    await send_bridge_cmd({"cmd": "pause"})
    await ctx.followup.send("Paused playback ⏸️")

@bot.slash_command(name="resume", description="Resume playback")
async def resume(ctx):
    await ctx.defer()
    await send_bridge_cmd({"cmd": "resume"})
    await ctx.followup.send("Resumed playback ▶️")

@bot.slash_command(name="stop", description="Stop playback and clear queue")
async def stop(ctx):
    await ctx.defer()
    await send_bridge_cmd({"cmd": "stop"})
    await ctx.followup.send("Playback stopped ⏹️")

@bot.slash_command(name="persona", description="Switch Echo's personality")
async def persona_cmd(ctx):
    await ctx.respond("Choose Echo's persona:", view=PersonaView())

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

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")
    
    print("[DEBUG] Warming up Ollama...")
    async def warmup():
        await asyncio.sleep(3)
        res = await ask_llm('Say "ready" in one word.')
        if res: print(f"[DEBUG] Ollama warmup success: {res.strip()}")
        else: print("[WARNING] Ollama warmup returned no response.")
    bot.loop.create_task(warmup())
        
    bot.loop.create_task(transcribe_worker())

if __name__ == "__main__":
    if not os.path.exists("logs"): os.makedirs("logs")
    if DISCORD_TOKEN: bot.run(DISCORD_TOKEN)
    else: print("Error: DISCORD_TOKEN not found in .env")
