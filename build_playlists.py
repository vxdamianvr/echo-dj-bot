import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from ytmusicapi import YTMusic

LIBRARY_GENRES = [
    "pop", "rock", "hip-hop", "phonk", "electronic", "lo-fi", "classical", "jazz",
    "metal", "indie", "rnb", "k-pop", "latin", "ambient", "country", "synthwave",
    "house", "techno", "dubstep", "drum and bass", "hyperpop", "trap", "shoegaze",
    "pop punk", "funk", "afrobeats", "reggaeton", "city pop", "folk", "bossa nova",
    "jersey club", "amapiano", "digicore", "sigilkore", "darksynth", "baile funk",
    "drift phonk", "breakcore", "speed house", "hardstyle", "rawstyle", "dub techno",
    "drone", "post-punk", "emo rap", "cloud rap", "rage", "plugg"
]

def fetch_genre_playlists(genre):
    yt = YTMusic()
    try:
        res = yt.search(f"{genre} playlist", filter="playlists", limit=10)
        pids = []
        for r in res:
            pid = r.get("browseId") or r.get("playlistId")
            if pid:
                if pid.startswith("VL"):
                    pid = pid[2:]
                if pid not in pids:
                    pids.append(pid)
            if len(pids) >= 5:
                break
        return genre, pids
    except Exception as e:
        print(f"Error fetching playlists for {genre}: {e}")
        return genre, []

def main():
    print(f"Building playlists.json for {len(LIBRARY_GENRES)} genres...")
    t0 = time.time()
    playlists = {}
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_genre_playlists, g): g for g in LIBRARY_GENRES}
        for fut in as_completed(futures):
            genre, pids = fut.result()
            playlists[genre] = pids
            print(f"[{len(playlists)}/{len(LIBRARY_GENRES)}] {genre}: {len(pids)} playlists")

    with open("playlists.json", "w", encoding="utf-8") as f:
        json.dump(playlists, f, indent=2)

    print(f"Successfully generated playlists.json in {time.time() - t0:.1f}s.")

if __name__ == "__main__":
    main()
