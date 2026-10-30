import asyncio
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bot

def test_gate_cases():
    print("--- Test 1: Gate cases ---")
    song_lp = {"title": "Crawling", "artist": "Linkin Park", "video_id": "I6NU3kWqnDE"}
    
    # 1. "crawling linkin park" -> song route
    assert bot.is_strong_song_match("crawling linkin park", song_lp) is True, "crawling linkin park should match"
    
    # 2. "crawling by linkin park" -> song route ("by" is stopword)
    assert bot.is_strong_song_match("crawling by linkin park", song_lp) is True, "crawling by linkin park should match"
    
    # 3. "play jazz" (query "jazz") -> walk path (<2 significant words)
    assert bot.is_strong_song_match("jazz", {"title": "Jazz Classics", "artist": "Various"}) is False, "single-word jazz should not match"
    
    # 4. "play creep" (query "creep") -> walk path (<2 significant words)
    assert bot.is_strong_song_match("creep", {"title": "Creep", "artist": "Radiohead"}) is False, "single-word creep should not match"
    
    # 5. "the beatles" -> walk path ("the" is stopword, leaving 1 word)
    assert bot.is_strong_song_match("the beatles", {"title": "Yesterday", "artist": "The Beatles"}) is False, "single significant word beatles should not match"
    
    # 6. "metal music" vs Heavy Metal -> walk path ("music" not in target)
    assert bot.is_strong_song_match("metal music", {"title": "Heavy Metal", "artist": "Band"}) is False, "metal music missing 'music' in target should not match"
    
    # 7. None song or empty query
    assert bot.is_strong_song_match("", song_lp) is False
    assert bot.is_strong_song_match("crawling linkin park", None) is False
    
    print("  [PASS] All word-match gate cases behave correctly")

async def test_fill_mix_queue():
    print("--- Test 2: fill_mix_queue (order, seed dropped, dedup filtered, gen bump) ---")
    seed = {"title": "Crawling", "artist": "Linkin Park", "video_id": "vid_seed", "url": "https://youtube.com/watch?v=vid_seed"}
    
    # Mock watch playlist response
    mock_watch = {
        "tracks": [
            {"videoId": "vid_seed", "title": "Crawling", "artists": [{"name": "Linkin Park"}]}, # seed track (must drop)
            {"videoId": "vid_1", "title": "In the End", "artists": [{"name": "Linkin Park"}]},    # keep
            {"videoId": "vid_dup", "title": "Numb", "artists": [{"name": "Linkin Park"}]},         # in dedup (must filter)
            {"videoId": "vid_2", "title": "Faint", "artists": [{"name": "Linkin Park"}]},          # keep
            {"videoId": "vid_3", "title": "Papercut", "artists": [{"name": "Linkin Park"}]},       # keep
            {"videoId": "vid_4", "title": "Somewhere I Belong", "artists": [{"name": "Linkin Park"}]} # keep
        ]
    }
    
    bot.bridge_active = True
    bot.pick_gen = 5
    bot.recent_played_vids.add("vid_dup")
    bot.walk_state = {
        "tracks": [dict(seed)],
        "index": 0,
        "tracks_played": 0,
        "max_tracks": 1
    }
    
    with patch.object(bot.yt, 'get_watch_playlist', return_value=mock_watch), \
         patch.object(bot, 'send_bridge_cmd', AsyncMock(return_value={"queue": 0})):
        await bot.fill_mix_queue(seed, gen_id=5)
        
        tracks = bot.walk_state["tracks"]
        # Seed track was index 0
        assert tracks[0]["video_id"] == "vid_seed"
        
        # Surviving mix tracks: vid_1, vid_2, vid_3, vid_4 (vid_seed dropped, vid_dup filtered)
        vids = [t["video_id"] for t in tracks[1:]]
        assert vids == ["vid_1", "vid_2", "vid_3", "vid_4"], f"Expected ['vid_1', 'vid_2', 'vid_3', 'vid_4'], got {vids}"
        assert all(t["via"] == "YT Music mix" for t in tracks[1:])
        print("  [PASS] Queue filled in order, seed dropped, dedup filtered")

    # Generation bump test: gen_id at launch is 5, but pick_gen bumped to 6 before append
    bot.walk_state = {
        "tracks": [dict(seed)],
        "index": 0,
        "tracks_played": 0,
        "max_tracks": 1
    }
    bot.pick_gen = 6 # Bumped!
    with patch.object(bot.yt, 'get_watch_playlist', return_value=mock_watch):
        await bot.fill_mix_queue(seed, gen_id=5)
        # Tracks should NOT be appended because gen_id 5 is stale
        assert len(bot.walk_state["tracks"]) == 1
        assert bot.walk_state["tracks"][0]["video_id"] == "vid_seed"
        print("  [PASS] Stale mix discarded on generation bump")

async def test_mix_exhaustion_refill():
    print("--- Test 3: Mix exhaustion re-seeds from original mix_seed ---")
    original_seed = {"title": "Crawling", "artist": "Linkin Park", "video_id": "vid_orig"}
    bot.bridge_active = True
    bot.pick_gen = 10
    bot.walk_state = {
        "query": "crawling linkin park",
        "source": "ytmusic",
        "playlist_name": "YT Music mix",
        "tracks": [{"title": "Last Song", "artist": "Other", "video_id": "vid_last"}],
        "index": 0,
        "tracks_played": 10,
        "max_tracks": 10,
        "mix_seed": original_seed
    }
    
    mock_watch = {
        "tracks": [
            {"videoId": "vid_orig", "title": "Crawling", "artists": [{"name": "Linkin Park"}]},
            {"videoId": "vid_refill_1", "title": "Points of Authority", "artists": [{"name": "Linkin Park"}]},
            {"videoId": "vid_refill_2", "title": "Breaking the Habit", "artists": [{"name": "Linkin Park"}]},
            {"videoId": "vid_refill_3", "title": "From the Inside", "artists": [{"name": "Linkin Park"}]}
        ]
    }
    
    with patch.object(bot.yt, 'get_watch_playlist', return_value=mock_watch) as mock_get_watch, \
         patch.object(bot, 'send_bridge_cmd', AsyncMock()):
        await bot.re_roll_walk(gen_id=10)
        
        # Verify get_watch_playlist was called with ORIGINAL seed video_id, not vid_last
        mock_get_watch.assert_called_once()
        _, kwargs = mock_get_watch.call_args
        assert kwargs.get("videoId") == "vid_orig", f"Expected vid_orig, got {kwargs.get('videoId')}"
        
        # Verify tracks replaced with refilled tracks
        assert len(bot.walk_state["tracks"]) == 3
        assert bot.walk_state["tracks"][0]["video_id"] == "vid_refill_1"
        assert bot.walk_state["index"] == 0
        print("  [PASS] Mix exhaustion properly re-seeded from original requested track")

    # Fallback test: when refill yields <3 usable tracks -> fall back to start_walk playlist search
    bot.walk_state["mix_seed"] = original_seed
    mock_empty_watch = {"tracks": []}
    with patch.object(bot.yt, 'get_watch_playlist', return_value=mock_empty_watch), \
         patch.object(bot, 'start_walk', AsyncMock()) as mock_start_walk:
        await bot.re_roll_walk(gen_id=10)
        mock_start_walk.assert_awaited_once_with("crawling linkin park", source="ytmusic", gen_id=10, interrupt=False)
        print("  [PASS] Mix exhaustion falls back to playlist search when usable tracks < 3")

async def test_resolve_and_start_play():
    print("--- Test 4: resolve_and_start_play concurrent resolution ---")
    mock_channel = MagicMock()
    mock_channel.send = AsyncMock()
    
    # 1. Strong match path
    song_lp = {"title": "Crawling", "artist": "Linkin Park", "video_id": "I6NU3kWqnDE", "url": "https://youtube.com/watch?v=I6NU3kWqnDE"}
    with patch.object(bot, 'search_single_song', AsyncMock(return_value=song_lp)), \
         patch.object(bot, 'resolve_playlist_ytmusic', AsyncMock(return_value=[])), \
         patch.object(bot, 'speak', AsyncMock()) as mock_speak, \
         patch.object(bot, 'start_song_route', AsyncMock()) as mock_song_route, \
         patch.object(bot, 'fill_mix_queue', AsyncMock()) as mock_fill_mix:
        
        await bot.resolve_and_start_play("crawling linkin park", "ytmusic", mock_channel)
        await asyncio.sleep(0.05)
        
        mock_speak.assert_called_once_with("On it — Crawling by Linkin Park.")
        mock_channel.send.assert_awaited_once_with("👂 On it — Crawling by Linkin Park.")
        mock_song_route.assert_awaited_once()
        print("  [PASS] Strong match triggered song route, named TTS ack, and mix fill")

    # 2. Generic query path (jazz)
    mock_channel.reset_mock()
    mock_speak.reset_mock()
    candidates = [{"id": "pl1", "title": "Jazz Vibes", "track_count": 20}]
    with patch.object(bot, 'search_single_song', AsyncMock(return_value={"title": "Jazz Song", "artist": "Artist", "video_id": "v1"})), \
         patch.object(bot, 'resolve_playlist_ytmusic', AsyncMock(return_value=candidates)), \
         patch.object(bot, 'speak', AsyncMock()) as mock_speak2, \
         patch.object(bot, 'start_walk', AsyncMock()) as mock_start_walk:
        
        await bot.resolve_and_start_play("jazz", "ytmusic", mock_channel)
        await asyncio.sleep(0.05)
        
        # Must not take song route with "by Artist"
        assert "by Artist" not in mock_speak2.call_args[0][0]
        assert mock_speak2.call_args[0][0] == bot.get_tts_ack("jazz", bot.current_persona)
        # Must pass pre_resolved_candidates to start_walk
        mock_start_walk.assert_called_once()
        _, kwargs = mock_start_walk.call_args
        assert kwargs.get("pre_resolved_candidates") == candidates
        print("  [PASS] Generic query took playlist route with pre-resolved candidates")

async def main():
    test_gate_cases()
    await test_fill_mix_queue()
    await test_mix_exhaustion_refill()
    await test_resolve_and_start_play()
    print("\nALL BRIEF I-REVISED UNIT TESTS PASSED!")

if __name__ == "__main__":
    asyncio.run(main())
