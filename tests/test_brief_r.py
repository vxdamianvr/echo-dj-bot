import unittest
import asyncio
import os
import sys
from unittest.mock import AsyncMock, patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bot

class TestBriefR(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        bot.recent_played_vids.clear()
        bot.pick_gen = 100
        bot.bridge_active = True

    async def test_searched_playlist_exhaustion_rerolls(self):
        print("\n--- Test 1: Searched Playlist Exhaustion Re-rolls ---")
        t1 = {"title": "T1", "artist": "A1", "video_id": "vid1", "url": "url1"}
        t2 = {"title": "T2", "artist": "A2", "video_id": "vid2", "url": "url2"}
        t3 = {"title": "T3", "artist": "A3", "video_id": "vid3", "url": "url3"}

        # All 3 tracks already played (exhaustion)
        bot.recent_played_vids.update(["vid1", "vid2", "vid3"])

        bot.walk_state = {
            "query": "chill indie",
            "source": "ytmusic",
            "playlist_name": "Old Short Playlist",
            "tracks": [t1, t2, t3],
            "index": 2, # playing track 3
            "tracks_played": 3,
            "max_tracks": 15, # hasn't hit max_tracks yet
            "mix_seed": None,
            "explicit_playlist": None
        }

        with patch.object(bot, 're_roll_walk', AsyncMock()) as mock_reroll, \
             patch.object(bot, 'send_bridge_cmd', AsyncMock()) as mock_bridge:

            await bot.advance_walk("keep_ahead", gen_id=100)

            mock_reroll.assert_called_once_with(gen_id=100)
            mock_bridge.assert_not_called()
            print("  [PASS] Searched playlist on exhaustion calls re_roll_walk and does not loop old tracks")

    async def test_explicit_pasted_playlist_exhaustion_loops(self):
        print("\n--- Test 2: Explicit (Pasted) Playlist Exhaustion Loops ---")
        t1 = {"title": "T1", "artist": "A1", "video_id": "vid1", "url": "url1"}
        t2 = {"title": "T2", "artist": "A2", "video_id": "vid2", "url": "url2"}
        t3 = {"title": "T3", "artist": "A3", "video_id": "vid3", "url": "url3"}

        bot.recent_played_vids.update(["vid1", "vid2", "vid3"])

        bot.walk_state = {
            "query": "",
            "source": "ytmusic",
            "playlist_name": "My Explicit Album",
            "tracks": [t1, t2, t3],
            "index": 2, # playing track 3, next wrap is index 0 (t1)
            "tracks_played": 3,
            "max_tracks": 15,
            "mix_seed": None,
            "explicit_playlist": {"source": "ytmusic", "title": "My Explicit Album", "ref": "http://..."}
        }

        with patch.object(bot, 're_roll_walk', AsyncMock()) as mock_reroll, \
             patch.object(bot, 'send_bridge_cmd', AsyncMock()) as mock_bridge:

            await bot.advance_walk("keep_ahead", gen_id=100)

            mock_reroll.assert_not_called()
            mock_bridge.assert_called_once()
            queued_cmd = mock_bridge.call_args[0][0]
            self.assertEqual(queued_cmd["cmd"], "queue")
            self.assertEqual(queued_cmd["video_id"], "vid1")
            print("  [PASS] Explicit playlist on exhaustion loops to start_idx and does not re-roll")

    async def test_mix_pending_wait_preserved(self):
        print("\n--- Test 3: Mix Pending Branch Preserved ---")
        seed = {"title": "Seed", "artist": "Artist", "video_id": "seed_vid", "url": "url_seed"}
        bot.recent_played_vids.add("seed_vid")

        bot.walk_state = {
            "query": "crawling",
            "source": "ytmusic",
            "playlist_name": "YouTube Music",
            "tracks": [seed],
            "index": 0,
            "tracks_played": 1,
            "max_tracks": 50,
            "mix_seed": seed,
            "explicit_playlist": None
        }

        with patch.object(bot, 're_roll_walk', AsyncMock()) as mock_reroll, \
             patch.object(bot, 'ensure_mix_fill') as mock_ensure, \
             patch.object(bot, 'send_bridge_cmd', AsyncMock()) as mock_bridge:

            await bot.advance_walk("keep_ahead", gen_id=100)

            mock_reroll.assert_not_called()
            mock_bridge.assert_not_called()
            mock_ensure.assert_called_once_with(100)
            print("  [PASS] Mix pending branch waits for mix fill without re-rolling or queuing")

if __name__ == "__main__":
    unittest.main()
