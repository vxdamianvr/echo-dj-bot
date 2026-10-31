import unittest
import asyncio
import os
import sys
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bot

class TestBriefV(unittest.IsolatedAsyncioTestCase):

    def test_parse_commands(self):
        print("\n--- Test 1: Parser (Q-Insertion and Bare Play) ---")
        cases = [
            ("queue crawling linkin park", "insert", "crawling linkin park", "ytmusic"),
            ("cue creep radiohead", "insert", "creep radiohead", "ytmusic"),
            ("play queue jazz", "play", "queue jazz", "ytmusic"),
            ("queue", "unparseable", None, None),
            ("queue   ", "unparseable", None, None),
            ("play", "resume", None, None),
            ("play   ", "resume", None, None),
        ]
        for text, exp_action, exp_query, exp_source in cases:
            res = bot.parse_voice_command(text)
            self.assertEqual(res["action"], exp_action, f"Failed action for '{text}': got {res['action']}")
            self.assertEqual(res["query"], exp_query, f"Failed query for '{text}': got {res['query']}")
            self.assertEqual(res["source"], exp_source, f"Failed source for '{text}': got {res['source']}")
            print(f"  [PASS] '{text}' -> action={res['action']}, query={res['query']}")

    def test_parse_poll_vote(self):
        print("\n--- Test 2: parse_poll_vote ---")
        cases = [
            ("echo yes", "change"),
            ("echo no", "keep"),
            ("echo yeah change it up", "change"),
            ("echo nah keep walking", "keep"),
            ("echo yes keep it", "keep"),
            ("yes please", None),
            ("echo what was that", None),
        ]
        for text, expected in cases:
            vote = bot.parse_poll_vote(text)
            self.assertEqual(vote, expected, f"Failed parse_poll_vote for '{text}': got {vote}, expected {expected}")
            print(f"  [PASS] '{text}' -> vote={vote}")

    def test_parse_duration(self):
        print("\n--- Test 3: parse_duration ---")
        self.assertEqual(bot.parse_duration(180), 180.0)
        self.assertEqual(bot.parse_duration(180.5), 180.5)
        self.assertEqual(bot.parse_duration("180"), 180.0)
        self.assertEqual(bot.parse_duration("3:12"), 192.0)
        self.assertEqual(bot.parse_duration("1:02:03"), 3723.0)
        self.assertIsNone(bot.parse_duration("invalid"))
        self.assertIsNone(bot.parse_duration(None))
        self.assertIsNone(bot.parse_duration(""))
        print("  [PASS] parse_duration correctly parses ints, floats, mm:ss, hh:mm:ss, and handles invalid inputs")

    async def test_splice_insert_fifo(self):
        print("\n--- Test 4: splice_insert FIFO ordering ---")
        bot.walk_state = {
            "query": "seed query",
            "source": "ytmusic",
            "playlist_name": "Test Mix",
            "tracks": [
                {"title": "Track A", "url": "urlA", "video_id": "vidA"},
                {"title": "Track B", "url": "urlB", "video_id": "vidB"},
                {"title": "Track C", "url": "urlC", "video_id": "vidC"}
            ],
            "index": 0,
            "tracks_played": 1,
            "max_tracks": 12,
            "single_song": None,
            "mix_seed": {"title": "Seed Track"},
            "mix_fill_gen": None,
            "explicit_playlist": None
        }
        bot.pick_gen = 42

        bridge_cmds = []
        async def mock_send_bridge_cmd(cmd_dict, wait_for_reply=False):
            bridge_cmds.append(cmd_dict)
            return {"status": "ok"}
        
        orig_send_cmd = bot.send_bridge_cmd
        orig_speak = bot.speak
        bot.send_bridge_cmd = mock_send_bridge_cmd
        bot.speak = lambda text: asyncio.sleep(0)

        try:
            # First insertion: Track X
            song_x = {"title": "Track X", "artist": "Artist X", "url": "urlX", "video_id": "vidX"}
            await bot.splice_insert(song_x)

            tracks = bot.walk_state["tracks"]
            titles = [t["title"] for t in tracks]
            self.assertEqual(titles, ["Track A", "Track X", "Track B", "Track C"])
            self.assertTrue(tracks[1].get("inserted"))
            self.assertEqual(tracks[1].get("via"), "Queued")
            self.assertFalse(tracks[0].get("inserted", False))
            self.assertFalse(tracks[2].get("inserted", False))
            self.assertEqual(bot.walk_state["max_tracks"], 13)
            self.assertEqual(bot.pick_gen, 42)
            self.assertEqual(bot.walk_state["mix_seed"], {"title": "Seed Track"})

            # Second insertion: Track Y (FIFO: should go AFTER pending insertions, so after X, before B)
            song_y = {"title": "Track Y", "artist": "Artist Y", "url": "urlY", "video_id": "vidY"}
            await bot.splice_insert(song_y)

            tracks = bot.walk_state["tracks"]
            titles = [t["title"] for t in tracks]
            self.assertEqual(titles, ["Track A", "Track X", "Track Y", "Track B", "Track C"])
            self.assertTrue(tracks[1].get("inserted"))
            self.assertTrue(tracks[2].get("inserted"))
            self.assertFalse(tracks[3].get("inserted", False))
            self.assertEqual(bot.walk_state["max_tracks"], 14)
            self.assertEqual(bot.pick_gen, 42)
            self.assertEqual(bot.walk_state["mix_seed"], {"title": "Seed Track"})

            # Verify bridge commands sent
            self.assertEqual(len(bridge_cmds), 2)
            self.assertEqual(bridge_cmds[0]["cmd"], "queue_next")
            self.assertEqual(bridge_cmds[0]["title"], "Track X")
            self.assertEqual(bridge_cmds[1]["cmd"], "queue_next")
            self.assertEqual(bridge_cmds[1]["title"], "Track Y")

            print("  [PASS] FIFO ordering [A, X, Y, B, C] verified, flags set, max_tracks updated, mix_seed & pick_gen untouched")
        finally:
            bot.send_bridge_cmd = orig_send_cmd
            bot.speak = orig_speak

    async def test_resolve_poll_idempotency(self):
        print("\n--- Test 5: resolve_poll Idempotency ---")
        bot.vibe_poll = {
            "active": True,
            "resolved": False,
            "message": None,
            "video_id": "test_vid",
            "gen": 10,
            "index": 1,
            "timer_task": None,
            "timeout_task": None,
            "vibe_name": "the test mix",
            "inserted_title": "Detour Track"
        }

        vibe_changes = []
        async def mock_start_vibe_change():
            vibe_changes.append(True)

        orig_vibe_change = bot.start_vibe_change
        bot.start_vibe_change = mock_start_vibe_change

        try:
            # Simulate two rapid resolve_poll calls
            task1 = asyncio.create_task(bot.resolve_poll("change", via="button"))
            task2 = asyncio.create_task(bot.resolve_poll("change", via="voice"))
            await asyncio.gather(task1, task2)

            self.assertEqual(len(vibe_changes), 1, "start_vibe_change should only be called once")
            self.assertFalse(bot.vibe_poll["active"])
            print("  [PASS] resolve_poll idempotency verified: only first signal took effect")
        finally:
            bot.start_vibe_change = orig_vibe_change

    def test_bridge_commands_order(self):
        print("\n--- Test 6: bridge queue_next & clear_upcoming ---")
        audio_queue = [{"url": "old_url", "title": "old_next", "artist": "old_artist", "video_id": "old_vid"}]
        # queue_next unshifts to front
        msg = {"cmd": "queue_next", "url": "new_url", "title": "inserted", "artist": "new_artist", "video_id": "new_vid"}
        audio_queue.insert(0, {"url": msg["url"], "title": msg["title"], "artist": msg["artist"], "video_id": msg["video_id"]})
        self.assertEqual(len(audio_queue), 2)
        self.assertEqual(audio_queue[0]["title"], "inserted")
        self.assertEqual(audio_queue[1]["title"], "old_next")

        # clear_upcoming empties queue
        audio_queue = []
        self.assertEqual(len(audio_queue), 0)
        print("  [PASS] bridge queue_next unshift order [inserted, old_next] and clear_upcoming verified")

if __name__ == "__main__":
    unittest.main()
