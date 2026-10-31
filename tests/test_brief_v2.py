import unittest
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bot

class TestBriefV2(unittest.IsolatedAsyncioTestCase):

    def test_vote_remainder(self):
        print("\n--- Test 1: _vote_remainder ---")
        self.assertIsNone(bot._vote_remainder("echo yes"))
        self.assertEqual(bot._vote_remainder("echo yes and skip"), "skip")
        
        rem_jazz = bot._vote_remainder("echo no, play some jazz")
        self.assertIsNotNone(rem_jazz)
        parsed_jazz = bot.parse_voice_command(rem_jazz)
        self.assertEqual(parsed_jazz["action"], "play")
        self.assertEqual(parsed_jazz["query"], "jazz")
        
        self.assertIsNone(bot._vote_remainder("echo yep"))
        self.assertIsNone(bot._vote_remainder("echo nah"))
        self.assertIsNone(bot._vote_remainder("echo yes please")) # "please" is unparseable
        print("  [PASS] _vote_remainder correctly extracts remainder commands and ignores bare votes or unparseable text")

    async def test_attribution_part1(self):
        print("\n--- Test 2: Attribution Fix (Part 1) ---")
        song_res_audio = {"title": "Crawling", "artist": "Linkin Park", "video_id": "vid1", "url": "https://youtube.com/watch?v=vid1", "resultType": "song"}
        song_res_video = {"title": "I get high better alone", "artist": "NaFr", "video_id": "vid2", "url": "https://youtube.com/watch?v=vid2", "resultType": "video"}

        bot.pick_gen = 1
        bot.walk_in_flight = False
        await bot.start_song_route(song_res_audio, query="crawling", source="ytmusic", gen_id=1)
        self.assertEqual(bot.walk_state["playlist_name"], "YouTube Music")
        self.assertEqual(bot.walk_state["tracks"][0]["via"], "YouTube Music")

        bot.pick_gen = 2
        bot.walk_in_flight = False
        await bot.start_song_route(song_res_video, query="nafr", source="ytmusic", gen_id=2)
        self.assertEqual(bot.walk_state["playlist_name"], "YouTube")
        self.assertEqual(bot.walk_state["tracks"][0]["via"], "YouTube")
        print("  [PASS] Song-type seed uses via 'YouTube Music', video-type seed uses via 'YouTube'")

    async def test_n1_now_playing_ref_gen_update(self):
        print("\n--- Test 3: N1 now_playing_ref gen update on vibe change ---")
        bot.walk_state = {
            "query": "test",
            "tracks": [{"title": "Current Track", "video_id": "cur_vid", "url": "url"}],
            "index": 0
        }
        bot.pick_gen = 10
        bot.now_playing_ref = {"video_id": "cur_vid", "gen": 10, "message": None}

        orig_send_cmd = bot.send_bridge_cmd
        orig_speak = bot.speak
        orig_ensure_mix = bot.ensure_mix_fill
        bot.send_bridge_cmd = lambda cmd, wait_for_reply=False: asyncio.sleep(0)
        bot.speak = lambda text: asyncio.sleep(0)
        bot.ensure_mix_fill = lambda gen: None

        try:
            await bot.start_vibe_change()
            self.assertEqual(bot.pick_gen, 11)
            self.assertEqual(bot.now_playing_ref["gen"], 11)
            print("  [PASS] now_playing_ref['gen'] successfully updated to new_gen (11)")
        finally:
            bot.send_bridge_cmd = orig_send_cmd
            bot.speak = orig_speak
            bot.ensure_mix_fill = orig_ensure_mix

    async def test_n2_debounce_reset_on_miss(self):
        print("\n--- Test 4: N2 debounce reset on lookup miss ---")
        bot.last_insert = {"query": None, "ts": 0.0}

        orig_search = bot.search_single_song
        orig_speak = bot.speak
        bot.search_single_song = lambda q, source="ytmusic": asyncio.sleep(0, result=None)
        bot.speak = lambda text: asyncio.sleep(0)

        try:
            await bot.handle_voice_command("queue nonexistentsong12345", 0)
            self.assertIsNone(bot.last_insert["query"])
            self.assertEqual(bot.last_insert["ts"], 0.0)
            print("  [PASS] last_insert reset to None/0.0 on failed song search")
        finally:
            bot.search_single_song = orig_search
            bot.speak = orig_speak

if __name__ == "__main__":
    unittest.main()
