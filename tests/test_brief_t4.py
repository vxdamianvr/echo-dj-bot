import unittest
import asyncio
import os
import sys
from unittest.mock import MagicMock
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bot

class TestBriefT4(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        bot.tts_cue_done = asyncio.Event()
        bot.tts_cue_done.set()
        bot.tts_queue = asyncio.Queue()

    async def test_a_tts_worker_pacing(self):
        print("\n--- Test A: TTS Worker Pacing via tts_cue_done ---")
        orig_enabled = bot.TTS_ENABLED
        orig_kokoro = bot.kokoro
        orig_send = bot.send_bridge_cmd

        bot.TTS_ENABLED = True
        mock_kokoro = MagicMock()
        # Return 0.2s of audio at 24kHz (4800 samples)
        mock_kokoro.create.return_value = (np.zeros(4800, dtype=np.float32), 24000)
        bot.kokoro = mock_kokoro

        sent_cmds = []
        async def mock_send(cmd_dict, wait_for_reply=False):
            sent_cmds.append(cmd_dict)
            return {"status": "ok"}
        bot.send_bridge_cmd = mock_send

        # Ensure tts_queue is empty
        while not bot.tts_queue.empty():
            bot.tts_queue.get_nowait()
            bot.tts_queue.task_done()

        bot.tts_cue_done.set()
        worker_task = asyncio.create_task(bot._tts_worker())

        try:
            await bot.tts_queue.put("cue1")
            await bot.tts_queue.put("cue2")

            # Allow worker to pick up cue1 and send it
            await asyncio.sleep(0.08)
            self.assertEqual(len(sent_cmds), 1)
            self.assertEqual(sent_cmds[0]["cue"], "cue1")
            self.assertFalse(bot.tts_cue_done.is_set(), "tts_cue_done should be cleared while cue1 is in flight")

            # Wait a bit longer — cue2 MUST NOT be sent while cue1 hasn't received tts_done
            await asyncio.sleep(0.15)
            self.assertEqual(len(sent_cmds), 1, "cue2 should not be dispatched before tts_done")

            # Simulate bridge emitting tts_done for cue1
            bot.tts_cue_done.set()

            # Worker should now advance and send cue2 promptly
            await asyncio.sleep(0.08)
            self.assertEqual(len(sent_cmds), 2)
            self.assertEqual(sent_cmds[1]["cue"], "cue2")
            print("  [PASS] cue2 was blocked until tts_cue_done was signaled, then dispatched promptly")
        finally:
            bot.tts_cue_done.set()
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass
            bot.TTS_ENABLED = orig_enabled
            bot.kokoro = orig_kokoro
            bot.send_bridge_cmd = orig_send
            while not bot.tts_queue.empty():
                bot.tts_queue.get_nowait()
                bot.tts_queue.task_done()

    async def test_b_timeout_fallback(self):
        print("\n--- Test B: Timeout Fallback when no tts_done Arrives ---")
        orig_enabled = bot.TTS_ENABLED
        orig_kokoro = bot.kokoro
        orig_send = bot.send_bridge_cmd
        orig_pad = bot.TTS_DONE_TIMEOUT_PAD

        bot.TTS_ENABLED = True
        bot.TTS_DONE_TIMEOUT_PAD = 0.1  # 100ms timeout pad
        mock_kokoro = MagicMock()
        # 240 samples at 24k -> 480 samples at 48k -> 0.01s audio
        mock_kokoro.create.return_value = (np.zeros(240, dtype=np.float32), 24000)
        bot.kokoro = mock_kokoro

        sent_cmds = []
        async def mock_send(cmd_dict, wait_for_reply=False):
            sent_cmds.append(cmd_dict)
            return {"status": "ok"}
        bot.send_bridge_cmd = mock_send

        worker_task = asyncio.create_task(bot._tts_worker())

        try:
            await bot.tts_queue.put("timeout_cue1")
            await bot.tts_queue.put("timeout_cue2")

            # Total timeout is duration (0.01s) + PAD (0.1s) = 0.11s
            # At 0.04s, cue1 has sent, but wait_for has not timed out
            await asyncio.sleep(0.04)
            self.assertEqual(len(sent_cmds), 1)
            self.assertEqual(sent_cmds[0]["cue"], "timeout_cue1")

            # At +0.18s (total 0.22s), timeout has fired and worker has processed cue2
            await asyncio.sleep(0.18)
            self.assertEqual(len(sent_cmds), 2)
            self.assertEqual(sent_cmds[1]["cue"], "timeout_cue2")
            print("  [PASS] Worker timed out safely after duration + PAD and unblocked subsequent cues")
        finally:
            bot.tts_cue_done.set()
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass
            bot.TTS_ENABLED = orig_enabled
            bot.kokoro = orig_kokoro
            bot.send_bridge_cmd = orig_send
            bot.TTS_DONE_TIMEOUT_PAD = orig_pad

    async def test_c_bridge_client_event_dispatch(self):
        print("\n--- Test C: handle_bridge_client 'tts_done' Dispatch ---")
        reader = asyncio.StreamReader()
        writer = MagicMock()
        writer.close = MagicMock()

        orig_leave = bot.intentional_leave
        bot.intentional_leave = True
        try:
            bot.tts_cue_done.clear()
            self.assertFalse(bot.tts_cue_done.is_set())

            # Feed the tts_done event and then close stream (EOF)
            reader.feed_data(b'{"event": "tts_done", "seq": 99}\n')
            reader.feed_eof()

            await bot.handle_bridge_client(reader, writer)

            self.assertTrue(bot.tts_cue_done.is_set(), "handle_bridge_client should set tts_cue_done on 'tts_done' event")
            print("  [PASS] 'tts_done' event received by handle_bridge_client successfully set tts_cue_done")
        finally:
            bot.intentional_leave = orig_leave

if __name__ == "__main__":
    unittest.main()

