import asyncio
import os
import sys
import time
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bot

def test_resampling_sine():
    print("--- Test 1: Resampling 24k to 48k on 440Hz sine wave ---")
    sr = 24000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    freq = 440.0
    sine_24k = np.sin(2 * np.pi * freq * t).astype(np.float32)
    
    # Run resample
    resampled_48k = bot.resample_24k_to_48k(sine_24k, sr=24000)
    
    # 1. Output shape check
    assert resampled_48k.shape == (48000,), f"Expected shape (48000,), got {resampled_48k.shape}"
    print("  [PASS] Output shape is exactly (48000,)")
    
    # 2. Rising slope check on output[1]
    # At t=0, sine is 0; at t=1/24000, sine is positive
    assert sine_24k[0] < sine_24k[1], "Sanity check: t[0]->t[1] should be rising slope"
    assert sine_24k[0] < resampled_48k[1] < sine_24k[1], (
        f"output[1] ({resampled_48k[1]}) must lie strictly between input[0] ({sine_24k[0]}) and input[1] ({sine_24k[1]})"
    )
    print("  [PASS] output[1] lies strictly between input[0] and input[1] on rising slope")
    
    # 3. FFT energy comparison (contrast linear interp vs sample repetition)
    repeated_48k = np.repeat(sine_24k, 2)
    
    fft_interp = np.abs(np.fft.rfft(resampled_48k))
    fft_repeat = np.abs(np.fft.rfft(repeated_48k))
    freqs = np.fft.rfftfreq(len(resampled_48k), d=1/48000)
    
    fund_idx = np.argmin(np.abs(freqs - 440.0))
    fund_mag_interp = fft_interp[fund_idx]
    
    # Energy above 12kHz
    above_12k_mask = freqs > 12000
    max_above_12k_interp = np.max(fft_interp[above_12k_mask])
    max_above_12k_repeat = np.max(fft_repeat[above_12k_mask])
    
    db_rejection_interp = 20 * np.log10(max_above_12k_interp / fund_mag_interp)
    db_rejection_repeat = 20 * np.log10(max_above_12k_repeat / np.max(fft_repeat[fund_idx]))
    
    print(f"  FFT peak above 12kHz (np.repeat): {db_rejection_repeat:.2f} dB relative to fundamental")
    print(f"  FFT peak above 12kHz (linear interp): {db_rejection_interp:.2f} dB relative to fundamental")
    
    assert db_rejection_interp < -20.0, f"Expected >20 dB suppression above 12kHz, got {db_rejection_interp:.2f} dB"
    assert db_rejection_interp < db_rejection_repeat, "Linear interp should have significantly lower high-frequency imaging than repeat"
    print("  [PASS] FFT energy above 12kHz is >20 dB below fundamental (strong anti-imaging)")

def test_resampling_fallback():
    print("--- Test 2: Resampling unexpected sample rate fallback ---")
    data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    res = bot.resample_24k_to_48k(data, sr=22050)
    assert np.array_equal(res, np.repeat(data, 2)), "Expected fallback to np.repeat when sr != 24000"
    print("  [PASS] Successfully fell back to np.repeat on sr != 24000")

def test_tts_serialization():
    print("--- Test 3: TTS Queue & Worker Serialization ---")
    async def async_test():
        # Clear queue
        while not bot.tts_queue.empty():
            bot.tts_queue.get_nowait()
            
        mock_kokoro = MagicMock()
        # Mock kokoro.create to return 0.1s of audio at 24kHz (2400 samples)
        fake_audio = np.zeros(2400, dtype=np.float32)
        mock_kokoro.create.return_value = (fake_audio, 24000)
        
        events = []
        async def fake_send_bridge_cmd(cmd_dict, wait_for_reply=False):
            if cmd_dict.get("cmd") == "speak":
                events.append(("send_bridge_cmd", time.time()))
            return None
            
        with patch.object(bot, "kokoro", mock_kokoro), \
             patch.object(bot, "TTS_ENABLED", True), \
             patch.object(bot, "send_bridge_cmd", side_effect=fake_send_bridge_cmd):
            
            # Start worker task
            worker_task = asyncio.create_task(bot._tts_worker())
            
            # Call speak twice concurrently
            t0 = time.time()
            await bot.speak("First message")
            await bot.speak("Second message")
            
            # Wait for both to be processed
            await bot.tts_queue.join()
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass
                
            assert len(events) == 2, f"Expected 2 bridge speak calls, got {len(events)}"
            t_gap = events[1][1] - events[0][1]
            # 2400 samples resampled to 4800 samples -> 4800 / 48000 = 0.1s duration
            # The second call must have waited at least 0.1s
            print(f"  Time between consecutive speak dispatches: {t_gap:.3f}s (expected >= 0.10s)")
            assert t_gap >= 0.09, f"Expected serialization delay >= ~0.10s, got {t_gap:.3f}s"
            print("  [PASS] Sequential dispatch serialized with audio playback duration")
            
    asyncio.run(async_test())

if __name__ == "__main__":
    test_resampling_sine()
    test_resampling_fallback()
    test_tts_serialization()
    print("\nALL BRIEF T UNIT TESTS PASSED!")
