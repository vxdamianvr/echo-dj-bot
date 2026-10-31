import asyncio
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bot

def test_score_and_strong_match():
    print("--- Test 1: score_song_match & is_strong_song_match ---")
    q = "i get high better alone nafr awannabe"
    cand_nafr = {"title": "I get high better alone (feat. aWannabe)", "artist": "NaFr"}
    cand_mhm = {"title": "Better Alone", "artist": "Mile High Mafia, Eric James, FNO CASTILLO"}

    score_nafr = bot.score_song_match(q, cand_nafr)
    assert score_nafr == 1.0, f"Expected 1.0 for exact match, got {score_nafr}"
    assert bot.is_strong_song_match(q, cand_nafr) is True

    score_mhm = bot.score_song_match(q, cand_mhm)
    assert score_mhm < 1.0, f"Expected <1.0 for partial match, got {score_mhm}"
    assert bot.is_strong_song_match(q, cand_mhm) is False

    assert bot.score_song_match("jazz", {"title": "Jazz Song", "artist": "Jazz Artist"}) == 0.0
    assert bot.is_strong_song_match("jazz", {"title": "Jazz Song", "artist": "Jazz Artist"}) is False
    print("  [PASS] score_song_match and is_strong_song_match behave correctly")

async def test_search_single_song_selection():
    print("--- Test 2: search_single_song video-aware selection ---")
    q = "i get high better alone nafr awannabe"

    # Case 1: Video candidate scores 1.0, song candidate scores 0.5 -> video candidate wins
    mock_songs = [{"videoId": "song_vid", "title": "Better Alone", "artists": [{"name": "Mile High Mafia"}], "duration": "180"}]
    mock_videos = [{"videoId": "vid_nafr", "title": "I get high better alone (feat. aWannabe)", "artists": [{"name": "NaFr"}], "duration": "180"}]

    def fake_search(query, filter=None, limit=5):
        if filter == "songs": return mock_songs
        if filter == "videos": return mock_videos
        return []

    with patch.object(bot.yt, 'search', side_effect=fake_search):
        best = await bot.search_single_song(q, source="ytmusic")
        assert best is not None
        assert best["video_id"] == "vid_nafr"
        assert best["resultType"] == "video"
        print("  [PASS] Video candidate scoring 1.0 beats song candidate scoring <1.0")

    # Case 2: Tie (both 1.0) -> song-type wins
    q_tie = "crawling linkin park"
    mock_songs_tie = [{"videoId": "song_cp", "title": "Crawling", "artists": [{"name": "Linkin Park"}], "duration": "180"}]
    mock_videos_tie = [{"videoId": "vid_cp", "title": "Crawling", "artists": [{"name": "Linkin Park"}], "duration": "180"}]

    def fake_search_tie(query, filter=None, limit=5):
        if filter == "songs": return mock_songs_tie
        if filter == "videos": return mock_videos_tie
        return []

    with patch.object(bot.yt, 'search', side_effect=fake_search_tie):
        best_tie = await bot.search_single_song(q_tie, source="ytmusic")
        assert best_tie is not None
        assert best_tie["video_id"] == "song_cp"
        assert best_tie["resultType"] == "song"
        print("  [PASS] Tie (both 1.0) selects song-type candidate")

    # Case 3: Both filters raise -> falls through to yt-dlp / None
    def fake_search_err(query, filter=None, limit=5):
        raise RuntimeError("Network down")

    with patch.object(bot.yt, 'search', side_effect=fake_search_err), \
         patch("asyncio.create_subprocess_exec", side_effect=RuntimeError("no yt-dlp")):
        best_none = await bot.search_single_song(q, source="ytmusic")
        assert best_none is None
        print("  [PASS] Both filters raising exception returns None cleanly")

async def test_start_walk_fallback_gate():
    print("--- Test 3: start_walk Direct Search fallback gate ---")
    mock_channel = MagicMock()
    mock_channel.send = AsyncMock()

    # Search returns a weak match (e.g. Mile High Mafia for NaFr query)
    weak_song = {"title": "Better Alone", "artist": "Mile High Mafia", "video_id": "mhm_vid", "url": "https://..."}
    bot.walk_state = {"tracks": [], "index": 0}

    with patch.object(bot, 'search_single_song', AsyncMock(return_value=weak_song)), \
         patch.object(bot, 'send_bridge_cmd', AsyncMock()) as mock_bridge, \
         patch.object(bot, 'speak', AsyncMock()) as mock_speak, \
         patch.object(bot, 'get_text_channel', AsyncMock(return_value=mock_channel)):

        bot.pick_gen = 1
        # Call start_walk with empty prefetched tracks and no candidates
        await bot.start_walk(query="i get high better alone nafr", source="ytmusic", gen_id=1, pre_resolved_candidates=[])

        # Assert no bridge cmd sent and walk_state tracks untouched
        mock_bridge.assert_not_called()
        assert bot.walk_state["tracks"] == []
        mock_speak.assert_called_once_with("Couldn't find i get high better alone nafr.")
        mock_channel.send.assert_awaited_once_with("❌ Couldn't find i get high better alone nafr.")
        print("  [PASS] Gate-rejected song is not queued and Honest 'Couldn't find' is spoken + texted")

async def test_start_walk_b2b_already_playing():
    print("--- Test 4: B2b already-playing fallback check ---")
    cur_vid = "playing_vid"
    strong_song = {"title": "My Track", "artist": "My Artist", "video_id": cur_vid, "url": f"https://youtube.com/watch?v={cur_vid}"}
    curated_tracks = [
        {"title": "Curated 1", "artist": "Artist 1", "video_id": "cur1", "url": "https://cur1"},
        {"title": "Curated 2", "artist": "Artist 2", "video_id": "cur2", "url": "https://cur2"}
    ]

    # Setup 1: interrupt = False and cur_vid matches song video_id -> Curated playlist installed
    bot.walk_state = {
        "tracks": [{"video_id": cur_vid, "title": "My Track", "artist": "My Artist"}],
        "index": 0
    }
    bot.pick_gen = 10

    with patch.object(bot, 'search_single_song', AsyncMock(return_value=strong_song)), \
         patch.object(bot, 'get_curated_playlist', return_value=("pid_test", "Rock Curated")), \
         patch.object(bot, 'fetch_playlist_tracks', AsyncMock(return_value=("Rock Curated", curated_tracks))), \
         patch.object(bot, 'send_bridge_cmd', AsyncMock()) as mock_bridge:

        await bot.start_walk(query="my track my artist", source="ytmusic", gen_id=10, interrupt=False, pre_resolved_candidates=[])

        # Assert NO bridge cmd for cur_vid
        for call in mock_bridge.call_args_list:
            cmd = call[0][0]
            assert cmd.get("video_id") != cur_vid, "cur_vid must NOT be queued again!"
        assert bot.walk_state["playlist_name"] == "Rock Curated"
        assert len(bot.walk_state["tracks"]) == 2
        print("  [PASS] interrupt=False with already playing track installs curated walk")

    # Setup 2: interrupt = True (fresh user command) -> queues track normally
    bot.walk_state = {
        "tracks": [{"video_id": cur_vid, "title": "My Track", "artist": "My Artist"}],
        "index": 0
    }
    bot.pick_gen = 11

    with patch.object(bot, 'search_single_song', AsyncMock(return_value=strong_song)), \
         patch.object(bot, 'send_bridge_cmd', AsyncMock()) as mock_bridge2:

        await bot.start_walk(query="my track my artist", source="ytmusic", gen_id=11, interrupt=True, pre_resolved_candidates=[])
        queued_vids = [call[0][0].get("video_id") for call in mock_bridge2.call_args_list if call[0][0].get("cmd") == "queue"]
        assert cur_vid in queued_vids
        assert bot.walk_state["playlist_name"] == "Direct Search"
        print("  [PASS] interrupt=True with explicit user request replays track normally")

async def test_fill_mix_queue_chain():
    print("--- Test 5: fill_mix_queue 4-attempt chain ---")
    seed_track = {"title": "Track Title", "artist": "Artist Name", "video_id": "seed_vid", "resultType": "video", "via": "YouTube"}
    bot.bridge_active = True
    bot.pick_gen = 20
    bot.walk_state = {"tracks": [dict(seed_track)], "mix_seed": seed_track, "max_tracks": 1}

    # Case A: Attempt 1 returns 1 usable track (< 3) -> Attempt 2 succeeds via song search
    att1_raw = {"tracks": [{"videoId": "seed_vid"}, {"videoId": "t1", "title": "T1", "artists": [{"name": "A1"}]}]}
    att2_search = [{"videoId": "song_match_vid", "title": "Track Title", "artists": [{"name": "Artist Name"}]}]
    att2_mix = {
        "tracks": [
            {"videoId": "song_match_vid"},
            {"videoId": "t2", "title": "T2", "artists": [{"name": "A2"}]},
            {"videoId": "t3", "title": "T3", "artists": [{"name": "A3"}]},
            {"videoId": "t4", "title": "T4", "artists": [{"name": "A4"}]}
        ]
    }

    call_counts = {"get_watch_playlist": 0, "search": 0}
    def fake_watch_playlist(videoId=None, radio=True, limit=25):
        call_counts["get_watch_playlist"] += 1
        if videoId == "seed_vid": return att1_raw
        if videoId == "song_match_vid": return att2_mix
        return {"tracks": []}

    def fake_search_songs(q, filter=None, limit=5):
        call_counts["search"] += 1
        return att2_search

    with patch.object(bot.yt, 'get_watch_playlist', side_effect=fake_watch_playlist), \
         patch.object(bot.yt, 'search', side_effect=fake_search_songs), \
         patch.object(bot, 'check_and_queue_ahead', AsyncMock()):

        await bot.fill_mix_queue(seed_track, gen_id=20)
        assert call_counts["get_watch_playlist"] == 2, "Expected attempt 1 direct seed then attempt 2 song match"
        assert len(bot.walk_state["tracks"]) == 5  # seed (1) + 4 usable from attempt 2
        print("  [PASS] Attempt 1 < 3 triggers Attempt 2 song-type retry successfully")

    # Case B: All attempts fail -> triggers re_roll_walk
    bot.walk_state = {"tracks": [dict(seed_track)], "mix_seed": seed_track, "max_tracks": 1}
    bot.pick_gen = 21

    with patch.object(bot.yt, 'get_watch_playlist', return_value={"tracks": []}), \
         patch.object(bot.yt, 'search', return_value=[]), \
         patch.object(bot, 're_roll_walk', AsyncMock()) as mock_reroll:

        await bot.fill_mix_queue(seed_track, gen_id=21)
        mock_reroll.assert_awaited_once_with(gen_id=21)
        assert bot.walk_state["mix_seed"] is None
        print("  [PASS] All attempts failing triggers silent re-roll fallback with mix_seed cleared")

async def test_advance_walk_guard_and_ensure_mix_fill():
    print("--- Test 6: advance_walk guard (A1) & ensure_mix_fill idempotency (A2) ---")
    seed = {"title": "Seed", "artist": "Artist", "video_id": "seed_vid", "url": "https://youtube.com/watch?v=seed_vid"}
    bot.bridge_active = True
    bot.pick_gen = 30
    bot.recent_played_vids.add("seed_vid")

    # A1 Case: Song route with seed currently playing, mix pending
    bot.walk_state = {
        "tracks": [dict(seed)],
        "index": 0,
        "tracks_played": 0,
        "max_tracks": 50,
        "mix_seed": seed,
        "mix_fill_gen": None
    }

    fill_called = []
    async def fake_fill(s, g):
        fill_called.append((s, g))

    with patch.object(bot, 'fill_mix_queue', side_effect=fake_fill), \
         patch.object(bot, 'send_bridge_cmd', AsyncMock()) as mock_bridge:

        await bot.advance_walk("keep_ahead", gen_id=30)
        await asyncio.sleep(0.01)
        mock_bridge.assert_not_called()
        assert len(fill_called) == 1, "ensure_mix_fill should trigger fill_mix_queue"
        print("  [PASS] advance_walk does not re-queue playing seed and triggers ensure_mix_fill")

        # A2 Idempotency: Calling ensure_mix_fill a second time with same gen does not trigger second task
        bot.ensure_mix_fill(30)
        await asyncio.sleep(0.01)
        assert len(fill_called) == 1, "ensure_mix_fill must be idempotent for same gen"
        print("  [PASS] ensure_mix_fill is idempotent on matching gen_id")

    # Non-song-route (mix_seed is None): old fallback queues track
    bot.walk_state = {
        "tracks": [dict(seed)],
        "index": 0,
        "tracks_played": 0,
        "max_tracks": 12,
        "mix_seed": None
    }
    with patch.object(bot, 'send_bridge_cmd', AsyncMock()) as mock_bridge2:
        await bot.advance_walk("keep_ahead", gen_id=30)
        queued_vids = [call[0][0].get("video_id") for call in mock_bridge2.call_args_list if call[0][0].get("cmd") == "queue"]
        assert "seed_vid" in queued_vids
        print("  [PASS] Non-song-route single-track preserves old fallback behavior")

async def test_mix_now_playing_edit():
    print("--- Test 7: A5 mix now-playing embed update ---")
    mock_msg = MagicMock()
    mock_msg.edit = AsyncMock()
    mock_msg.send = AsyncMock()
    mock_embed = MagicMock()
    mock_embed.thumbnail.url = "https://thumb.url"
    mock_embed.image.url = None
    mock_msg.embeds = [mock_embed]

    seed = {"title": "Seed Title", "artist": "Seed Artist", "video_id": "seed_vid", "resultType": "video", "via": "YouTube"}
    bot.bridge_active = True
    bot.pick_gen = 40
    bot.walk_state = {"tracks": [dict(seed)], "mix_seed": seed, "max_tracks": 1}
    bot.now_playing_ref = {"message": mock_msg, "video_id": "seed_vid", "gen": 40}

    watch_res = {
        "tracks": [
            {"videoId": "seed_vid"},
            {"videoId": "t1", "title": "T1", "artists": [{"name": "A1"}]},
            {"videoId": "t2", "title": "T2", "artists": [{"name": "A2"}]},
            {"videoId": "t3", "title": "T3", "artists": [{"name": "A3"}]}
        ]
    }

    with patch.object(bot.yt, 'get_watch_playlist', return_value=watch_res), \
         patch.object(bot, 'check_and_queue_ahead', AsyncMock()):

        await bot.fill_mix_queue(seed, gen_id=40)

        # Assert message.edit was called with updated embed (and NEVER sent)
        mock_msg.send.assert_not_called()
        mock_msg.edit.assert_awaited_once()
        _, kwargs = mock_msg.edit.call_args
        assert "content" not in kwargs, "content must be untouched so one-liner lives there"
        new_emb = kwargs["embed"]
        assert 'via "YouTube" · track 1/4' in new_emb.description
        assert "🎶 Okay, continuing with this vibe — 3 more queued." in new_emb.description
        assert new_emb.description.count("🎶 Okay, continuing with this vibe") == 1

        # Simulate second invocation: embed description already has vibe line
        mock_msg.embeds = [new_emb]
        await bot.fill_mix_queue(seed, gen_id=40)
        _, kwargs2 = mock_msg.edit.call_args
        new_emb2 = kwargs2["embed"]
        assert new_emb2.description.count("🎶 Okay, continuing with this vibe") == 1, "Vibe line must only appear once across multiple invocations"
        print("  [PASS] Now-playing message edited in place with 1/N, vibe line added exactly once across multiple invocations, and preserved via label")

async def main():
    test_score_and_strong_match()
    await test_search_single_song_selection()
    await test_start_walk_fallback_gate()
    await test_start_walk_b2b_already_playing()
    await test_fill_mix_queue_chain()
    await test_advance_walk_guard_and_ensure_mix_fill()
    await test_mix_now_playing_edit()
    print("\nALL BRIEF W-REVISED UNIT TESTS PASSED!")

if __name__ == "__main__":
    asyncio.run(main())
