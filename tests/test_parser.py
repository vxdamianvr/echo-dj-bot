import os
import sys

# Add project root to sys.path so bot can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bot import parse_voice_command, is_play_debounced

def test_a5_cases():
    # 1. "Echo, skip." -> skip
    res = parse_voice_command("Echo, skip.")
    assert res["action"] == "skip", f"Expected skip, got {res}"

    # 2. Trailing speech after skip
    res = parse_voice_command("Echo, skip now")
    assert res["action"] == "skip", f"Expected skip, got {res}"
    res = parse_voice_command("Echo skip please")
    assert res["action"] == "skip", f"Expected skip, got {res}"

    # 3. "Echo pause the music please" -> pause
    res = parse_voice_command("Echo pause the music please")
    assert res["action"] == "pause", f"Expected pause, got {res}"

    # 4. "Echo play jazz. How do you hear it" -> query "jazz"
    res = parse_voice_command("Echo play jazz. How do you hear it")
    assert res["action"] == "play", f"Expected play, got {res}"
    assert res["query"] == "jazz", f"Expected query 'jazz', got '{res.get('query')}'"
    assert res["source"] == "ytmusic", f"Expected source 'ytmusic', got '{res.get('source')}'"

    # 5. "Echo, I want to skip this" -> unparseable
    res = parse_voice_command("Echo, I want to skip this")
    assert res["action"] == "unparseable", f"Expected unparseable, got {res}"

    # 6. "Echo play radiohead" -> query "radiohead"
    res = parse_voice_command("Echo play radiohead")
    assert res["action"] == "play", f"Expected play, got {res}"
    assert res["query"] == "radiohead", f"Expected query 'radiohead', got '{res.get('query')}'"
    assert res["source"] == "ytmusic", f"Expected source 'ytmusic', got '{res.get('source')}'"

    # Extra controls
    res = parse_voice_command("Echo back")
    assert res["action"] == "back", f"Expected back, got {res}"

    res = parse_voice_command("Echo resume")
    assert res["action"] == "resume", f"Expected resume, got {res}"

    res = parse_voice_command("Echo stop")
    assert res["action"] == "stop", f"Expected stop, got {res}"

    res = parse_voice_command("Echo play lofi on soundcloud")
    assert res["action"] == "play" and res["query"] == "lofi" and res["source"] == "soundcloud", f"Got {res}"

    print("PASS: test_a5_cases")

def test_a6_debounce_cases():
    walk_state = {
        "query": "jazz",
        "tracks": [{"title": "Song 1"}],
        "started_ts": 100.0
    }

    # Same query at t=0 (delta = 0s < 30s) -> blocked
    assert is_play_debounced("jazz", walk_state, now=100.0) is True, "Expected blocked at t=0"
    assert is_play_debounced("Jazz ", walk_state, now=110.0) is True, "Expected blocked within 30s window"

    # Same query at t=31 (delta = 31s >= 30s) -> passes
    assert is_play_debounced("jazz", walk_state, now=131.0) is False, "Expected pass at t=31"

    # Different query -> passes
    assert is_play_debounced("radiohead", walk_state, now=100.0) is False, "Expected different query to pass"

    print("PASS: test_a6_debounce_cases")

if __name__ == "__main__":
    test_a5_cases()
    test_a6_debounce_cases()
    print("ALL TESTS PASSED")
