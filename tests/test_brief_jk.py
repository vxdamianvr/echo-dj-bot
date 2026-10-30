import asyncio
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import AsyncMock, MagicMock, patch
import discord
import bot

async def test_case_a_mention_strip_and_parse():
    print("--- Test (a): Mention-strip and parse ---")
    raw = "<@123456789> play creep"
    remainder = re.sub(r'<@!?\d+>', '', raw).strip()
    assert remainder == "play creep"
    parsed = bot.parse_voice_command(remainder)
    assert parsed["action"] == "play"
    assert parsed["query"] == "creep"

    raw2 = "<@!987654321>   radio"
    remainder2 = re.sub(r'<@!?\d+>', '', raw2).strip()
    assert remainder2 == "radio"
    parsed2 = bot.parse_voice_command(remainder2)
    assert parsed2["action"] == "radio"
    print("  [PASS] Test (a): Mention stripped and correctly parsed")

async def test_case_b_bare_mention():
    print("--- Test (b): Bare mention routes to persona picker ---")
    mock_msg = MagicMock()
    mock_msg.author = MagicMock()
    mock_msg.author.bot = False
    mock_msg.author.id = 42
    mock_msg.channel.id = bot.TEXT_CHANNEL_ID
    mock_msg.channel.send = AsyncMock()
    mock_msg.content = "<@123456789>"

    mock_user = MagicMock()
    mock_user.mentioned_in = MagicMock(return_value=True)

    with patch.object(bot.bot._connection, 'user', mock_user), \
         patch.object(bot, 'retire_picker_message', AsyncMock()) as mock_retire:
        await bot.on_message(mock_msg)
        mock_retire.assert_awaited_once()
        mock_msg.channel.send.assert_awaited_once()
        args, kwargs = mock_msg.channel.send.call_args
        assert args[0] == "Choose Echo's persona:"
        assert isinstance(kwargs['view'], bot.PersonaView)
    print("  [PASS] Test (b): Bare mention retired previous picker and posted fresh one")

async def test_case_c_unparseable_mention_routes_to_chat():
    print("--- Test (c): Unparseable mention routes to chat, never to handle_voice_command ---")
    mock_msg = MagicMock()
    mock_msg.author = MagicMock()
    mock_msg.author.bot = False
    mock_msg.author.id = 42
    mock_msg.channel.id = bot.TEXT_CHANNEL_ID
    mock_msg.content = "<@123456789> what is the meaning of life"

    mock_user = MagicMock()
    mock_user.mentioned_in = MagicMock(return_value=True)

    with patch.object(bot.bot._connection, 'user', mock_user), \
         patch.object(bot, 'handle_voice_command', AsyncMock()) as mock_voice, \
         patch.object(bot, 'handle_mention_chat', AsyncMock()) as mock_chat:
        await bot.on_message(mock_msg)
        mock_chat.assert_called_once_with(mock_msg, "what is the meaning of life")
        mock_voice.assert_not_called()
    print("  [PASS] Test (c): Unparseable mention routed to chat; voice handler not called")

async def test_case_d_chat_cooldown():
    print("--- Test (d): Chat cooldown enforces 10s window ---")
    bot.last_chat_reply_ts = 0.0

    mock_msg1 = MagicMock()
    mock_msg1.author.display_name = "Alice"
    mock_msg1.author.bot = False
    mock_msg1.reply = AsyncMock()
    class AsyncIter:
        def __init__(self, items):
            self.items = list(items)
        def __aiter__(self):
            return self
        async def __anext__(self):
            if not self.items:
                raise StopAsyncIteration
            return self.items.pop(0)

    mock_msg1.channel.history = MagicMock(return_value=AsyncIter([]))

    with patch.object(bot, 'ask_llm', AsyncMock(return_value="First reply")):
        await bot.handle_mention_chat(mock_msg1, "Hello 1")
        mock_msg1.reply.assert_awaited_once_with("First reply", mention_author=False)
        assert bot.last_chat_reply_ts > 0.0

    mock_msg2 = MagicMock()
    mock_msg2.author.display_name = "Bob"
    mock_msg2.author.bot = False
    mock_msg2.reply = AsyncMock()

    with patch.object(bot, 'ask_llm', AsyncMock(return_value="Second reply")):
        # Within 10s cooldown
        await bot.handle_mention_chat(mock_msg2, "Hello 2")
        mock_msg2.reply.assert_not_called()
    print("  [PASS] Test (d): Second chat call within 10s returned silently")

async def test_case_e_not_in_voice_skip():
    print("--- Test (e): Not in voice + @Echo skip ---")
    bot.bridge_active = False

    mock_msg = MagicMock()
    mock_msg.author.bot = False
    mock_msg.author.id = 42
    mock_msg.channel.id = bot.TEXT_CHANNEL_ID
    mock_msg.reply = AsyncMock()
    mock_msg.content = "<@123456789> skip"

    mock_user = MagicMock()
    mock_user.mentioned_in = MagicMock(return_value=True)

    with patch.object(bot.bot._connection, 'user', mock_user), \
         patch.object(bot, 'send_bridge_cmd', AsyncMock()) as mock_bridge:
        await bot.on_message(mock_msg)
        mock_msg.reply.assert_awaited_once_with("Not in a voice channel right now.", mention_author=False)
        mock_bridge.assert_not_called()
    print("  [PASS] Test (e): Replied 'Not in a voice channel right now.' and no bridge command sent")

async def test_case_f_not_in_voice_play():
    print("--- Test (f): Not in voice + @Echo play x autojoins ---")
    bot.bridge_active = False

    mock_msg = MagicMock()
    mock_msg.author.bot = False
    mock_msg.author.id = 42
    mock_msg.channel.id = bot.TEXT_CHANNEL_ID
    mock_msg.channel.send = AsyncMock(return_value=MagicMock())
    mock_msg.content = "<@123456789> play jazz"

    mock_user = MagicMock()
    mock_user.mentioned_in = MagicMock(return_value=True)

    with patch.object(bot.bot._connection, 'user', mock_user), \
         patch.object(bot, 'ensure_voice_joined', AsyncMock(return_value=True)) as mock_join, \
         patch.object(bot, 'handle_voice_command', AsyncMock()) as mock_voice:
        await bot.on_message(mock_msg)
        mock_join.assert_awaited_once()
        assert mock_msg.channel.send.await_count == 2
        calls = mock_msg.channel.send.call_args_list
        assert calls[0][0][0] == "Joining voice… 🎧"
        assert "Joined and listening" in calls[1][0][0]
        mock_voice.assert_awaited_once_with("play jazz", 42)
    print("  [PASS] Test (f): Autojoined voice channel and dispatched play command")

async def test_case_g_lyrics_snippet_and_timeout():
    print("--- Test (g): K lyrics snippet passed to sentiment & timeout protection ---")
    bot.LYRICS_IN_SENTIMENT = True
    base_desc = "Now playing: [Test Song — Test Artist](https://youtube.com/watch?v=vid123)\nvia \"Curated\" · track 1/1"

    # Part 1: Fast lyrics -> snippet passed
    mock_message = MagicMock()
    mock_message.id = 7001
    mock_message.edit = AsyncMock()
    mock_thread = MagicMock()
    mock_thread.send = AsyncMock()

    sample_lyrics = "First line of the song\nSecond line of the song\nThird line of the song"
    with patch.object(bot, 'get_text_channel', AsyncMock(return_value=MagicMock())), \
         patch.object(bot, 'fetch_lyrics', AsyncMock(return_value=sample_lyrics)), \
         patch.object(bot, 'ask_llm_sentiment', AsyncMock(return_value="Cool vibes.")) as mock_sentiment:
        await bot._finalize_now_playing(mock_message, mock_thread, "Title", "Artist", "Curated", "vid123", base_desc=base_desc)
        mock_sentiment.assert_awaited_once()
        _, kwargs = mock_sentiment.call_args
        assert kwargs.get("lyrics_snippet") is not None
        assert "First line" in kwargs["lyrics_snippet"]
        print("  [PASS] Test (g.1): Lyrics snippet passed into ask_llm_sentiment")

    # Part 2: Slow lyrics (>3s timeout) -> one-liner proceeds without waiting
    async def slow_lyrics(*args, **kwargs):
        await asyncio.sleep(5)
        return "Slow lyrics"

    mock_message.reset_mock()
    with patch.object(bot, 'get_text_channel', AsyncMock(return_value=MagicMock())), \
         patch.object(bot, 'fetch_lyrics', slow_lyrics), \
         patch.object(bot, 'ask_llm_sentiment', AsyncMock(return_value="Quick reply.")) as mock_sentiment_slow:
        # We replace the 3.0s timeout with 0.1s for test speed
        with patch('asyncio.wait_for', side_effect=asyncio.TimeoutError):
            await bot._finalize_now_playing(mock_message, mock_thread, "Slow Title", "Slow Artist", "Curated", "vid123", base_desc=base_desc)
            mock_sentiment_slow.assert_awaited_once()
            _, slow_kwargs = mock_sentiment_slow.call_args
            assert slow_kwargs.get("lyrics_snippet") is None
            print("  [PASS] Test (g.2): Slow lyrics timeout shielded; one-liner posted without snippet")

async def test_case_h_lyrics_in_sentiment_flag():
    print("--- Test (h): K LYRICS_IN_SENTIMENT = False disables excerpt ---")
    bot.LYRICS_IN_SENTIMENT = False

    prompt_captured = None
    def mock_post(url, json=None, timeout=None):
        nonlocal prompt_captured
        prompt_captured = json.get("prompt")
        resp = MagicMock()
        resp.status = 200
        resp.json = AsyncMock(return_value={"response": "No lyrics test."})
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=resp)
        cm.__aexit__ = AsyncMock(return_value=None)
        return cm

    mock_session = MagicMock()
    mock_session.post = mock_post

    with patch('aiohttp.ClientSession', return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_session), __aexit__=AsyncMock(return_value=None))):
        await bot.ask_llm_sentiment("Title", "Artist", "Playlist", lyrics_snippet=None)
        assert prompt_captured is not None
        assert "Song lyrics (excerpt" not in prompt_captured
        print("  [PASS] Test (h): No lyrics excerpt section present when disabled or None")

    bot.LYRICS_IN_SENTIMENT = True

async def test_case_i_picker_invariant():
    print("--- Test (i): Picker invariant: every join / auto-join retires previous picker ---")
    
    # 1. join(ctx) twice in a row: first picker message receives edit(view=None), last_picker_message is the second
    bot.bridge_active = False
    bot.last_picker_message = None
    
    mock_msg1 = MagicMock()
    mock_msg1.edit = AsyncMock()
    mock_msg2 = MagicMock()
    mock_msg2.edit = AsyncMock()
    
    mock_ctx1 = MagicMock()
    mock_ctx1.defer = AsyncMock()
    mock_ctx1.followup.send = AsyncMock(return_value=mock_msg1)
    
    mock_ctx2 = MagicMock()
    mock_ctx2.defer = AsyncMock()
    mock_ctx2.followup.send = AsyncMock(return_value=mock_msg2)
    
    with patch.object(bot, 'ensure_voice_joined', AsyncMock(return_value=True)):
        # First join
        await bot.join(ctx=mock_ctx1)
        assert bot.last_picker_message == mock_msg1
        mock_msg1.edit.assert_not_called()
        
        # Second join (bridge_active is now simulated as True or False)
        # Testing join() again:
        await bot.join(ctx=mock_ctx2)
        mock_msg1.edit.assert_awaited_once_with(view=None)
        assert bot.last_picker_message == mock_msg2

    # 2. join(ctx=None) via channel.send twice in a row
    mock_ch_msg1 = MagicMock()
    mock_ch_msg1.edit = AsyncMock()
    mock_ch_msg2 = MagicMock()
    mock_ch_msg2.edit = AsyncMock()
    
    mock_channel = MagicMock()
    mock_channel.send = AsyncMock(side_effect=[mock_ch_msg1, mock_ch_msg2])
    
    bot.bridge_active = False
    bot.last_picker_message = None
    with patch.object(bot, 'get_text_channel', AsyncMock(return_value=mock_channel)), \
         patch.object(bot, 'ensure_voice_joined', AsyncMock(return_value=True)):
        await bot.join(ctx=None)
        assert bot.last_picker_message == mock_ch_msg1
        mock_ch_msg1.edit.assert_not_called()
        
        bot.bridge_active = True
        await bot.join(ctx=None)
        mock_ch_msg1.edit.assert_awaited_once_with(view=None)
        assert bot.last_picker_message == mock_ch_msg2

    # 3. Simulate the @Echo mention auto-join path with a stored picker
    bot.bridge_active = False
    prior_picker = MagicMock()
    prior_picker.edit = AsyncMock()
    bot.last_picker_message = prior_picker
    
    new_picker = MagicMock()
    mock_mention_msg = MagicMock()
    mock_mention_msg.author.bot = False
    mock_mention_msg.author.id = 42
    mock_mention_msg.channel.id = bot.TEXT_CHANNEL_ID
    mock_mention_msg.channel.send = AsyncMock(side_effect=[MagicMock(), new_picker])
    mock_mention_msg.content = "<@123456789> play jazz"

    mock_user = MagicMock()
    mock_user.mentioned_in = MagicMock(return_value=True)

    with patch.object(bot.bot._connection, 'user', mock_user), \
         patch.object(bot, 'ensure_voice_joined', AsyncMock(return_value=True)), \
         patch.object(bot, 'handle_voice_command', AsyncMock()):
        await bot.on_message(mock_mention_msg)
        prior_picker.edit.assert_awaited_once_with(view=None)
        assert bot.last_picker_message == new_picker

    print("  [PASS] Test (i): All picker invariant paths retire previous picker properly")

async def main():
    await test_case_a_mention_strip_and_parse()
    await test_case_b_bare_mention()
    await test_case_c_unparseable_mention_routes_to_chat()
    await test_case_d_chat_cooldown()
    await test_case_e_not_in_voice_skip()
    await test_case_f_not_in_voice_play()
    await test_case_g_lyrics_snippet_and_timeout()
    await test_case_h_lyrics_in_sentiment_flag()
    await test_case_i_picker_invariant()
    print("\nALL J+K TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    asyncio.run(main())
