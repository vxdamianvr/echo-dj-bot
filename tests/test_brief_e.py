import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import AsyncMock, MagicMock, patch
import discord
import bot

async def test_e1_persona_picker_and_switch():
    print("--- Testing E1: Persona Switch & Picker Lifecycle ---")
    bot.last_picker_message = None
    bot.current_persona = "hype"

    mock_channel = MagicMock()
    new_msg1 = MagicMock()
    new_msg1.id = 1002
    new_msg1.edit = AsyncMock()
    mock_channel.send = AsyncMock(return_value=new_msg1)

    view1 = bot.PersonaView()
    select1 = view1.children[0]
    select1._interaction = MagicMock()
    select1._interaction.data = {"values": ["unhinged"]}
    select1._selected_values = ["unhinged"]

    mock_inter1 = MagicMock()
    mock_inter1.message = MagicMock()
    mock_inter1.message.id = 1001
    mock_inter1.response.edit_message = AsyncMock()

    with patch.object(bot, 'get_text_channel', AsyncMock(return_value=mock_channel)):
        # 1. Switch to Unhinged
        await select1.callback(mock_inter1)

        mock_inter1.response.edit_message.assert_awaited_once_with(
            content="Persona locked in: **Unhinged**.",
            view=None
        )
        assert bot.last_picker_message == new_msg1, "last_picker_message should be updated to new channel message"
        assert bot.current_persona == "unhinged", "current_persona should be unhinged"
        mock_channel.send.assert_awaited_once()
        args, kwargs = mock_channel.send.call_args
        assert args[0] == "Persona switched to: **Unhinged** — want to change again?"
        assert isinstance(kwargs['view'], bot.PersonaView)
        print("  [PASS] Switch 1: interaction died, fresh picker posted at bottom")

        # 2. Switch to Velvet
        new_msg2 = MagicMock()
        new_msg2.id = 1004
        new_msg2.edit = AsyncMock()
        mock_channel.send = AsyncMock(return_value=new_msg2)

        view2 = bot.PersonaView()
        select2 = view2.children[0]
        select2._interaction = MagicMock()
        select2._interaction.data = {"values": ["velvet"]}
        select2._selected_values = ["velvet"]

        mock_inter2 = MagicMock()
        mock_inter2.message = MagicMock()
        mock_inter2.message.id = 1003
        mock_inter2.response.edit_message = AsyncMock()

        await select2.callback(mock_inter2)

        # Previous live picker (new_msg1) must have had its view removed
        new_msg1.edit.assert_awaited_once_with(view=None)
        mock_inter2.response.edit_message.assert_awaited_once_with(
            content="Persona locked in: **Velvet**.",
            view=None
        )
        assert bot.last_picker_message == new_msg2, "last_picker_message should be updated to new_msg2"
        assert bot.current_persona == "velvet", "current_persona should be velvet"
        print("  [PASS] Switch 2: previous picker retired, new picker posted, persona updated")

        # 3. /persona slash command
        ctx = MagicMock()
        ctx.defer = AsyncMock()
        slash_msg = MagicMock()
        slash_msg.id = 1005
        slash_msg.edit = AsyncMock()
        ctx.followup.send = AsyncMock(return_value=slash_msg)

        await bot.persona_cmd(ctx)

        ctx.defer.assert_awaited_once()
        # Previous live picker (new_msg2) must have been retired
        new_msg2.edit.assert_awaited_once_with(view=None)
        ctx.followup.send.assert_awaited_once()
        f_args, f_kwargs = ctx.followup.send.call_args
        assert f_args[0] == "Choose Echo's persona:"
        assert isinstance(f_kwargs['view'], bot.PersonaView)
        assert bot.last_picker_message == slash_msg
        print("  [PASS] /persona command: retired previous picker, returned fresh picker")

async def test_e2_e4_embed_thumbnail_and_sentiment_append():
    print("--- Testing E2 & E4: Embed Thumbnail & Sentiment Append ---")
    mock_channel = MagicMock()
    mock_message = MagicMock()
    mock_message.id = 2001
    mock_message.edit = AsyncMock()
    mock_channel.send = AsyncMock(return_value=mock_message)

    with patch.object(bot, 'get_text_channel', AsyncMock(return_value=mock_channel)):
        # Test announce_track_start
        with patch.object(bot, 'post_lyrics', AsyncMock()) as mock_post_lyrics, \
             patch.object(bot, '_append_sentiment', AsyncMock()) as mock_append_sentiment:
            mock_message.create_thread = AsyncMock(return_value=MagicMock())

            await bot.announce_track_start("Test Song", "Test Artist", "https://youtube.com/watch?v=vid123", "vid123")

            mock_channel.send.assert_awaited_once()
            _, kwargs = mock_channel.send.call_args
            embed = kwargs['embed']
            assert embed.thumbnail.url == "https://i.ytimg.com/vi/vid123/hqdefault.jpg", "Thumbnail should match YT video_id"
            assert "Now playing: [Test Song — Test Artist](https://youtube.com/watch?v=vid123)" in embed.description
            print("  [PASS] E2: Thumbnail set correctly on now-playing embed")

            # Check thread created and passed
            mock_message.create_thread.assert_awaited_once_with(name="Lyrics: Test Song", auto_archive_duration=60)
            mock_post_lyrics.assert_called_once()
            mock_append_sentiment.assert_called_once()
            print("  [PASS] E3: Lyrics thread created under message and passed to post_lyrics")

        # Test _append_sentiment
        base_desc = "Now playing: [Test Song — Test Artist](https://youtube.com/watch?v=vid123)\nvia \"Curated\" · track 1/1"
        with patch.object(bot, 'ask_llm_sentiment', AsyncMock(return_value="Certified banger.")):
            await bot._append_sentiment(mock_message, base_desc, "Test Song", "Test Artist", "Curated", video_id="vid123")

            mock_message.edit.assert_awaited_once()
            _, edit_kwargs = mock_message.edit.call_args
            edited_embed = edit_kwargs['embed']
            lines = edited_embed.description.split("\n")
            assert len(lines) == 3, f"Expected 3 lines, got: {lines}"
            assert lines[0] == "Now playing: [Test Song — Test Artist](https://youtube.com/watch?v=vid123)"
            assert lines[1] == 'via "Curated" · track 1/1'
            assert lines[2] == "Certified banger."
            # Verify thumbnail was NOT wiped
            assert edited_embed.thumbnail.url == "https://i.ytimg.com/vi/vid123/hqdefault.jpg", "Thumbnail must be preserved after edit"
            print("  [PASS] E4: Sentiment appended as 3rd line and thumbnail preserved")

async def test_e3_lyrics_threading_and_miss_handling():
    print("--- Testing E3: Lyrics Threading & Miss Handling ---")
    mock_channel = MagicMock()
    mock_channel.send = AsyncMock()

    mock_thread = MagicMock()
    mock_thread.send = AsyncMock()
    mock_thread.delete = AsyncMock()

    with patch.object(bot, 'get_text_channel', AsyncMock(return_value=mock_channel)):
        # Case A: Lyrics found -> sent to thread
        with patch.object(bot, 'fetch_lyrics', AsyncMock(return_value="Line 1\nLine 2")):
            await bot.post_lyrics("Hit Song", "Pop Star", thread=mock_thread)
            mock_thread.send.assert_awaited_once()
            mock_channel.send.assert_not_called()
            mock_thread.delete.assert_not_called()
            print("  [PASS] Lyrics found -> posted into thread")

        # Case B: Lyrics miss -> delete thread, post miss notice to main channel
        mock_thread.reset_mock()
        mock_channel.reset_mock()
        with patch.object(bot, 'fetch_lyrics', AsyncMock(return_value=None)):
            await bot.post_lyrics("Obscure Instrumental", "Unknown", thread=mock_thread)
            mock_thread.delete.assert_awaited_once()
            mock_thread.send.assert_not_called()
            mock_channel.send.assert_awaited_once_with("🎵 Couldn't find lyrics for Obscure Instrumental.")
            print("  [PASS] Lyrics miss -> thread deleted, notice posted to main channel")

async def main():
    await test_e1_persona_picker_and_switch()
    await test_e2_e4_embed_thumbnail_and_sentiment_append()
    await test_e3_lyrics_threading_and_miss_handling()
    print("\nALL UNIT & INTEGRATION TESTS PASSED FOR BRIEF E!")

if __name__ == "__main__":
    asyncio.run(main())
