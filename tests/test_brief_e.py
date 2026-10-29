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

async def test_announce_track_start_and_large_artwork():
    print("--- Testing F1: Embed Large Image & Thread Creation ---")
    mock_channel = MagicMock()
    mock_message = MagicMock()
    mock_message.id = 2001
    mock_message.edit = AsyncMock()
    mock_channel.send = AsyncMock(return_value=mock_message)

    with patch.object(bot, 'get_text_channel', AsyncMock(return_value=mock_channel)):
        with patch.object(bot, '_finalize_now_playing', AsyncMock()) as mock_finalize:
            mock_message.create_thread = AsyncMock(return_value=MagicMock())

            await bot.announce_track_start("Test Song", "Test Artist", "https://youtube.com/watch?v=vid123", "vid123")

            mock_channel.send.assert_awaited_once()
            _, kwargs = mock_channel.send.call_args
            embed = kwargs['embed']
            assert embed.image.url == "https://i.ytimg.com/vi/vid123/hqdefault.jpg", "Image should match YT video_id"
            assert "Now playing: [Test Song — Test Artist](https://youtube.com/watch?v=vid123)" in embed.description
            print("  [PASS] F1: Large image set correctly on now-playing embed")

            # Check thread created and passed
            mock_message.create_thread.assert_awaited_once_with(name="Lyrics: Test Song", auto_archive_duration=60)
            mock_finalize.assert_called_once()
            print("  [PASS] E3: Lyrics thread created under message and passed to _finalize_now_playing")

        # Test announce_track_start without video_id (no image)
        mock_channel.send.reset_mock()
        with patch.object(bot, '_finalize_now_playing', AsyncMock()):
            await bot.announce_track_start("No Vid Song", "Artist", "https://example.com/audio.mp3", None)
            mock_channel.send.assert_awaited_once()
            _, no_vid_kwargs = mock_channel.send.call_args
            no_vid_embed = no_vid_kwargs['embed']
            assert no_vid_embed.image is None or no_vid_embed.image.url is None, "No image when video_id is absent"
            print("  [PASS] F1: No image set when video_id is absent")

async def test_g1_finalize_now_playing():
    print("--- Testing G1 & H1: Finalize Now Playing (Emoji Bookends & Inline Miss) ---")
    base_desc = "Now playing: [Test Song — Test Artist](https://youtube.com/watch?v=vid123)\nvia \"Curated\" · track 1/1"
    bot.current_persona = "velvet"

    # Case A: Lyrics found
    mock_channel = MagicMock()
    mock_channel.send = AsyncMock()
    mock_message = MagicMock()
    mock_message.id = 3001
    mock_message.edit = AsyncMock()
    mock_thread = MagicMock()
    mock_thread.send = AsyncMock()
    mock_thread.delete = AsyncMock()

    with patch.object(bot, 'get_text_channel', AsyncMock(return_value=mock_channel)), \
         patch.object(bot, 'fetch_lyrics', AsyncMock(return_value="Line 1\nLine 2")), \
         patch.object(bot, 'ask_llm_sentiment', AsyncMock(return_value="Certified banger.")):

        await bot._finalize_now_playing(mock_message, mock_thread, "Hit Song", "Pop Star", "Curated", "vid123", base_desc=base_desc)

        # First edit sets sentiment as message content with persona emoji
        mock_message.edit.assert_awaited_once()
        _, edit_kwargs = mock_message.edit.call_args
        assert edit_kwargs['content'] == "🍸 Certified banger.", "One-liner must start with persona emoji"
        assert edit_kwargs['content'].startswith("🍸")
        assert not edit_kwargs['content'].endswith("🎵")
        edited_embed = edit_kwargs['embed']
        assert edited_embed.description == base_desc, "Embed description must contain only the 2 base info lines"
        assert edited_embed.image.url == "https://i.ytimg.com/vi/vid123/hqdefault.jpg", "Large image must be preserved"

        # Lyrics sent to thread, thread not deleted, no miss indicator
        mock_thread.send.assert_awaited_once()
        mock_thread.delete.assert_not_called()
        mock_channel.send.assert_not_called()
        assert "no lyrics found" not in edit_kwargs['content']
        print("  [PASS] G1/H1 Case A: One-liner starts with persona emoji, lyrics in thread, no miss indicator")

    # Case B: Lyrics miss -> inline indicator appended with trailing 🎵, thread deleted, no separate channel message
    bot.current_persona = "unhinged"
    mock_channel.reset_mock()
    mock_message.reset_mock()
    mock_message.edit = AsyncMock()
    mock_thread.reset_mock()
    mock_thread.send = AsyncMock()
    mock_thread.delete = AsyncMock()

    with patch.object(bot, 'get_text_channel', AsyncMock(return_value=mock_channel)), \
         patch.object(bot, 'fetch_lyrics', AsyncMock(return_value=None)), \
         patch.object(bot, 'ask_llm_sentiment', AsyncMock(return_value="Late night vibe.")):

        await bot._finalize_now_playing(mock_message, mock_thread, "Obscure Song", "Unknown", "Curated", "vid123", base_desc=base_desc)

        # Expected 2 edits: (1) content="🤪 Late night vibe.", (2) content="🤪 Late night vibe. · no lyrics found 🎵"
        assert mock_message.edit.await_count == 2
        calls = mock_message.edit.call_args_list
        assert calls[0].kwargs['content'] == "🤪 Late night vibe."
        second_edit_content = calls[1].kwargs['content']
        assert second_edit_content == "🤪 Late night vibe. · no lyrics found 🎵"
        assert second_edit_content.startswith("🤪")
        assert second_edit_content.endswith("· no lyrics found 🎵")
        assert "· 🎵 no lyrics found" not in second_edit_content

        # Thread deleted, thread send not called, channel send not called
        mock_thread.delete.assert_awaited_once()
        mock_thread.send.assert_not_called()
        mock_channel.send.assert_not_called()
        print("  [PASS] G1/H1 Case B: Starts with emoji, ends with '· no lyrics found 🎵', thread deleted, no separate channel message")

async def main():
    await test_e1_persona_picker_and_switch()
    await test_announce_track_start_and_large_artwork()
    await test_g1_finalize_now_playing()
    print("\nALL UNIT & INTEGRATION TESTS PASSED FOR BRIEFS E, F & G!")

if __name__ == "__main__":
    asyncio.run(main())
