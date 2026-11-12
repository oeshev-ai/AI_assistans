"""Handler registration."""

from __future__ import annotations

from aiogram import Dispatcher, F
from aiogram.filters import Command, CommandStart

from app.bot import runtime as R
from app.bot.constants import (
    _MENU_ASSISTANTS,
    _MENU_TOPICS,
    EditAssistantSG,
    ImageSG,
    NewAssistantSG,
    NewTopicSG,
    RenameTopicSG,
)
from app.bot.handlers.assistants import (
    on_assistants_page,
    on_asst_del_ask,
    on_asst_del_yes,
    on_asst_edit,
    on_asst_edit_ask,
    on_asst_icon_menu,
    on_asst_keep_prompt,
    on_asst_list,
    on_asst_new_commit,
    on_asst_new_icon,
    on_asst_new_icon_menu,
    on_asst_new_image,
    on_asst_new_start,
    on_asst_new_text,
    on_asst_rename_ask,
    on_asst_use,
    on_menu_assistants,
    on_set_persona_icon,
    sg_edit_name,
    sg_edit_prompt,
    sg_new_assistant_models,
    sg_new_assistant_name,
    sg_new_assistant_prompt,
)
from app.bot.handlers.chat import handle_text, handle_voice
from app.bot.handlers.commands import (
    _CommandLeavesWizard,
    cmd_cancel,
    cmd_start,
    cmd_unknown,
)
from app.bot.handlers.image import (
    on_image_go,
    on_image_model,
    on_image_size,
    sg_image_prompt,
)
from app.bot.handlers.media import (
    handle_audio,
    handle_document,
    handle_photo,
    handle_video_file,
    on_upload_cancel,
)
from app.bot.handlers.presets import (
    on_preset_apply,
    on_preset_edit,
    on_preset_view,
    on_presets_open,
    on_presets_page,
)
from app.bot.handlers.settings import (
    on_menu_settings,
    on_picker_done,
    on_set_image_menu,
    on_set_image_model,
    on_set_persona_size,
    on_set_text_menu,
    on_set_text_model,
    on_set_tts_model,
    on_settings_close,
    on_settings_open,
    on_settings_persona,
    on_voice_menu,
    on_voice_mode,
    on_voice_name,
    on_voice_role,
)
from app.bot.handlers.topics import (
    on_menu_topics,
    on_topic_clear_ask,
    on_topic_clear_yes,
    on_topic_del_ask,
    on_topic_del_yes,
    on_topic_delete_ask,
    on_topic_delete_yes,
    on_topic_list,
    on_topic_new,
    on_topic_rename,
    on_topic_use,
    on_topics_page,
    on_topics_refresh,
    sg_new_topic_name,
    sg_rename_topic_name,
)


def _register_handlers(dispatcher: Dispatcher) -> None:
    """Служебные команды: /start, /help, /cancel, /settings.

    Внизу — кнопки ассистентов и тем; остальное — inline.
    ``aiogram.filters.Text`` удалён в aiogram 3.31; любой некомандный текст
    ловит ``F.text``.
    """
    dispatcher.message.middleware(R._CleanChat())
    dispatcher.callback_query.middleware(R._CleanChat())
    dispatcher.message.middleware(_CommandLeavesWizard())
    dispatcher.message.register(cmd_start, CommandStart())
    dispatcher.message.register(cmd_start, Command("help"))
    dispatcher.message.register(cmd_cancel, Command("cancel"))
    dispatcher.message.register(on_menu_settings, Command("settings"))
    dispatcher.message.register(
        on_menu_assistants,
        F.text.startswith(f"{_MENU_ASSISTANTS} [")
        | F.text.startswith(f"{_MENU_ASSISTANTS} (")
        | F.text.startswith("Ассистенты [")
        | F.text.startswith("Ассистенты ("),
    )
    dispatcher.message.register(
        on_menu_topics,
        F.text.startswith(f"{_MENU_TOPICS} [") | F.text.startswith(f"{_MENU_TOPICS} ("),
    )
    dispatcher.callback_query.register(
        on_assistants_page, F.data.startswith("asst_page:")
    )
    dispatcher.callback_query.register(on_topics_page, F.data.startswith("topic_page:"))
    dispatcher.callback_query.register(
        on_asst_rename_ask, F.data.startswith("asst_rename:")
    )
    dispatcher.callback_query.register(
        on_asst_edit_ask, F.data.startswith("asst_edit:")
    )
    dispatcher.callback_query.register(on_topics_refresh, F.data == "topic_list")
    dispatcher.callback_query.register(
        on_topic_delete_ask, F.data.startswith("topic_del_ask:")
    )
    dispatcher.callback_query.register(
        on_topic_delete_yes, F.data.startswith("topic_del_yes:")
    )
    dispatcher.callback_query.register(
        on_upload_cancel, F.data.startswith("upload:cancel:")
    )
    dispatcher.message.register(cmd_unknown, F.text.startswith("/"))
    dispatcher.message.register(sg_new_assistant_name, NewAssistantSG.name)
    dispatcher.message.register(sg_edit_name, EditAssistantSG.name)
    dispatcher.message.register(sg_edit_prompt, EditAssistantSG.prompt)
    dispatcher.message.register(sg_new_assistant_prompt, NewAssistantSG.prompt)
    dispatcher.message.register(sg_new_assistant_models, NewAssistantSG.models)
    dispatcher.message.register(sg_new_topic_name, NewTopicSG.name)
    dispatcher.message.register(sg_rename_topic_name, RenameTopicSG.name)
    dispatcher.message.register(sg_image_prompt, ImageSG.prompt)
    dispatcher.callback_query.register(on_asst_list, F.data == "asst:list")
    dispatcher.callback_query.register(on_asst_use, F.data.startswith("asst:use:"))
    dispatcher.callback_query.register(on_asst_edit, F.data.startswith("asst:edit:"))
    dispatcher.callback_query.register(
        on_asst_del_ask, F.data.startswith("asst:del_ask:")
    )
    dispatcher.callback_query.register(
        on_asst_del_yes, F.data.startswith("asst:del_yes:")
    )
    dispatcher.callback_query.register(on_asst_new_start, F.data == "asst:new")
    dispatcher.callback_query.register(
        on_asst_new_text, F.data.startswith("asst:new_text:")
    )
    dispatcher.callback_query.register(
        on_asst_new_image, F.data.startswith("asst:new_image:")
    )
    dispatcher.callback_query.register(on_asst_new_commit, F.data == "asst:new_commit")
    dispatcher.callback_query.register(
        on_asst_new_icon_menu, F.data == "asst:new_icon_menu"
    )
    dispatcher.callback_query.register(
        on_asst_new_icon, F.data.startswith("asst:new_icon:")
    )
    dispatcher.callback_query.register(
        on_asst_keep_prompt, F.data == "asst:keep_prompt"
    )
    dispatcher.callback_query.register(on_asst_icon_menu, F.data == "asst_icon_menu")
    dispatcher.callback_query.register(
        on_set_persona_icon, F.data.startswith("set_icon:")
    )
    dispatcher.callback_query.register(on_presets_open, F.data == "preset:list")
    dispatcher.callback_query.register(
        on_presets_page, F.data.startswith("preset_page:")
    )
    dispatcher.callback_query.register(
        on_preset_view, F.data.startswith("preset:view:")
    )
    dispatcher.callback_query.register(
        on_preset_apply, F.data.startswith("preset:apply:")
    )
    dispatcher.callback_query.register(
        on_preset_edit, F.data.startswith("preset:edit:")
    )
    dispatcher.callback_query.register(on_topic_list, F.data == "topic:list")
    dispatcher.callback_query.register(on_topic_use, F.data.startswith("topic:use:"))
    dispatcher.callback_query.register(
        on_topic_clear_ask, F.data.startswith("topic:clear_ask:")
    )
    dispatcher.callback_query.register(
        on_topic_clear_yes, F.data.startswith("topic:clear_yes:")
    )
    dispatcher.callback_query.register(
        on_topic_rename, F.data.startswith("topic:rename:")
    )
    dispatcher.callback_query.register(
        on_topic_del_ask, F.data.startswith("topic:del_ask:")
    )
    dispatcher.callback_query.register(
        on_topic_del_yes, F.data.startswith("topic:del_yes:")
    )
    dispatcher.callback_query.register(on_topic_new, F.data == "topic:new")
    dispatcher.callback_query.register(
        on_image_model, F.data.startswith("image:model:")
    )
    dispatcher.callback_query.register(on_image_size, F.data.startswith("image:size:"))
    dispatcher.callback_query.register(on_image_go, F.data == "image:go")
    dispatcher.callback_query.register(on_settings_open, F.data == "settings:open")
    dispatcher.callback_query.register(on_settings_close, F.data == "settings:close")
    dispatcher.callback_query.register(
        on_settings_persona, F.data == "settings:persona"
    )
    dispatcher.callback_query.register(on_set_text_menu, F.data == "set_text_menu")
    dispatcher.callback_query.register(on_set_image_menu, F.data == "set_image_menu")
    dispatcher.callback_query.register(on_picker_done, F.data == "picker:done")
    dispatcher.callback_query.register(on_voice_menu, F.data == "voice_menu")
    dispatcher.callback_query.register(
        on_set_persona_size, F.data.startswith("set_size:")
    )
    dispatcher.callback_query.register(on_voice_mode, F.data.startswith("voice_mode:"))
    dispatcher.callback_query.register(on_voice_name, F.data.startswith("voice_name:"))
    dispatcher.callback_query.register(on_voice_role, F.data.startswith("voice_role:"))
    dispatcher.callback_query.register(
        on_set_text_model, F.data.startswith("set_text:")
    )
    dispatcher.callback_query.register(
        on_set_image_model, F.data.startswith("set_image:")
    )
    dispatcher.callback_query.register(on_set_tts_model, F.data.startswith("set_tts:"))
    dispatcher.message.register(handle_voice, F.voice)
    dispatcher.message.register(handle_audio, F.audio)
    dispatcher.message.register(handle_video_file, F.video | F.video_note)
    dispatcher.message.register(handle_photo, F.photo)
    dispatcher.message.register(handle_document, F.document)
    dispatcher.message.register(handle_text, F.text & ~F.text.startswith("/"))
