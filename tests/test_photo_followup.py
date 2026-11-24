"""Повторная правка: результат остаётся в pending, пустая подпись берёт просьбу."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.bot.photo_flow import caption_from_resend, result_file_id
from app.bot.photo_router import PhotoDecision
from app.storage.pending_media import PendingMedia, PendingMediaItem


def _history(*rows: tuple[str, str]) -> list[dict[str, str]]:
    return [{"role": role, "content": content} for role, content in rows]


def test_caption_reuses_request_after_resend_ask() -> None:
    text = caption_from_resend(
        _history(
            ("user", "поправь небо"),
            ("assistant", "Картинка уже недоступна. Пришлите фото ещё раз."),
        )
    )
    assert text == "поправь небо"


def test_album_placeholder_is_not_a_caption() -> None:
    text = caption_from_resend(
        _history(
            ("user", "[Альбом: 1 фото, основа — фото 1] поправь небо"),
            ("assistant", "Фото недоступно, пришлите картинку снова."),
        )
    )
    assert text is None


def test_unrelated_chat_is_not_a_caption() -> None:
    text = caption_from_resend(
        _history(
            ("user", "какая погода"),
            ("assistant", "Сегодня ясно."),
        )
    )
    assert text is None


def test_result_file_id_prefers_largest_photo_then_document() -> None:
    photo = SimpleNamespace(
        photo=[
            SimpleNamespace(file_id="small"),
            SimpleNamespace(file_id="large"),
        ]
    )
    document = SimpleNamespace(photo=None, document=SimpleNamespace(file_id="doc"))
    assert result_file_id(photo) == "large"
    assert result_file_id(document) == "doc"


class _History:
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows

    async def get_history(self, *_args: Any, **_kwargs: Any) -> list[dict[str, str]]:
        return self.rows


class _Router:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.reply = (
            '{"intent":"edit","base_index":0,"instruction_en":"fix the sky",'
            '"instruction_ru":"поправь небо","confidence":0.95,"reference_indexes":[]}'
        )
        self.config = SimpleNamespace(
            photo_router_model="router",
            router_confidence=0.55,
            router_max_tokens=200,
        )

    async def chat_completion(
        self, messages: list[dict[str, Any]], model: str | None = None, **kwargs: Any
    ) -> str:
        user = messages[-1]["content"]
        text = user if isinstance(user, str) else str(user)
        self.calls.append(text)
        return self.reply


@pytest.mark.asyncio
async def test_answer_to_question_keeps_low_confidence_edit() -> None:
    from app.bot.photo_flow import _decision

    router = _Router()
    router.reply = (
        '{"intent":"generate","base_index":0,"instruction_en":"sunset sky",'
        '"instruction_ru":"закатное небо","confidence":0.1,"reference_indexes":[]}'
    )
    media = PendingMedia(
        chat_id=1,
        user_id=2,
        assistant_id="a",
        items=[PendingMediaItem(file_id="photo", message_id=3)],
        caption="сделай закатное небо",
        clarify_rounds=1,
    )
    turn = SimpleNamespace(
        assistant=router,
        topic_storage=_History([]),
    )
    decision = await _decision(
        turn, media, [{"data": b"png", "mime": "image/png"}], "t"
    )
    assert decision.intent == "generate"
    assert decision.instruction_en == "sunset sky"


@pytest.mark.asyncio
async def test_fresh_caption_still_rejects_low_confidence() -> None:
    from app.bot.photo_flow import _decision

    router = _Router()
    router.reply = (
        '{"intent":"generate","base_index":0,"instruction_en":"sunset sky",'
        '"confidence":0.1,"reference_indexes":[]}'
    )
    media = PendingMedia(
        chat_id=1,
        user_id=2,
        assistant_id="a",
        items=[PendingMediaItem(file_id="photo", message_id=3)],
        caption="сделай закатное небо",
    )
    turn = SimpleNamespace(assistant=router, topic_storage=_History([]))
    decision = await _decision(
        turn, media, [{"data": b"png", "mime": "image/png"}], "t"
    )
    assert decision.intent == "unclear"
    assert decision.question_ru == "Уточните, что сделать с фото."


@pytest.mark.asyncio
async def test_empty_caption_adopts_resend_request() -> None:
    from app.bot.photo_flow import _decision

    router = _Router()
    media = PendingMedia(
        chat_id=1,
        user_id=2,
        assistant_id="a",
        items=[PendingMediaItem(file_id="again", message_id=3)],
        caption="",
    )
    turn = SimpleNamespace(
        assistant=router,
        topic_storage=_History(
            _history(
                ("user", "поправь небо"),
                ("assistant", "Снимок недоступен, пришлите фото."),
            )
        ),
    )
    decision = await _decision(
        turn, media, [{"data": b"png", "mime": "image/png"}], "t"
    )
    assert decision.intent == "edit"
    assert media.caption == "поправь небо"
    assert router.calls


@pytest.mark.asyncio
async def test_chat_puts_pending_image_back() -> None:
    from app.bot.photo_flow import _execute
    from app.storage.pending_media import InMemoryPendingMediaStorage

    pending = InMemoryPendingMediaStorage()
    media = PendingMedia(
        chat_id=1,
        user_id=2,
        assistant_id="a",
        items=[PendingMediaItem(file_id="result-file", message_id=4)],
        caption="спасибо",
    )
    seen: list[str] = []

    async def process(*_args: Any, **kwargs: Any) -> None:
        seen.append(kwargs["user_text"])

    turn = SimpleNamespace(
        message=SimpleNamespace(chat=SimpleNamespace(id=1)),
        assistant=None,
        topic_storage=None,
        assistant_storage=None,
        state=None,
        pending=pending,
    )
    import app.bot.photo_flow as flow

    original = flow.R._process_user_message
    flow.R._process_user_message = process  # type: ignore[method-assign]
    try:
        await _execute(
            turn,
            media,
            profile=None,
            loaded=[],
            decision=PhotoDecision(intent="chat"),
            topic_id="t",
            pending_claimed=True,
        )
    finally:
        flow.R._process_user_message = original  # type: ignore[method-assign]
    stored = await pending.get(1, "a", 2)
    assert stored is not None
    assert stored.items[0].file_id == "result-file"
    assert stored.caption == ""
    assert seen == ["спасибо"]


@pytest.mark.asyncio
async def test_edit_keeps_result_file_id_not_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.bot.photo_flow as flow
    import app.infra.job_queue as job_queue
    from app.media.media_tools import UserMedia
    from app.storage.pending_media import InMemoryPendingMediaStorage

    pending = InMemoryPendingMediaStorage()
    media = PendingMedia(
        chat_id=1,
        user_id=2,
        assistant_id="a",
        items=[PendingMediaItem(file_id="source-file", message_id=1)],
        caption="поправь небо",
    )
    await pending.put(media)

    async def send(
        _message: Any, _image: str, _index: int, *, as_file: bool = False
    ) -> SimpleNamespace:
        return SimpleNamespace(
            photo=[
                SimpleNamespace(file_id="thumb"),
                SimpleNamespace(file_id="result-file"),
            ]
        )

    async def edited(*_args: Any, **_kwargs: Any) -> list[str]:
        return ["image"]

    async def noop(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def allow(*_args: object, **_kwargs: object) -> bool:
        return True

    monkeypatch.setattr(job_queue, "edit_images", edited)
    monkeypatch.setattr(flow.R, "_send_generated_image", send)
    monkeypatch.setattr(flow, "confirm_reply", lambda *_a, **_k: _ready())
    monkeypatch.setattr(flow, "_status", lambda *_a, **_k: _ready())
    monkeypatch.setattr(flow, "_drop_status", noop)
    monkeypatch.setattr(flow, "_remember_pair", noop)
    monkeypatch.setattr(flow.R, "_reset_flow", noop)
    monkeypatch.setattr(flow.rate_limiter, "consume_image_bucket", allow)
    monkeypatch.setattr(flow.cost_tracker, "check_budget", noop)
    monkeypatch.setattr(flow.cost_tracker, "release_reservation", noop)
    monkeypatch.setattr(flow.M, "_text_model_of", lambda *_a, **_k: "text")

    answers: list[str] = []

    class _Message:
        chat = SimpleNamespace(id=1)
        from_user = SimpleNamespace(id=2)

        async def answer(self, text: str, **_kwargs: Any) -> SimpleNamespace:
            answers.append(text)
            return SimpleNamespace(message_id=8)

    turn = SimpleNamespace(
        message=_Message(),
        assistant=SimpleNamespace(
            config=SimpleNamespace(default_image_edit_model="edit-model")
        ),
        topic_storage=None,
        assistant_storage=None,
        state=SimpleNamespace(),
        pending=pending,
    )
    loaded = [
        UserMedia(
            kind="image",
            filename="photo.jpg",
            mime_type="image/jpeg",
            data=b"jpeg",
        )
    ]
    await flow._edit(
        turn,
        media,
        profile=SimpleNamespace(),
        loaded=loaded,
        decision=PhotoDecision(
            intent="edit",
            base_index=0,
            instruction_en="fix the sky",
            instruction_ru="поправь небо",
            confidence=1.0,
        ),
        topic_id="t",
    )
    stored = await pending.get(1, "a", 2)
    assert stored is not None
    assert [item.file_id for item in stored.items] == ["result-file"]
    assert stored.caption == ""


@pytest.mark.asyncio
async def test_partial_edit_delivery_keeps_the_sent_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.bot.photo_flow as flow
    import app.infra.job_queue as job_queue
    from app.llm.errors import AssistAIError
    from app.media.media_tools import UserMedia
    from app.storage.pending_media import InMemoryPendingMediaStorage

    pending = InMemoryPendingMediaStorage()
    media = PendingMedia(
        chat_id=1,
        user_id=2,
        assistant_id="a",
        items=[PendingMediaItem(file_id="source-file", message_id=1)],
        caption="поправь небо",
    )
    await pending.put(media)

    async def send(
        _message: Any, _image: str, index: int, *, as_file: bool = False
    ) -> SimpleNamespace:
        if index > 0:
            raise AssistAIError("файл не дошёл")
        return SimpleNamespace(photo=[SimpleNamespace(file_id="result-file")])

    async def edited(*_args: Any, **_kwargs: Any) -> list[str]:
        return ["first", "second"]

    async def noop(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def allow(*_args: object, **_kwargs: object) -> bool:
        return True

    monkeypatch.setattr(job_queue, "edit_images", edited)
    monkeypatch.setattr(flow.R, "_send_generated_image", send)
    monkeypatch.setattr(flow, "confirm_reply", lambda *_a, **_k: _ready())
    monkeypatch.setattr(flow, "_status", lambda *_a, **_k: _ready())
    monkeypatch.setattr(flow, "_drop_status", noop)
    monkeypatch.setattr(flow, "_remember_pair", noop)
    monkeypatch.setattr(flow.R, "_reset_flow", noop)
    monkeypatch.setattr(flow.rate_limiter, "consume_image_bucket", allow)
    monkeypatch.setattr(flow.cost_tracker, "check_budget", noop)
    monkeypatch.setattr(flow.cost_tracker, "release_reservation", noop)
    monkeypatch.setattr(flow.M, "_text_model_of", lambda *_a, **_k: "text")

    answers: list[str] = []

    class _Message:
        chat = SimpleNamespace(id=1)
        from_user = SimpleNamespace(id=2)

        async def answer(self, text: str, **_kwargs: Any) -> SimpleNamespace:
            answers.append(text)
            return SimpleNamespace(message_id=8)

    turn = SimpleNamespace(
        message=_Message(),
        assistant=SimpleNamespace(
            config=SimpleNamespace(default_image_edit_model="edit-model")
        ),
        topic_storage=None,
        assistant_storage=None,
        state=SimpleNamespace(),
        pending=pending,
    )
    await flow._edit(
        turn,
        media,
        profile=SimpleNamespace(),
        loaded=[
            UserMedia(
                kind="image",
                filename="photo.jpg",
                mime_type="image/jpeg",
                data=b"jpeg",
            )
        ],
        decision=PhotoDecision(
            intent="edit",
            base_index=0,
            instruction_en="fix the sky",
            instruction_ru="поправь небо",
            confidence=1.0,
        ),
        topic_id="t",
    )
    stored = await pending.get(1, "a", 2)
    assert stored is not None
    assert [item.file_id for item in stored.items] == ["result-file"]
    assert answers == ["файл не дошёл"]


@pytest.mark.asyncio
async def test_edit_failure_puts_the_source_album_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.bot.photo_flow as flow
    import app.infra.job_queue as job_queue
    from app.llm.errors import AssistAIError
    from app.media.media_tools import UserMedia
    from app.storage.pending_media import InMemoryPendingMediaStorage

    pending = InMemoryPendingMediaStorage()
    media = PendingMedia(
        chat_id=1,
        user_id=2,
        assistant_id="a",
        items=[PendingMediaItem(file_id="source-file", message_id=1)],
        caption="поправь небо",
    )
    resets: list[object] = []

    async def edited(*_args: Any, **_kwargs: Any) -> list[str]:
        raise AssistAIError("сервис недоступен", error_code="SERVICE_UNAVAILABLE")

    async def noop(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def allow(*_args: object, **_kwargs: object) -> bool:
        return True

    async def reset(state: object) -> None:
        resets.append(state)

    monkeypatch.setattr(job_queue, "edit_images", edited)
    monkeypatch.setattr(flow, "_status", lambda *_a, **_k: _ready())
    monkeypatch.setattr(flow, "_drop_status", noop)
    monkeypatch.setattr(flow.R, "_reset_flow", reset)
    monkeypatch.setattr(flow.rate_limiter, "consume_image_bucket", allow)
    monkeypatch.setattr(flow.cost_tracker, "release_reservation", noop)

    answers: list[str] = []

    class _Message:
        chat = SimpleNamespace(id=1)
        from_user = SimpleNamespace(id=2)

        async def answer(self, text: str, **_kwargs: Any) -> SimpleNamespace:
            answers.append(text)
            return SimpleNamespace(message_id=8)

    state = SimpleNamespace()
    turn = SimpleNamespace(
        message=_Message(),
        assistant=SimpleNamespace(
            config=SimpleNamespace(default_image_edit_model="edit-model")
        ),
        topic_storage=None,
        assistant_storage=None,
        state=state,
        pending=pending,
    )
    await flow._edit(
        turn,
        media,
        profile=SimpleNamespace(),
        loaded=[
            UserMedia(
                kind="image",
                filename="photo.jpg",
                mime_type="image/jpeg",
                data=b"jpeg",
            )
        ],
        decision=PhotoDecision(
            intent="edit",
            base_index=0,
            instruction_en="fix the sky",
            instruction_ru="поправь небо",
            confidence=1.0,
        ),
        topic_id="t",
    )
    stored = await pending.get(1, "a", 2)
    assert stored is not None
    assert stored.items[0].file_id == "source-file"
    assert resets == [state]
    assert answers == ["сервис недоступен"]


@pytest.mark.asyncio
async def test_image_rate_limit_restores_album_and_resets_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.bot.photo_flow as flow
    from app.storage.pending_media import InMemoryPendingMediaStorage

    pending = InMemoryPendingMediaStorage()
    media = PendingMedia(
        chat_id=1,
        user_id=2,
        assistant_id="a",
        items=[PendingMediaItem(file_id="source-file", message_id=1)],
    )
    resets: list[object] = []

    async def blocked(*_args: object, **_kwargs: object) -> bool:
        return False

    async def retry(_user_id: int, _kind: str) -> int:
        return 15

    async def reset(state: object) -> None:
        resets.append(state)

    monkeypatch.setattr(flow.rate_limiter, "consume_image_bucket", blocked)
    monkeypatch.setattr(flow.rate_limiter, "get_retry_after", retry)
    monkeypatch.setattr(flow.R, "_reset_flow", reset)

    class _Message:
        chat = SimpleNamespace(id=1)
        from_user = SimpleNamespace(id=2)

        async def answer(self, text: str, **_kwargs: Any) -> None:
            return None

    state = SimpleNamespace()
    turn = SimpleNamespace(
        message=_Message(),
        state=state,
        pending=pending,
    )
    await flow._generate(
        turn,
        media,
        profile=SimpleNamespace(),
        loaded=[],
        decision=PhotoDecision(intent="generate", instruction_en="poster"),
        topic_id="t",
    )
    stored = await pending.get(1, "a", 2)
    assert stored is not None
    assert stored.items[0].file_id == "source-file"
    assert resets == [state]


@pytest.mark.asyncio
async def test_analyze_puts_the_album_back() -> None:
    from app.bot.photo_flow import _execute
    from app.storage.pending_media import InMemoryPendingMediaStorage

    pending = InMemoryPendingMediaStorage()
    media = PendingMedia(
        chat_id=1,
        user_id=2,
        assistant_id="a",
        items=[PendingMediaItem(file_id="source-file", message_id=3)],
        caption="что на фото",
        clarify_rounds=1,
    )
    seen: list[str] = []

    async def vision(*_args: Any, **kwargs: Any) -> None:
        seen.append(kwargs["history_placeholder"])

    async def reset(_state: object) -> None:
        return None

    turn = SimpleNamespace(
        message=SimpleNamespace(chat=SimpleNamespace(id=1)),
        assistant=None,
        topic_storage=None,
        assistant_storage=None,
        state=None,
        pending=pending,
    )
    import app.bot.photo_flow as flow

    original = flow.R._process_vision_message
    flow.R._process_vision_message = vision  # type: ignore[method-assign]
    flow.R._reset_flow = reset  # type: ignore[method-assign]
    try:
        await _execute(
            turn,
            media,
            profile=None,
            loaded=[],
            decision=PhotoDecision(intent="analyze", instruction_ru="что на фото"),
            topic_id="t",
            pending_claimed=True,
        )
    finally:
        flow.R._process_vision_message = original  # type: ignore[method-assign]
    stored = await pending.get(1, "a", 2)
    assert stored is not None
    assert stored.items[0].file_id == "source-file"
    assert stored.caption == ""
    assert stored.clarify_rounds == 0
    assert seen == ["[Альбом: 0 фото]"]


async def _ready() -> str:
    return "Готово."
