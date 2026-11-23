"""Bot entrypoint: startup, polling, ChromeBot."""

from __future__ import annotations

import asyncio
import functools
import sys

from app.bot import runtime as R
from app.bot.constants import (
    _CHROME_BUCKET,
    _KEEP_OUTGOING,
    AioHttpIntegration,
    AiohttpSession,
    Any,
    AssistAIClient,
    AssistAIConfig,
    AssistantStorage,
    BaseStorage,
    Bot,
    BotCommand,
    Dispatcher,
    LoggingIntegration,
    MemoryStorage,
    MenuButtonCommands,
    Message,
    Path,
    Redis,
    RedisConfig,
    SendMessage,
    TelegramMethod,
    TopicConfig,
    TopicHistoryStorage,
    UploadSession,
    UploadSessionStore,
    ValidationError,
    cost_tracker,
    create_assistant_storage,
    create_topic_storage,
    get_storage,
    is_upload_feature_ready,
    job_queue,
    load_dotenv,
    logger,
    logging,
    os,
    rate_limiter,
    sentry_sdk,
    setup_upload_routes,
    start_health_server,
    start_metrics_server,
)
from app.bot.handlers import _register_handlers
from app.bot.middleware import AllowlistMiddleware
from app.infra.access import is_production, parse_allowed_user_ids
from app.infra.security import redact_url_credentials
from app.storage.pending_media import PendingMediaStorage, create_pending_media_storage

_SENSITIVE_KEYS = frozenset(
    {
        "password",
        "token",
        "api_key",
        "authorization",
        "secret",
        "redis_url",
        "redis_password",
    }
)


def _redact_sentry_value(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        for key, item in value.items():
            if isinstance(key, str) and key.lower() in _SENSITIVE_KEYS:
                out[key] = "[REDACTED]"
            else:
                out[key] = _redact_sentry_value(item)
        return out
    if isinstance(value, list):
        return [_redact_sentry_value(item) for item in value]
    if isinstance(value, str):
        return redact_url_credentials(value)
    return value


def filter_sensitive_data(event: Any, hint: dict[str, Any]) -> Any:
    """Strip secrets, URL passwords, and Telegram identity from Sentry payloads."""
    _ = hint
    if not isinstance(event, dict):
        return event
    event.pop("user", None)
    contexts = event.get("contexts")
    if isinstance(contexts, dict):
        contexts.pop("chat", None)
    return _redact_sentry_value(event)


def init_sentry() -> None:
    """Initialize Sentry when ``SENTRY_DSN`` is present."""
    dsn = os.getenv("SENTRY_DSN", "").strip()
    if not dsn:
        if is_production():
            logger.error(
                "sentry_disabled",
                reason="SENTRY_DSN not set",
                hint="Set SENTRY_DSN in .env for production error tracking",
            )
        else:
            logger.warning("sentry_disabled", reason="SENTRY_DSN not set")
        return
    sentry_sdk.init(
        dsn=dsn,
        environment=os.getenv("ENVIRONMENT", "development"),
        release=os.getenv("APP_VERSION", "1.0.0"),
        integrations=[
            AioHttpIntegration(),
            LoggingIntegration(level=logging.INFO, event_level=logging.ERROR),
        ],
        traces_sample_rate=0.1,
        send_default_pii=False,
        before_send=filter_sensitive_data,  # type: ignore[arg-type]
    )
    logger.info(
        "sentry_initialized",
        environment=os.getenv("ENVIRONMENT", "development"),
    )


def warn_polza_user_hash_secret() -> None:
    """In production, plain SHA256 user hashes are dictionary-guessable."""
    if not is_production():
        return
    if (os.getenv("POLZA_USER_HASH_SECRET") or "").strip():
        return
    logger.error(
        "polza_user_hash_secret_missing",
        hint="Set POLZA_USER_HASH_SECRET in .env for HMAC aliases to Polza",
    )


async def create_fsm_storage(redis_url: str) -> BaseStorage:
    """Redis FSM storage; MemoryStorage only outside production."""
    if not redis_url:
        if is_production():
            logger.error("fsm_redis_required_in_production")
            raise SystemExit(1)
        logger.warning("fsm_memory_storage", reason="REDIS_URL not set")
        return MemoryStorage()
    try:
        from aiogram.fsm.storage.redis import RedisStorage

        probe = Redis.from_url(
            redis_url,
            socket_connect_timeout=2.0,
            socket_timeout=2.0,
        )
        try:
            await probe.ping()
        finally:
            await probe.aclose()
        storage = RedisStorage.from_url(redis_url)
        safe_url = redact_url_credentials(redis_url)
        logger.info("fsm_redis_storage", url=safe_url)
        return storage
    except SystemExit:
        raise
    except Exception as exc:
        if is_production():
            logger.error("fsm_redis_failed_fatal", error=str(exc))
            raise SystemExit(1) from exc
        logger.error("fsm_redis_failed", error=str(exc), fallback="memory")
        return MemoryStorage()


async def _open_shared_redis(redis_url: str) -> Redis | None:
    """Shared Redis for rate limits, cost tracker, and upload sessions.

    Required in production (same fail-closed policy as FSM/topics). Outside
    production an empty or unreachable URL falls back to in-process memory.
    """
    if not redis_url:
        if is_production():
            logger.error("shared_redis_required_in_production")
            raise SystemExit(1)
        return None
    try:
        client = Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=2.0,
            socket_timeout=2.0,
        )
        await client.ping()
        return client
    except Exception as exc:
        if is_production():
            logger.error("shared_redis_failed_fatal", error=str(exc))
            raise SystemExit(1) from exc
        logger.warning("shared_redis_unavailable", error=str(exc))
        return None


async def on_startup(dispatcher: Dispatcher, bot: Bot) -> None:
    """Создаёт клиент LLM, темы и профили ассистентов. Все живут в workflow_data."""
    topic_storage: TopicHistoryStorage | None = None
    assistant_storage: AssistantStorage | None = None
    pending_media: PendingMediaStorage | None = None
    redis_url = os.getenv("REDIS_URL", "").strip()
    shared_redis = await _open_shared_redis(redis_url)
    rate_limiter.attach_redis(shared_redis)
    cost_tracker.attach_redis(shared_redis)
    cost_tracker.require_llm_budgets_configured()
    from app.bot.album import get_album_hub

    get_album_hub().attach_redis(shared_redis)
    if os.getenv("PHOTO_ROUTER_FAST_PATH", "0").strip() == "1":
        logger.warning(
            "photo_router_fast_path_enabled",
            hint="Skips LLM router; forces edit at confidence 1.0 — keep 0 in production",
        )
    await job_queue.init_job_queue(redis_url)
    dispatcher.workflow_data["shared_redis"] = shared_redis
    try:
        topic_storage = await create_topic_storage(TopicConfig.from_env())
        assistant_storage = await create_assistant_storage(RedisConfig.from_env())
        assistant = AssistAIClient(AssistAIConfig.from_env())
        pending_media = await create_pending_media_storage(redis_url)
    except (ValidationError, ValueError) as exc:
        logger.exception("bot_config_invalid", error=str(exc))
        if pending_media is not None:
            await pending_media.close()
        if assistant_storage is not None:
            await assistant_storage.close()
        if topic_storage is not None:
            await topic_storage.close()
        try:
            await job_queue.close_job_queue()
        except Exception as exc:
            logger.warning("job_queue_close_failed", error=str(exc))
        if shared_redis is not None:
            await shared_redis.aclose()
        raise
    dispatcher.workflow_data["topic_storage"] = topic_storage
    dispatcher.workflow_data["assistant_storage"] = assistant_storage
    dispatcher.workflow_data["assistant"] = assistant
    dispatcher.workflow_data["pending_media"] = pending_media
    from app.bot.image_retry import bind_image_delivery, image_delivery_loop
    from app.storage.image_delivery import open_image_delivery_store

    image_store = open_image_delivery_store(shared_redis)
    bind_image_delivery(image_store)

    async def _resume_image(provider: str, job_id: str) -> bytes:
        return await assistant.recover_paid_file(provider, job_id, kind="image")

    image_task = asyncio.create_task(
        image_delivery_loop(
            bot, image_store, R._download_image_url, resume=_resume_image
        )
    )
    dispatcher.workflow_data["image_delivery_task"] = image_task
    from app.bot.video_retry import (
        bind_video_delivery,
        download_video_for_retry,
        video_delivery_loop,
    )
    from app.storage.video_delivery import open_video_delivery_store

    video_store = open_video_delivery_store(shared_redis)
    bind_video_delivery(video_store)

    async def _resume_video(provider: str, job_id: str) -> bytes:
        return await assistant.recover_paid_file(provider, job_id, kind="video")

    video_task = asyncio.create_task(
        video_delivery_loop(
            bot, video_store, download_video_for_retry, resume=_resume_video
        )
    )
    dispatcher.workflow_data["video_delivery_task"] = video_task

    from app.infra.envparse import env_int

    metrics_port = env_int("METRICS_PORT", 8083)
    health_port = env_int("HEALTH_PORT", 8082)
    start_metrics_server(port=metrics_port)

    upload_store = UploadSessionStore(shared_redis)
    s3_storage = get_storage()
    assistant.upload_session_store = upload_store  # type: ignore[attr-defined]
    dispatcher.workflow_data["bot"] = bot
    dispatcher.workflow_data["upload_store"] = upload_store
    dispatcher.workflow_data["s3_storage"] = s3_storage

    def _setup_upload_http(app: Any) -> None:
        if s3_storage is None or not os.getenv("UPLOAD_PUBLIC_BASE_URL", "").strip():
            logger.info(
                "large_upload_disabled",
                reason="YC_* or UPLOAD_PUBLIC_BASE_URL not configured",
            )
            return

        async def _on_complete(session: UploadSession) -> None:
            await R._process_s3_upload_session(
                bot=bot,
                assistant=assistant,
                topic_storage=topic_storage,
                assistant_storage=assistant_storage,
                store=upload_store,
                storage=s3_storage,
                session=session,
            )

        setup_upload_routes(
            app,
            store=upload_store,
            storage=s3_storage,
            on_complete=_on_complete,
        )
        app["upload_enabled"] = True
        logger.info(
            "large_upload_enabled",
            public_base=os.getenv("UPLOAD_PUBLIC_BASE_URL", "").strip(),
        )

    health_runner = await start_health_server(
        port=health_port,
        redis_client=shared_redis,
        proxy_probe=lambda: assistant.probe_provider("proxyapi"),
        provider_probes={
            provider_id: functools.partial(assistant.probe_provider, provider_id)
            for provider_id in assistant.provider_ids()
            if provider_id != "proxyapi"
        },
        setup_app=_setup_upload_http,
    )
    dispatcher.workflow_data["health_runner"] = health_runner
    logger.info(
        "assistant_ready",
        tools=len(assistant.get_tools_schema()),
        metrics_port=metrics_port,
        health_port=health_port,
        large_upload=is_upload_feature_ready(),
    )
    try:
        await bot.set_my_commands(
            [
                BotCommand(command="settings", description="Настройки персоны"),
                BotCommand(command="help", description="Что умеет бот"),
            ]
        )
        await bot.set_chat_menu_button(menu_button=MenuButtonCommands())
    except Exception:
        logger.exception("bot_commands_failed")


async def on_shutdown(dispatcher: Dispatcher) -> None:
    """Закрывает клиент LLM, темы и профили, если startup успел их создать."""
    image_task = dispatcher.workflow_data.get("image_delivery_task")
    if isinstance(image_task, asyncio.Task):
        image_task.cancel()
        try:
            await image_task
        except asyncio.CancelledError:
            pass
    from app.bot.image_retry import bind_image_delivery

    bind_image_delivery(None)
    video_task = dispatcher.workflow_data.get("video_delivery_task")
    if isinstance(video_task, asyncio.Task):
        video_task.cancel()
        try:
            await video_task
        except asyncio.CancelledError:
            pass
    from app.bot.video_retry import bind_video_delivery

    bind_video_delivery(None)
    health_runner = dispatcher.workflow_data.get("health_runner")
    if health_runner is not None:
        app = getattr(health_runner, "app", None)
        if app is not None:
            from app.media.upload_server import drain_upload_tasks

            await drain_upload_tasks(app)
        await health_runner.cleanup()
    await job_queue.close_job_queue()
    assistant = dispatcher.workflow_data.get("assistant")
    if isinstance(assistant, AssistAIClient):
        await assistant.close()
        logger.info("assistant_closed")
    storage = dispatcher.workflow_data.get("topic_storage")
    if isinstance(storage, TopicHistoryStorage):
        await storage.close()
    profiles = dispatcher.workflow_data.get("assistant_storage")
    if isinstance(profiles, AssistantStorage):
        await profiles.close()
    pending = dispatcher.workflow_data.get("pending_media")
    if isinstance(pending, PendingMediaStorage):
        await pending.close()
    shared_redis = dispatcher.workflow_data.get("shared_redis")
    if isinstance(shared_redis, Redis):
        await shared_redis.aclose()
    fsm_storage = dispatcher.storage
    close = getattr(fsm_storage, "close", None)
    if callable(close):
        result = close()
        if asyncio.iscoroutine(result):
            await result


class _ChromeBot(Bot):
    """Запоминает исходящий текст, чтобы убрать его из чата.

    ``message.answer`` вызывает бота напрямую, минуя ``send_message``.
    Ответ персоны и невидимое сообщение с нижним меню отправляются с
    выключенным учётом и в чате остаются.
    """

    async def __call__(
        self,
        method: TelegramMethod[Any],
        request_timeout: int | None = None,
    ) -> Any:
        result = await super().__call__(method, request_timeout=request_timeout)
        if (
            not _KEEP_OUTGOING.get()
            and isinstance(method, SendMessage)
            and isinstance(result, Message)
        ):
            bucket = _CHROME_BUCKET.get()
            if bucket is not None and result.message_id not in bucket:
                bucket.append(result.message_id)
        return result


async def main() -> None:
    """Запускает long polling. Токен бота читается из TELEGRAM_BOT_TOKEN."""
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    load_dotenv()
    init_sentry()
    warn_polza_user_hash_secret()
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        logger.error("telegram_token_missing")
        raise SystemExit(1)

    bot = _ChromeBot(token=token, session=AiohttpSession())
    storage = await create_fsm_storage(os.getenv("REDIS_URL", "").strip())
    dispatcher = Dispatcher(storage=storage)
    allowed_ids = parse_allowed_user_ids()
    if allowed_ids:
        logger.info("allowlist_enabled", users=len(allowed_ids))
    else:
        if is_production():
            logger.error(
                "allowlist_required_in_production",
                hint="Set TELEGRAM_ALLOWED_USER_IDS to limit API spend",
            )
            raise SystemExit(1)
        logger.info("allowlist_disabled", mode="allow_all")
    allowlist = AllowlistMiddleware(allowed_ids)
    dispatcher.message.middleware(allowlist)
    dispatcher.callback_query.middleware(allowlist)
    dispatcher.startup.register(on_startup)
    dispatcher.shutdown.register(on_shutdown)
    _register_handlers(dispatcher)
    await dispatcher.start_polling(bot)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Остановка по Ctrl+C")
