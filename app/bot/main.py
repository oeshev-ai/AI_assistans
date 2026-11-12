"""Bot entrypoint: startup, polling, ChromeBot."""

from __future__ import annotations

import asyncio
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


def filter_sensitive_data(event: Any, hint: dict[str, Any]) -> Any:
    """Strip secrets from Sentry payloads before upload."""
    _ = hint
    if not isinstance(event, dict):
        return event
    request = event.get("request")
    if isinstance(request, dict):
        data = request.get("data")
        if isinstance(data, dict):
            for key in ("password", "token", "api_key", "Authorization"):
                if key in data:
                    data[key] = "[REDACTED]"
    return event


def init_sentry() -> None:
    """Initialize Sentry when ``SENTRY_DSN`` is present."""
    dsn = os.getenv("SENTRY_DSN", "").strip()
    if not dsn:
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


async def create_fsm_storage(redis_url: str) -> BaseStorage:
    """Redis FSM storage with MemoryStorage fallback."""
    if not redis_url:
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
        logger.info("fsm_redis_storage", url=redis_url)
        return storage
    except Exception as exc:
        logger.error("fsm_redis_failed", error=str(exc), fallback="memory")
        return MemoryStorage()


async def _open_shared_redis(redis_url: str) -> Redis | None:
    """Optional Redis client for rate limits and health checks."""
    if not redis_url:
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
        logger.warning("shared_redis_unavailable", error=str(exc))
        return None


async def on_startup(dispatcher: Dispatcher, bot: Bot) -> None:
    """Создаёт клиент LLM, темы и профили ассистентов. Все живут в workflow_data."""
    topic_storage: TopicHistoryStorage | None = None
    assistant_storage: AssistantStorage | None = None
    redis_url = os.getenv("REDIS_URL", "").strip()
    shared_redis = await _open_shared_redis(redis_url)
    rate_limiter.attach_redis(shared_redis)
    cost_tracker.attach_redis(shared_redis)
    await job_queue.init_job_queue(redis_url)
    dispatcher.workflow_data["shared_redis"] = shared_redis
    try:
        topic_storage = await create_topic_storage(TopicConfig.from_env())
        assistant_storage = await create_assistant_storage(RedisConfig.from_env())
        assistant = AssistAIClient(AssistAIConfig.from_env())
    except ValidationError:
        logger.exception("bot_config_invalid")
        if assistant_storage is not None:
            await assistant_storage.close()
        if topic_storage is not None:
            await topic_storage.close()
        if shared_redis is not None:
            await shared_redis.aclose()
        raise
    dispatcher.workflow_data["topic_storage"] = topic_storage
    dispatcher.workflow_data["assistant_storage"] = assistant_storage
    dispatcher.workflow_data["assistant"] = assistant

    metrics_port = int(os.getenv("METRICS_PORT", "8083"))
    health_port = int(os.getenv("HEALTH_PORT", "8082"))
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
        proxy_probe=assistant.test_connection,
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
                BotCommand(command="cancel", description="Отмена мастера"),
            ]
        )
        await bot.set_chat_menu_button(menu_button=MenuButtonCommands())
    except Exception:
        logger.exception("bot_commands_failed")


async def on_shutdown(dispatcher: Dispatcher) -> None:
    """Закрывает клиент LLM, темы и профили, если startup успел их создать."""
    health_runner = dispatcher.workflow_data.get("health_runner")
    if health_runner is not None:
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
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        logger.error("telegram_token_missing")
        raise SystemExit(1)

    bot = _ChromeBot(token=token, session=AiohttpSession())
    storage = await create_fsm_storage(os.getenv("REDIS_URL", "").strip())
    dispatcher = Dispatcher(storage=storage)
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
