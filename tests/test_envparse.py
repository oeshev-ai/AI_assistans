"""Safe env parsers and HTTPS URL guard."""

from __future__ import annotations

import pytest

from app.infra.envparse import ensure_https_url, env_float, env_int


def test_env_float_invalid_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COST_CHAT_INPUT_PER_1K", "not-a-number")
    with pytest.raises(ValueError, match="COST_CHAT_INPUT_PER_1K"):
        env_float("COST_CHAT_INPUT_PER_1K", 0.00015)


def test_env_int_invalid_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UPLOAD_MAX_CONCURRENT", "x")
    with pytest.raises(ValueError, match="UPLOAD_MAX_CONCURRENT"):
        env_int("UPLOAD_MAX_CONCURRENT", 2)


def test_env_float_empty_keeps_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COST_CHAT_INPUT_PER_1K", raising=False)
    assert env_float("COST_CHAT_INPUT_PER_1K", 0.00015) == pytest.approx(0.00015)


def test_polza_wait_typo_does_not_use_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.infra.settings import get_settings

    monkeypatch.setenv("POLZA_TOTAL_WAIT", "abc")
    with pytest.raises(ValueError, match="POLZA_TOTAL_WAIT"):
        _ = get_settings().polza_total_wait


def test_user_budget_comma_decimal_does_not_become_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.infra.settings import get_settings

    monkeypatch.setenv("LLM_USER_DAILY_BUDGET_USD", "1,5")
    with pytest.raises(ValueError, match="LLM_USER_DAILY_BUDGET_USD"):
        _ = get_settings().llm_user_daily_budget_usd


def test_user_budget_zero_stays_unlimited(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.infra.settings import get_settings

    monkeypatch.setenv("LLM_USER_DAILY_BUDGET_USD", "0")
    assert get_settings().llm_user_daily_budget_usd == 0


def test_positive_zero_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.infra.settings import get_settings

    monkeypatch.setenv("POLZA_POLL_INTERVAL", "0")
    with pytest.raises(ValueError, match="больше 0"):
        _ = get_settings().polza_poll_interval


def test_require_valid_settings_exits_on_typo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.infra.settings import require_valid_settings

    monkeypatch.setenv("POLZA_TOTAL_WAIT", "abc")
    with pytest.raises(SystemExit):
        require_valid_settings()


@pytest.mark.parametrize(
    ("name", "raw", "expected"),
    [
        ("IMAGE_LEGACY_ALIASES", "off", False),
        ("IMAGE_LEGACY_ALIASES", "yes", True),
        ("PHOTO_ROUTER_FAST_PATH", "true", True),
        ("PHOTO_ROUTER_FAST_PATH", "on", True),
        ("ARQ_ENABLED", "yes", True),
        ("TRUST_PROXY", "false", False),
        ("UPLOAD_LEGACY_PRESIGN", "ON", True),
        ("POLZA_ALLOW_UPSTREAM_FALLBACKS", "0", False),
    ],
)
def test_flags_share_one_vocabulary(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    raw: str,
    expected: bool,
) -> None:
    from app.infra.settings import get_settings

    monkeypatch.setenv(name, raw)
    assert getattr(get_settings(), name.lower()) is expected


def test_legacy_alias_typo_is_one_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.infra.settings import get_settings

    monkeypatch.setenv("IMAGE_LEGACY_ALIASES", "banana")
    with pytest.raises(ValueError, match="IMAGE_LEGACY_ALIASES"):
        _ = get_settings().image_legacy_aliases


def test_fast_path_rejects_text_other_than_the_flag_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.infra.settings import get_settings

    monkeypatch.setenv("PHOTO_ROUTER_FAST_PATH", "yes please")
    with pytest.raises(ValueError, match="PHOTO_ROUTER_FAST_PATH"):
        _ = get_settings().photo_router_fast_path


def test_snapshot_ignores_later_env_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.infra.settings as settings

    monkeypatch.setenv("PHOTO_ROUTER_FAST_PATH", "1")
    settings.reset_settings_cache()
    assert settings.get_settings().photo_router_fast_path is True
    settings._snapshot = {"photo_router_fast_path": False}
    monkeypatch.setenv("PHOTO_ROUTER_FAST_PATH", "1")
    assert settings.get_settings().photo_router_fast_path is False
    settings.reset_settings_cache()
    assert settings.get_settings().photo_router_fast_path is True


def test_library_imports_do_not_load_env_or_logging() -> None:
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]

    def module_calls(relative: str) -> set[str]:
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        found: set[str] = set()
        for node in tree.body:
            if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                continue
            func = node.value.func
            if isinstance(func, ast.Name):
                found.add(func.id)
        return found

    assert "load_project_dotenv" not in module_calls("app/infra/settings.py")
    assert "configure_logging" not in module_calls("app/bot/constants.py")
    assert "load_dotenv" not in module_calls("app/llm/config.py")
    assert module_calls("app/infra/worker.py") == set()


def test_unset_legacy_aliases_stay_on(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.infra.settings import get_settings

    monkeypatch.delenv("IMAGE_LEGACY_ALIASES", raising=False)
    monkeypatch.delenv("TEXT_LEGACY_ALIASES", raising=False)
    monkeypatch.delenv("TTS_LEGACY_ALIASES", raising=False)
    settings = get_settings()
    assert settings.image_legacy_aliases is True
    assert settings.text_legacy_aliases is True
    assert settings.tts_legacy_aliases is True


def test_ensure_https_accepts_https() -> None:
    assert (
        ensure_https_url(
            "https://storage.yandexcloud.net",
            name="YC_S3_ENDPOINT",
            default="https://example.com",
        )
        == "https://storage.yandexcloud.net"
    )


def test_ensure_https_fatal_in_production(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    with pytest.raises(SystemExit):
        ensure_https_url(
            "http://insecure.example",
            name="SPEECHKIT_URL",
            default="https://tts.example/ok",
        )


def test_ensure_https_allows_http_outside_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "development")
    assert (
        ensure_https_url(
            "http://local.example",
            name="YC_S3_ENDPOINT",
            default="https://fallback.example",
        )
        == "http://local.example"
    )
