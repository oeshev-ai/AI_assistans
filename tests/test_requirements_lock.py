"""CI и образ ставят один и тот же lock, без пакета openai."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, sep, version = line.partition("==")
        assert sep, line
        pins[name.lower()] = version.split(";", 1)[0].strip()
    return pins


def test_locks_pin_every_package_and_share_prod_versions() -> None:
    prod = _pins(ROOT / "requirements-prod.txt")
    full = _pins(ROOT / "requirements.txt")
    assert prod
    assert "openai" not in prod
    assert "openai" not in full
    for name in ("httpx", "python-dotenv", "tenacity"):
        assert prod[name]
        assert full[name] == prod[name]
    for name, version in prod.items():
        assert full[name] == version


def test_direct_requirements_do_not_list_openai() -> None:
    for name in ("requirements.in", "requirements-prod.in"):
        text = (ROOT / name).read_text(encoding="utf-8").casefold()
        assert "openai" not in text
