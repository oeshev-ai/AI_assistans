"""CI и образ ставят один и тот же lock, без пакета openai."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _pins(path: Path) -> tuple[dict[str, str], dict[str, int]]:
    pins: dict[str, str] = {}
    hashes: dict[str, int] = {}
    current: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("--hash="):
            assert current is not None
            hashes[current] = hashes.get(current, 0) + 1
            continue
        body = line.removesuffix("\\").strip()
        name, sep, version = body.partition("==")
        assert sep, line
        current = name.lower()
        pins[current] = version.split(";", 1)[0].strip()
        hashes.setdefault(current, 0)
    return pins, hashes


def test_locks_pin_every_package_and_share_prod_versions() -> None:
    prod, prod_hashes = _pins(ROOT / "requirements-prod.txt")
    full, full_hashes = _pins(ROOT / "requirements.txt")
    assert prod
    assert "openai" not in prod
    assert "openai" not in full
    for name in ("httpx", "python-dotenv", "tenacity"):
        assert prod[name]
        assert full[name] == prod[name]
    for name, version in prod.items():
        assert full[name] == version
        assert prod_hashes[name] >= 1
    for name in full:
        assert full_hashes[name] >= 1
    assert "--generate-hashes" in (ROOT / "requirements-prod.txt").read_text(
        encoding="utf-8"
    )
    assert "--generate-hashes" in (ROOT / "requirements.txt").read_text(encoding="utf-8")


def test_direct_requirements_do_not_list_openai() -> None:
    for name in ("requirements.in", "requirements-prod.in"):
        text = (ROOT / name).read_text(encoding="utf-8").casefold()
        assert "openai" not in text
