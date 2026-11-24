"""Deploy must refresh the worker image without starting a stopped worker."""

from __future__ import annotations

from pathlib import Path


def test_deploy_pulls_arq_profile_and_ups_only_a_live_worker() -> None:
    text = Path(".github/workflows/deploy.yml").read_text(encoding="utf-8")
    assert "раскомментировать" not in text
    script = text.split("script: |", 1)[1]
    assert "docker compose --profile arq pull" in script
    assert "running|restarting" in script
    assert "docker compose --profile arq up -d" in script
    assert "docker compose up -d" in script
    profile_up = script.index("docker compose --profile arq up -d")
    plain_up = script.index("docker compose up -d")
    branch = script.index("update_worker")
    assert branch < profile_up < plain_up
