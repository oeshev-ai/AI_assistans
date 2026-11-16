"""A/B complexity для GPT Image.

Пишет чистый JSON. Уже сохранённый summary.json не переформатировать
внешними форматтерами: в приложенной копии ключи с висячими пробелами.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import httpx

_DEFAULT_PROMPT = "Плотная афиша концерта, мелкий шрифт внизу, второй ярус лайнапа."


def call(
    client: httpx.Client,
    *,
    url: str,
    api_key: str,
    model: str,
    size: str,
    quality: str,
    complexity: str,
    prompt: str,
    out_path: Path,
) -> dict[str, Any]:
    """Один запрос генерации. Сохраняет картинку и generation_id, если он есть."""
    started = time.perf_counter()
    response = client.post(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "complexity": complexity,
            "n": 1,
        },
    )
    seconds = time.perf_counter() - started
    payload: dict[str, Any] = {}
    try:
        parsed = response.json()
    except ValueError:
        parsed = None
    if isinstance(parsed, dict):
        payload = parsed
    generation_id: str | None = None
    items = payload.get("data")
    if isinstance(items, list) and items and isinstance(items[0], dict):
        raw_id = items[0].get("generation_id")
        if isinstance(raw_id, str) and raw_id:
            generation_id = raw_id
        encoded = items[0].get("b64_json")
        if isinstance(encoded, str) and encoded:
            out_path.write_bytes(base64.b64decode(encoded))
    raw_usage = payload.get("usage")
    usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
    raw_details = usage.get("output_tokens_details")
    details: dict[str, Any] = raw_details if isinstance(raw_details, dict) else {}
    image_out = details.get("image_tokens", usage.get("output_tokens", 0))
    text_out = details.get("text_tokens", 0)
    return {
        "ok": response.status_code == 200,
        "status": response.status_code,
        "seconds": seconds,
        "size": size,
        "quality": quality,
        "generation_id": generation_id,
        "usage": usage,
        "image_out": image_out if isinstance(image_out, int) else 0,
        "text_out": text_out if isinstance(text_out, int) else 0,
    }


def _load_prompt(path: str | None) -> str:
    if not path:
        return _DEFAULT_PROMPT
    return Path(path).read_text(encoding="utf-8").strip() or _DEFAULT_PROMPT


def _summary(
    rows: list[dict[str, Any]], *, model: str, size: str, quality: str, runs: int
) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for arm in ("low", "high"):
        arm_rows = [row for row in rows if row.get("arm") == arm and row.get("ok")]
        seconds = [float(row["seconds"]) for row in arm_rows]
        arms[arm] = {
            "runs_ok": len(arm_rows),
            "rejected": runs - len(arm_rows),
            "median_s": round(statistics.median(seconds), 2) if seconds else None,
            "min_s": round(min(seconds), 2) if seconds else None,
            "max_s": round(max(seconds), 2) if seconds else None,
            "image_out_tokens": [row["image_out"] for row in arm_rows],
            "text_out_tokens": [row["text_out"] for row in arm_rows],
            "sizes": sorted({str(row["size"]) for row in arm_rows}),
        }
    return {
        "model": model,
        "size": size,
        "quality": quality,
        "runs_per_arm": runs,
        "arms": arms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B complexity GPT Image")
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--model", default="openai/gpt-image-2.6-flare")
    parser.add_argument("--size", default="1024x1536")
    parser.add_argument("--quality", default="high")
    args = parser.parse_args()
    api_key = os.getenv("PROXYAPI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("PROXYAPI_API_KEY не задан")
    prompt = _load_prompt(args.prompt_file)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = Path("artifacts") / "complexity_ab" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    url = os.getenv("PROXYAPI_BASE_URL", "https://api.proxyapi.ru/v1").rstrip("/")
    url = f"{url}/images/generations"
    rows: list[dict[str, Any]] = []
    with httpx.Client(timeout=180.0) as client:
        for run in range(1, args.runs + 1):
            for arm in ("low", "high"):
                row = call(
                    client,
                    url=url,
                    api_key=api_key,
                    model=args.model,
                    size=args.size,
                    quality=args.quality,
                    complexity=arm,
                    prompt=prompt,
                    out_path=out_dir / f"{arm}_{run}.png",
                )
                row["arm"] = arm
                row["run"] = run
                rows.append(row)
                mark = "ok" if row["ok"] else "fail"
                print(
                    f"[{run}/{args.runs}] complexity={arm:<4} {mark} "
                    f"{row['seconds']:.1f}s out={row['image_out']} "
                    f"text_out={row['text_out']}"
                )
    summary = _summary(
        rows,
        model=args.model,
        size=args.size,
        quality=args.quality,
        runs=args.runs,
    )
    document = {"summary": summary, "rows": rows}
    (out_dir / "summary.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\n=== Итог ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
