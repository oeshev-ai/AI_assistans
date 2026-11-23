"""Проверка контракта Polza Media API на практике (G0).

Ключ: POLZA_API_KEY. Артефакты: ``artifacts/polza_probe/<UTC>/``.
Фикстуры: ``tests/fixtures/polza/``.
Контракт: ``app/docs/polza-media-contract.md``.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import ssl
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_ROOT / ".env")
load_dotenv()

_MODEL = "openai/gpt-image-2.6-flare"
_DEFAULT_BASE = "https://polza.ai/api"
_PROVIDER_PIN = {"only": ["OpenAI"], "allow_fallbacks": False}


def _ssl_context() -> ssl.SSLContext:
    return ssl.create_default_context()


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def _api_key() -> str:
    key = (
        os.getenv("POLZA_API_KEY", "").strip()
        or os.getenv("POLZA_AI_API_KEY", "").strip()
    )
    if not key:
        raise SystemExit("POLZA_API_KEY не задан")
    return key


def _tiny_png_b64(
    *,
    width: int = 96,
    height: int = 64,
    color: tuple[int, int, int] = (200, 120, 80),
) -> str:
    image = Image.new("RGB", (width, height), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _redact(payload: Any) -> Any:
    if isinstance(payload, dict):
        out: dict[str, Any] = {}
        for key, value in payload.items():
            lower = str(key).lower()
            if lower in {"authorization", "api_key", "apikey", "token"}:
                out[key] = "<redacted>"
                continue
            if (
                lower in {"b64_json", "data"}
                and isinstance(value, str)
                and len(value) > 200
            ):
                out[key] = f"<base64 len={len(value)}>"
                continue
            if lower == "url" and isinstance(value, str) and "://" in value:
                out[key] = value.split("?")[0]
                continue
            out[key] = _redact(value)
        return out
    if isinstance(payload, list):
        return [_redact(item) for item in payload]
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_redact(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _first_data_item(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Polza отдаёт ``data`` как list[{url|b64_json}], OpenAPI допускает и object."""
    data = payload.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    if isinstance(data, dict):
        return data
    return None


def _png_size_from_bytes(raw: bytes) -> tuple[int, int] | None:
    if len(raw) < 24 or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[12:16] != b"IHDR":
        return None
    return int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big")


def _image_size_from_payload(
    payload: dict[str, Any], out_dir: Path, stem: str
) -> tuple[int, int] | None:
    item = _first_data_item(payload)
    if item is None:
        return None
    encoded = item.get("b64_json")
    if isinstance(encoded, str) and encoded:
        raw = base64.b64decode(encoded)
        (out_dir / f"{stem}.png").write_bytes(raw)
        size = _png_size_from_bytes(raw)
        if size is not None:
            return size
        with Image.open(io.BytesIO(raw)) as image:
            return image.size
    url = item.get("url")
    if isinstance(url, str) and url.startswith("http"):
        try:
            # CDN иногда отдаёт медленно — для размеров хватает IHDR (range).
            with httpx.Client(timeout=90.0, verify=_ssl_context()) as client:
                ranged = client.get(url, headers={"Range": "bytes=0-63"})
                if ranged.status_code in {200, 206} and ranged.content:
                    size = _png_size_from_bytes(ranged.content)
                    (out_dir / f"{stem}.hdr.bin").write_bytes(ranged.content)
                    if size is not None:
                        return size
                response = client.get(url)
            if response.status_code == 200 and response.content:
                (out_dir / f"{stem}.bin").write_bytes(response.content)
                size = _png_size_from_bytes(response.content)
                if size is not None:
                    return size
                with Image.open(io.BytesIO(response.content)) as image:
                    image.save(out_dir / f"{stem}.png")
                    return image.size
        except (OSError, httpx.HTTPError, ValueError) as exc:
            print(f"  warn: не удалось скачать/открыть url: {exc}")
    return None


def _post_media(
    client: httpx.Client,
    *,
    base_url: str,
    api_key: str,
    body: dict[str, Any],
    timeout: float,
) -> tuple[int, dict[str, Any], float]:
    started = time.perf_counter()
    response = client.post(
        f"{base_url.rstrip('/')}/v1/media",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=timeout,
    )
    seconds = time.perf_counter() - started
    try:
        parsed = response.json()
    except ValueError:
        parsed = {"raw": response.text[:2000]}
    payload = parsed if isinstance(parsed, dict) else {"raw": parsed}
    return response.status_code, payload, seconds


def _get_media(
    client: httpx.Client,
    *,
    base_url: str,
    api_key: str,
    media_id: str,
    version: str,
) -> tuple[int, dict[str, Any], float]:
    started = time.perf_counter()
    response = client.get(
        f"{base_url.rstrip('/')}/{version}/media/{media_id}",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60.0,
    )
    seconds = time.perf_counter() - started
    try:
        parsed = response.json()
    except ValueError:
        parsed = {"raw": response.text[:2000]}
    payload = parsed if isinstance(parsed, dict) else {"raw": parsed}
    return response.status_code, payload, seconds


def _wait_completed(
    client: httpx.Client,
    *,
    base_url: str,
    api_key: str,
    media_id: str,
    poll_interval: float,
    total_wait: float,
    prefer_version: str = "v1",
) -> dict[str, Any]:
    deadline = time.perf_counter() + total_wait
    started = time.perf_counter()
    last: dict[str, Any] = {}
    polls = 0
    while time.perf_counter() < deadline:
        status_code, payload, _ = _get_media(
            client,
            base_url=base_url,
            api_key=api_key,
            media_id=media_id,
            version=prefer_version,
        )
        polls += 1
        last = {
            "http_status": status_code,
            "poll": polls,
            "payload": payload,
            "elapsed_s": round(time.perf_counter() - started, 2),
        }
        status = payload.get("status") if isinstance(payload, dict) else None
        if status in {"completed", "failed", "cancelled"}:
            last["terminal"] = True
            return last
        time.sleep(poll_interval)
    last["terminal"] = False
    last["timed_out"] = True
    return last


def _ensure_completed(
    client: httpx.Client,
    *,
    base_url: str,
    api_key: str,
    code: int,
    payload: dict[str, Any],
    poll_interval: float,
    total_wait: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Вернуть (финальный payload, meta poll)."""
    has_data = _first_data_item(payload) is not None
    meta: dict[str, Any] = {
        "http_status": code,
        "initial_status": payload.get("status"),
        "has_data_on_post": has_data,
        "data_shape": type(payload.get("data")).__name__,
        "id": payload.get("id"),
        "usage_on_post": payload.get("usage"),
        "warnings_on_post": payload.get("warnings"),
        "error_on_post": payload.get("error"),
    }
    if code >= 400:
        return payload, meta
    if has_data and payload.get("status") == "completed":
        meta["became_async"] = False
        return payload, meta
    if payload.get("status") in {"pending", "processing"} and isinstance(
        payload.get("id"), str
    ):
        meta["became_async"] = True
        waited = _wait_completed(
            client,
            base_url=base_url,
            api_key=api_key,
            media_id=str(payload["id"]),
            poll_interval=poll_interval,
            total_wait=total_wait,
        )
        raw_final = waited.get("payload")
        final: dict[str, Any] = raw_final if isinstance(raw_final, dict) else {}
        meta["poll"] = {
            "polls": waited.get("poll"),
            "elapsed_s": waited.get("elapsed_s"),
            "terminal": waited.get("terminal"),
            "final_status": final.get("status"),
        }
        return final, meta
    meta["became_async"] = False
    return payload, meta


def run_probes(
    *,
    base_url: str,
    api_key: str,
    out_dir: Path,
    fixtures_dir: Path,
    poll_interval: float,
    total_wait: float,
    sync_timeout: float,
) -> dict[str, Any]:
    observations: dict[str, Any] = {
        "checks": {},
        "model": _MODEL,
        "base_url": base_url,
    }

    with httpx.Client(verify=_ssl_context(), timeout=sync_timeout) as client:
        # --- 0. image_resolution rejection (documented TZ assumption) ---
        print("=== 0. image_resolution support ===")
        body_res = {
            "model": _MODEL,
            "input": {
                "prompt": "probe image_resolution",
                "aspect_ratio": "3:2",
                "image_resolution": "1K",
                "quality": "low",
            },
            "provider": _PROVIDER_PIN,
            "async": True,
        }
        code, payload, seconds = _post_media(
            client,
            base_url=base_url,
            api_key=api_key,
            body=body_res,
            timeout=30.0,
        )
        res_rec = {
            "http_status": code,
            "seconds": round(seconds, 2),
            "error": payload.get("error"),
            "supported": code < 400,
        }
        _write_json(
            out_dir / "check00_image_resolution.json",
            {"request": body_res, "response": payload, "meta": res_rec},
        )
        observations["checks"]["image_resolution"] = res_rec
        print(json.dumps(res_rec, ensure_ascii=False))

        # --- 2. aspect_ratio 3:2 + size 1536x1024 ---
        print("=== 2. aspect_ratio 3:2 + size 1536x1024 ===")
        body_gen = {
            "model": _MODEL,
            "input": {
                "prompt": "Solid coral rectangle, minimal, no text.",
                "aspect_ratio": "3:2",
                "size": "1536x1024",
                "quality": "low",
                "output_format": "png",
                "max_images": 1,
            },
            "provider": _PROVIDER_PIN,
        }
        code, payload, seconds = _post_media(
            client,
            base_url=base_url,
            api_key=api_key,
            body=body_gen,
            timeout=sync_timeout,
        )
        payload, meta = _ensure_completed(
            client,
            base_url=base_url,
            api_key=api_key,
            code=code,
            payload=payload,
            poll_interval=poll_interval,
            total_wait=total_wait,
        )
        meta["post_seconds"] = round(seconds, 2)
        size = _image_size_from_payload(payload, out_dir, "gen_3x2")
        meta["pixels"] = list(size) if size else None
        meta["expected_near"] = [1536, 1024]
        meta["usage"] = payload.get("usage")
        meta["warnings"] = payload.get("warnings")
        meta["final_status"] = payload.get("status")
        _write_json(
            out_dir / "check02_generate.json",
            {"request": body_gen, "response": payload, "meta": meta},
        )
        if payload.get("status") == "completed":
            _write_json(fixtures_dir / "generate_completed.json", payload)
        observations["checks"]["aspect_size"] = meta
        print(json.dumps(meta, ensure_ascii=False))

        # --- 4. cost_rub ---
        usage = payload.get("usage") if isinstance(payload, dict) else None
        cost_obs = {
            "usage_present": isinstance(usage, dict),
            "cost_rub": usage.get("cost_rub") if isinstance(usage, dict) else None,
            "cost": usage.get("cost") if isinstance(usage, dict) else None,
            "in_initial_post": bool(meta.get("usage_on_post")),
            "post_seconds": meta.get("post_seconds"),
            "became_async": meta.get("became_async"),
        }
        observations["checks"]["cost_rub"] = cost_obs
        print("=== 4. usage.cost_rub ===")
        print(json.dumps(cost_obs, ensure_ascii=False))

        # --- 1. base64 reference ---
        print("=== 1. base64 reference edit ===")
        ref_b64 = _tiny_png_b64(width=192, height=128)
        body_edit = {
            "model": _MODEL,
            "input": {
                "prompt": "Slightly warmer tones, keep composition.",
                "aspect_ratio": "3:2",
                "size": "1536x1024",
                "quality": "low",
                "output_format": "png",
                "max_images": 1,
                "images": [{"type": "base64", "data": ref_b64}],
            },
            "provider": _PROVIDER_PIN,
        }
        code, payload, seconds = _post_media(
            client,
            base_url=base_url,
            api_key=api_key,
            body=body_edit,
            timeout=sync_timeout,
        )
        payload, meta = _ensure_completed(
            client,
            base_url=base_url,
            api_key=api_key,
            code=code,
            payload=payload,
            poll_interval=poll_interval,
            total_wait=total_wait,
        )
        meta["post_seconds"] = round(seconds, 2)
        failed = (
            code >= 400
            or payload.get("status") == "failed"
            or isinstance(payload.get("error"), dict)
        )
        base64_accepted = not failed and payload.get("status") == "completed"
        # If still pending somehow, treat non-400 create as accepted for transport.
        if not failed and payload.get("status") in {
            "completed",
            "pending",
            "processing",
        }:
            if payload.get("status") == "completed":
                base64_accepted = True
            elif code < 400 and meta.get("initial_status") in {
                "pending",
                "processing",
                "completed",
            }:
                # create accepted; wait already done above
                base64_accepted = payload.get("status") == "completed"
        size = _image_size_from_payload(payload, out_dir, "edit_base64")
        meta["pixels"] = list(size) if size else None
        meta["base64_accepted"] = base64_accepted
        meta["usage"] = payload.get("usage")
        meta["warnings"] = payload.get("warnings")
        meta["final_status"] = payload.get("status")
        meta["error"] = payload.get("error") or meta.get("error_on_post")
        raw_input = body_edit["input"]
        input_edit: dict[str, Any] = (
            dict(raw_input) if isinstance(raw_input, dict) else {}
        )
        input_edit["images"] = [
            {"type": "base64", "data": f"<base64 len={len(ref_b64)}>"}
        ]
        body_edit_redacted: dict[str, Any] = {
            "model": body_edit["model"],
            "input": input_edit,
            "provider": body_edit["provider"],
        }
        _write_json(
            out_dir / "check01_edit_base64.json",
            {"request": body_edit_redacted, "response": payload, "meta": meta},
        )
        if payload.get("status") == "completed":
            _write_json(fixtures_dir / "edit_completed.json", payload)
        observations["checks"]["base64_reference"] = meta
        print(json.dumps(meta, ensure_ascii=False))
        if not base64_accepted:
            observations["stop"] = "base64_not_accepted"
            _write_json(out_dir / "observations.json", observations)
            return observations

        # --- 3. quality xhigh / max (accept check; complete xhigh for realism) ---
        print("=== 3. quality xhigh / max ===")
        quality_obs: dict[str, Any] = {}
        for quality in ("xhigh", "max"):
            body_q = {
                "model": _MODEL,
                "input": {
                    "prompt": "Tiny blue square, no text.",
                    "aspect_ratio": "1:1",
                    "size": "1024x1024",
                    "quality": quality,
                    "output_format": "png",
                    "max_images": 1,
                },
                "provider": _PROVIDER_PIN,
            }
            # max: only acceptance via async flag to save spend if both work;
            # still complete at least one high-tier for fixture richness.
            if quality == "max":
                body_q_async = {**body_q, "async": True}
                code, payload, seconds = _post_media(
                    client,
                    base_url=base_url,
                    api_key=api_key,
                    body=body_q_async,
                    timeout=60.0,
                )
                err = payload.get("error")
                err_text = json.dumps(err, ensure_ascii=False) if err else ""
                rejected = code >= 400 or (
                    isinstance(err, dict) and "quality" in err_text.lower()
                )
                q_rec: dict[str, Any] = {
                    "http_status": code,
                    "seconds": round(seconds, 2),
                    "status": payload.get("status"),
                    "error": err,
                    "warnings": payload.get("warnings"),
                    "id": payload.get("id"),
                    "accepted": not rejected,
                    "completed": False,
                }
                _write_json(
                    out_dir / f"check03_quality_{quality}.json",
                    {
                        "request": body_q_async,
                        "response": payload,
                        "meta": q_rec,
                    },
                )
            else:
                code, payload, seconds = _post_media(
                    client,
                    base_url=base_url,
                    api_key=api_key,
                    body=body_q,
                    timeout=sync_timeout,
                )
                payload, meta_q = _ensure_completed(
                    client,
                    base_url=base_url,
                    api_key=api_key,
                    code=code,
                    payload=payload,
                    poll_interval=poll_interval,
                    total_wait=total_wait,
                )
                err = payload.get("error") or meta_q.get("error_on_post")
                err_text = json.dumps(err, ensure_ascii=False) if err else ""
                rejected = code >= 400 or (
                    isinstance(err, dict) and "quality" in err_text.lower()
                )
                q_rec = {
                    "http_status": code,
                    "post_seconds": round(seconds, 2),
                    "accepted": not rejected and payload.get("status") != "failed",
                    "final_status": payload.get("status"),
                    "usage": payload.get("usage"),
                    "warnings": payload.get("warnings"),
                    "id": meta_q.get("id"),
                    "became_async": meta_q.get("became_async"),
                    "poll": meta_q.get("poll"),
                }
                if payload.get("status") == "completed":
                    _image_size_from_payload(payload, out_dir, f"quality_{quality}")
                _write_json(
                    out_dir / f"check03_quality_{quality}.json",
                    {"request": body_q, "response": payload, "meta": q_rec},
                )
            quality_obs[quality] = q_rec
            print(f"  {quality}: {json.dumps(q_rec, ensure_ascii=False)}")
        observations["checks"]["quality_xhigh_max"] = quality_obs

        # --- 5. unknown complexity ---
        print("=== 5. unknown field complexity ===")
        body_cx = {
            "model": _MODEL,
            "input": {
                "prompt": "Green circle, no text.",
                "aspect_ratio": "1:1",
                "size": "1024x1024",
                "quality": "low",
                "complexity": "low",
                "output_format": "png",
                "max_images": 1,
            },
            "provider": _PROVIDER_PIN,
            "async": True,
        }
        code, payload, seconds = _post_media(
            client,
            base_url=base_url,
            api_key=api_key,
            body=body_cx,
            timeout=60.0,
        )
        cx_rec: dict[str, Any] = {
            "http_status": code,
            "seconds": round(seconds, 2),
            "status": payload.get("status"),
            "error": payload.get("error"),
            "warnings": payload.get("warnings"),
            "id": payload.get("id"),
        }
        if isinstance(payload.get("id"), str) and code < 400:
            waited = _wait_completed(
                client,
                base_url=base_url,
                api_key=api_key,
                media_id=str(payload["id"]),
                poll_interval=poll_interval,
                total_wait=min(total_wait, 180.0),
            )
            raw_final = waited.get("payload")
            final_cx: dict[str, Any] = raw_final if isinstance(raw_final, dict) else {}
            cx_rec["final_status"] = final_cx.get("status")
            cx_rec["warnings_final"] = final_cx.get("warnings")
            cx_rec["usage"] = final_cx.get("usage")
            cx_rec["error_final"] = final_cx.get("error")
            if final_cx.get("status") == "completed":
                payload = final_cx
        if code >= 400:
            _write_json(fixtures_dir / "error_bad_request.json", payload)
            behavior = "http_error"
        elif cx_rec.get("warnings") or cx_rec.get("warnings_final"):
            behavior = "warnings"
        else:
            behavior = "ignored_silent"
        cx_rec["behavior"] = behavior
        _write_json(
            out_dir / "check05_complexity.json",
            {"request": body_cx, "response": payload, "meta": cx_rec},
        )
        observations["checks"]["complexity_unknown"] = cx_rec
        print(json.dumps(cx_rec, ensure_ascii=False))

        # --- 6. polling v1 vs v2 ---
        print("=== 6. polling /v1 vs /v2 + async window ===")
        body_async = {
            "model": _MODEL,
            "input": {
                "prompt": "Yellow triangle, no text.",
                "aspect_ratio": "1:1",
                "size": "1024x1024",
                "quality": "low",
                "output_format": "png",
                "max_images": 1,
            },
            "provider": _PROVIDER_PIN,
            "async": True,
        }
        code, payload, seconds = _post_media(
            client,
            base_url=base_url,
            api_key=api_key,
            body=body_async,
            timeout=60.0,
        )
        media_id = payload.get("id")
        _write_json(fixtures_dir / "async_pending.json", payload)
        poll_obs: dict[str, Any] = {
            "create_http_status": code,
            "create_seconds": round(seconds, 2),
            "create_status": payload.get("status"),
            "id": media_id,
        }
        if isinstance(media_id, str):
            v1_code, v1_payload, v1_s = _get_media(
                client,
                base_url=base_url,
                api_key=api_key,
                media_id=media_id,
                version="v1",
            )
            v2_code, v2_payload, v2_s = _get_media(
                client,
                base_url=base_url,
                api_key=api_key,
                media_id=media_id,
                version="v2",
            )
            poll_obs["v1"] = {
                "http_status": v1_code,
                "seconds": round(v1_s, 3),
                "status": v1_payload.get("status")
                if isinstance(v1_payload, dict)
                else None,
                "error": v1_payload.get("error")
                if isinstance(v1_payload, dict)
                else None,
            }
            poll_obs["v2"] = {
                "http_status": v2_code,
                "seconds": round(v2_s, 3),
                "status": v2_payload.get("status")
                if isinstance(v2_payload, dict)
                else None,
                "error": v2_payload.get("error")
                if isinstance(v2_payload, dict)
                else None,
            }
            prefer = "v1" if v1_code == 200 else ("v2" if v2_code == 200 else "v1")
            poll_obs["working_status_path"] = f"/{prefer}/media/{{id}}"
            waited = _wait_completed(
                client,
                base_url=base_url,
                api_key=api_key,
                media_id=media_id,
                poll_interval=poll_interval,
                total_wait=total_wait,
                prefer_version=prefer,
            )
            raw_final = waited.get("payload")
            final_poll: dict[str, Any] = (
                raw_final if isinstance(raw_final, dict) else {}
            )
            poll_obs["polls"] = waited.get("poll")
            poll_obs["elapsed_s"] = waited.get("elapsed_s")
            poll_obs["final_status"] = final_poll.get("status")
            poll_obs["usage"] = final_poll.get("usage")
            if final_poll.get("status") == "completed":
                _image_size_from_payload(final_poll, out_dir, "async_completed")

        if not (fixtures_dir / "error_bad_request.json").exists():
            bad_code, bad_payload, _ = _post_media(
                client,
                base_url=base_url,
                api_key=api_key,
                body={"model": _MODEL, "input": {}},
                timeout=30.0,
            )
            _write_json(
                fixtures_dir / "error_bad_request.json",
                bad_payload
                if bad_code >= 400
                else {"forced": True, "response": bad_payload},
            )

        _write_json(
            out_dir / "check06_polling.json",
            {"request": body_async, "response": payload, "meta": poll_obs},
        )
        observations["checks"]["polling"] = poll_obs
        print(json.dumps(poll_obs, ensure_ascii=False))

    _write_json(out_dir / "observations.json", observations)
    return observations


def _write_contract(observations: dict[str, Any], path: Path) -> None:
    checks = observations.get("checks") or {}
    aspect = checks.get("aspect_size") or {}
    base64_ref = checks.get("base64_reference") or {}
    cost = checks.get("cost_rub") or {}
    quality = checks.get("quality_xhigh_max") or {}
    complexity = checks.get("complexity_unknown") or {}
    polling = checks.get("polling") or {}
    image_res = checks.get("image_resolution") or {}

    def _q(name: str) -> str:
        row = quality.get(name) or {}
        return (
            f"accepted={row.get('accepted')} http={row.get('http_status')} "
            f"final={row.get('final_status') or row.get('status')} "
            f"warnings={row.get('warnings')!r}"
        )

    lines = [
        "# Polza Media contract (G0 probe)",
        "",
        f"Model: `{observations.get('model')}`  ",
        f"Base: `{observations.get('base_url')}`  ",
        "Источник: живой прогон `scripts/probe_polza.py`.",
        "",
        "## Поле → наблюдение",
        "",
        "| поле / вопрос | наблюдение |",
        "|---|---|",
        f"| `image_resolution` | supported=`{image_res.get('supported')}`; "
        f"http=`{image_res.get('http_status')}`; "
        f"error=`{((image_res.get('error') or {}).get('message') if isinstance(image_res.get('error'), dict) else image_res.get('error'))}` |",
        "| размер для flare | использовать `aspect_ratio` + пиксельный `size` "
        "(напр. `1536x1024`); `image_resolution`/`resolution` → 400 |",
        f"| `images[].type=base64` | accepted=`{base64_ref.get('base64_accepted')}`; "
        f"final=`{base64_ref.get('final_status')}`; "
        f"pixels=`{base64_ref.get('pixels')}` |",
        f"| `aspect_ratio=3:2` + `size=1536x1024` | pixels=`{aspect.get('pixels')}`; "
        f"became_async=`{aspect.get('became_async')}`; "
        f"post_s=`{aspect.get('post_seconds')}`; poll=`{aspect.get('poll')}` |",
        f"| `quality=xhigh` | {_q('xhigh')} |",
        f"| `quality=max` | {_q('max')} |",
        f"| `usage.cost_rub` | present=`{cost.get('usage_present')}`; "
        f"cost_rub=`{cost.get('cost_rub')}`; "
        f"in_initial_post=`{cost.get('in_initial_post')}`; "
        f"became_async=`{cost.get('became_async')}` |",
        f"| неизвестное `complexity` | behavior=`{complexity.get('behavior')}`; "
        f"http=`{complexity.get('http_status')}`; "
        f"warnings=`{complexity.get('warnings') or complexity.get('warnings_final')!r}` |",
        f"| polling path | `{polling.get('working_status_path')}`; "
        f"v1_http=`{(polling.get('v1') or {}).get('http_status')}`; "
        f"v2_http=`{(polling.get('v2') or {}).get('http_status')}` |",
        f"| async create | status=`{polling.get('create_status')}`; "
        f"polls=`{polling.get('polls')}`; elapsed_s=`{polling.get('elapsed_s')}`; "
        f"final=`{polling.get('final_status')}` |",
        "",
        "## Маппинг размера (для G3)",
        "",
        "Для `openai/gpt-image-2.6-flare` на `/v1/media`:",
        "",
        "1. `fit_image_size` / исходник → ближайший `aspect_ratio` из enum Polza.",
        "2. Пиксели передавать полем `size` (`WIDTHxHEIGHT`), не `image_resolution`.",
        "3. `image_resolution` не отправлять — каталог модели отвечает 400.",
        "4. `complexity` и `background` не отправлять.",
        "",
        "## Фикстуры",
        "",
        "- `tests/fixtures/polza/generate_completed.json`",
        "- `tests/fixtures/polza/edit_completed.json`",
        "- `tests/fixtures/polza/async_pending.json`",
        "- `tests/fixtures/polza/error_bad_request.json`",
        "",
        "## Стоп-условие",
        "",
    ]
    if observations.get("stop") == "base64_not_accepted":
        lines.append(
            "**STOP:** base64-референс не принят. S3-путь не строить — нужно решение оператора."
        )
    else:
        lines.append(
            "base64 принят — можно строить адаптер на `MediaFileDto.type=base64`."
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    _configure_stdout()
    parser = argparse.ArgumentParser(description="Probe Polza /v1/media contract")
    parser.add_argument(
        "--base-url", default=os.getenv("POLZA_BASE_URL", _DEFAULT_BASE)
    )
    parser.add_argument("--poll-interval", type=float, default=4.0)
    parser.add_argument("--total-wait", type=float, default=240.0)
    parser.add_argument("--sync-timeout", type=float, default=130.0)
    args = parser.parse_args()

    api_key = _api_key()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = _ROOT / "artifacts" / "polza_probe" / stamp
    fixtures_dir = _ROOT / "tests" / "fixtures" / "polza"
    out_dir.mkdir(parents=True, exist_ok=True)
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    print(f"artifacts: {out_dir}")
    observations = run_probes(
        base_url=args.base_url,
        api_key=api_key,
        out_dir=out_dir,
        fixtures_dir=fixtures_dir,
        poll_interval=args.poll_interval,
        total_wait=args.total_wait,
        sync_timeout=args.sync_timeout,
    )
    contract_path = _ROOT / "app" / "docs" / "polza-media-contract.md"
    _write_contract(observations, contract_path)
    print(f"\ncontract: {contract_path}")
    if observations.get("stop") == "base64_not_accepted":
        raise SystemExit(
            "STOP: base64 не принят — не продолжаем G3 и не строим S3-путь."
        )
    print("G0 probe done.")


if __name__ == "__main__":
    main()
