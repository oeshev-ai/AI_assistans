"""Compatibility shim for ``arq worker.WorkerSettings``."""

from app.infra.worker import WorkerSettings

__all__ = ["WorkerSettings"]
