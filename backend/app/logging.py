from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Standard record fields that help debugging in prod.
        base.update(
            {
                "module": record.module,
                "func": record.funcName,
                "line": record.lineno,
            }
        )

        # Include exception info if present.
        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)

        # Merge structured context passed via logger.*(..., extra={...})
        extra = getattr(record, "__dict__", {})
        for k, v in extra.items():
            if k in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
            }:
                continue
            if k.startswith("_"):
                continue
            # Avoid overwriting base keys.
            if k not in base:
                base[k] = v

        return json.dumps(base, default=str, ensure_ascii=False)


def init_logging(*, level: str = "INFO", json_logs: bool = True) -> None:
    """
    Initialize process-wide logging once at startup.

    - JSON logs by default (good for prod + log aggregation)
    - Can switch to plaintext by setting LOG_JSON=false
    """

    root = logging.getLogger()
    resolved_level = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(resolved_level)

    handler = logging.StreamHandler(stream=sys.stdout)
    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(fmt="%(asctime)s %(levelname)s %(name)s - %(message)s")
        )

    # Replace handlers to avoid duplicate logs under reload.
    root.handlers = [handler]

    # Keep noisy libraries readable.
    logging.getLogger("uvicorn").setLevel(resolved_level)
    logging.getLogger("uvicorn.error").setLevel(resolved_level)
    logging.getLogger("uvicorn.access").setLevel(resolved_level)

