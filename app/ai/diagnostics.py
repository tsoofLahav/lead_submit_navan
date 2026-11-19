"""Privacy-preserving evaluation events and local reports."""

import hashlib
import hmac
import json
import logging
import os
import secrets
import sys
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from time import monotonic
from uuid import UUID, uuid4

logger = logging.getLogger("app.evaluation_diagnostics")
logger.addHandler(logging.StreamHandler())
logger.propagate = False

_query_key = secrets.token_bytes(32)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def query_fingerprint(value: str) -> dict:
    # A per-process key prevents offline guessing of private query terms.
    raw = value.encode()
    return {"utf8_bytes": len(raw), "process_local_hmac": hmac.digest(_query_key, raw, "sha256").hex()}


def _json(value) -> str:
    return json.dumps(value, default=str, ensure_ascii=True, separators=(",", ":"))


@dataclass
class EvaluationDiagnostics:
    correlation_id: str
    attempt_id: str
    operation: str
    retry: bool
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    start: float = field(default_factory=monotonic)
    stages: list[dict] = field(default_factory=list)
    incidents: list[dict] = field(default_factory=list)
    active_stage: str | None = None
    stage_start: float = 0.0
    configuration: dict | None = None

    def emit(self, level: int, event: str, **details) -> None:
        payload = {"timestamp": datetime.now(UTC), "event": event, "correlation_id": self.correlation_id,
                   "attempt_id": self.attempt_id, **details}
        if level >= logging.WARNING:
            self.incidents.append(payload)
        if logger.isEnabledFor(level):
            logger.log(level, "%s", _json(payload))

    def enter(self, name: str) -> None:
        self.finish_stage()
        self.active_stage = name
        self.stage_start = monotonic()
        self.emit(logging.INFO, "stage_started", stage=name)

    def finish_stage(self, error: Exception | None = None) -> None:
        if self.active_stage is None:
            return
        entry = {"stage": self.active_stage, "duration_seconds": round(monotonic() - self.stage_start, 6),
                 "status": "failed" if error else "completed"}
        if error is not None:
            entry.update(exception=type(error).__name__, reason=getattr(error, "reason", None))
        self.stages.append(entry)
        self.emit(logging.WARNING if error else logging.INFO, "stage_finished", **entry)
        self.active_stage = None

    def finish(self, result=None, error: Exception | None = None) -> None:
        self.finish_stage(error)
        executions = getattr(error if error is not None else result, "executions", ())
        requests = [request for execution in executions for request in execution.requests]
        measured_usage = [asdict(request.usage) for request in requests if request.usage is not None]
        token_totals = {
            key: sum(row[key] for row in measured_usage)
            if len(measured_usage) == len(requests) and all(row[key] is not None for row in measured_usage) else None
            for key in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens")
        }
        costs = [request.estimated_cost_usd for request in requests]
        priced = sum(cost for cost in costs if cost is not None)
        total = priced if all(cost is not None for cost in costs) else None
        status = "failed" if error else ("completed" if executions else "no_ai")
        duration = round(monotonic() - self.start, 6)
        failure = {"exception": type(error).__name__, "stage": getattr(error, "stage", None),
                   "reason": getattr(error, "reason", None)} if error else None
        report = {
            "format_version": 1, "correlation_id": self.correlation_id, "attempt_id": self.attempt_id,
            "operation": self.operation, "retry": self.retry, "started_at": self.started_at, "status": status,
            "duration_seconds": duration, "stages": self.stages, "limits": self.configuration,
            "incidents": self.incidents, "failure": failure,
            "executions": [asdict(execution) for execution in executions],
            "totals": {
                "measured_tokens": token_totals, "requests": len(requests), "requests_with_usage": len(measured_usage),
                "estimated_cost_usd": total, "known_estimated_cost_usd": priced,
                "evaluation_accounted_usd": executions[-1].evaluation_accounted_usd if executions else None,
            },
            "metrics": {
                "context_sizes": "measured_utf8_bytes", "token_usage": "api_reported",
                "retrieved_context_size": "unavailable_from_hosted_search", "usd": "estimate_from_public_rates",
                "query_fingerprints": "process_local_hmac_no_query_text",
            },
        }
        report_path = None
        directory = os.getenv("AI_DIAGNOSTICS_REPORT_DIR", ".evaluation-diagnostics")
        if directory:
            path = Path(directory)
            try:
                path.mkdir(mode=0o700, parents=True, exist_ok=True)
                target = path / f"{self.correlation_id}-{self.attempt_id}.json"
                temporary = target.with_suffix(".tmp")
                with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as handle:
                    handle.write(json.dumps(report, default=str, indent=2) + "\n")
                temporary.replace(target)
                report_path = target.name
            except OSError as failure:
                self.emit(logging.WARNING, "diagnostic_report_write_failed", exception=type(failure).__name__)
        cost_text = f"${total:.3f} estimated" if total is not None else "cost unavailable"
        reason = f" reason={getattr(error, 'reason', None) or type(error).__name__}" if error else ""
        print(
            f"Evaluation {self.correlation_id} attempt {self.attempt_id} {self.operation} {status} in {duration:.2f}s"
            f" | requests={len(requests)} input={token_totals['input_tokens']} cached={token_totals['cached_input_tokens']}"
            f" output={token_totals['output_tokens']} reasoning={token_totals['reasoning_output_tokens']}"
            f" | {cost_text}{reason} | report={report_path}",
            file=sys.stderr,
        )


current_evaluation: ContextVar[EvaluationDiagnostics | None] = ContextVar("current_evaluation", default=None)


def enter_stage(name: str) -> None:
    evaluation = current_evaluation.get()
    if evaluation is not None:
        evaluation.enter(name)


def observe_evaluation(operation: str, *, retry: bool = False):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, correlation_id: str | None = None, **kwargs):
            try:
                identifier = str(UUID(str(correlation_id))) if correlation_id is not None else str(uuid4())
            except ValueError:
                identifier = str(uuid4())
            level = os.getenv("AI_DIAGNOSTICS_LOG_LEVEL", "INFO").upper()
            logger.setLevel({"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING,
                             "ERROR": logging.ERROR}.get(level, logging.INFO))
            evaluation = EvaluationDiagnostics(identifier, str(uuid4()), operation, retry)
            token = current_evaluation.set(evaluation)
            try:
                evaluation.emit(logging.INFO, "evaluation_started", operation=operation, retry=retry)
                result = function(*args, **kwargs)
            except Exception as error:
                error.evaluation_correlation_id = identifier
                evaluation.finish(error=error)
                raise
            else:
                evaluation.finish(result=result)
                return result
            finally:
                current_evaluation.reset(token)
        return wrapped
    return decorate
