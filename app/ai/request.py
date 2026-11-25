"""Request and validation contract shared by callers and AI providers."""

from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import BaseModel

from .config import AISettings, Component


@dataclass(frozen=True, kw_only=True)
class AIRequest:
    component: Component
    instructions: str
    input_json: str
    settings: AISettings
    output_schema: type[BaseModel] | None
    tool_options: dict
    diagnostics: dict = field(default_factory=dict)
    validate_output: Callable[[BaseModel], BaseModel] | None = None
    validate_correction: Callable[[BaseModel, BaseModel], None] | None = None


@dataclass(frozen=True)
class ValidationDiagnostic:
    rule_id: str
    policy_section: str
    field_path: str
    observed: object
    message: str


class AIOutputValidationError(ValueError):
    """An AI response violates an output constraint."""

    def __init__(
        self, message: str, diagnostic: ValidationDiagnostic,
        additional_diagnostics: tuple[ValidationDiagnostic, ...] = (),
    ):
        super().__init__(message)
        self.diagnostic = diagnostic
        self.additional_diagnostics = additional_diagnostics
