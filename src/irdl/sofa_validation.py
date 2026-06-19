"""Small shared types for SOFA validation callbacks."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SofaValidationIssue:
    """Validation issue found while checking a SOFA file."""

    code: str
    message: str
    variable: str | None = None

    def __str__(self) -> str:
        """Return a compact human-readable issue description."""
        if self.variable is None:
            return f"{self.code}: {self.message}"
        return f"{self.code}: {self.variable}: {self.message}"
