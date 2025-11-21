"""Base configuration class."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field


class BaseConfig(BaseModel):
    """
    Base configuration class with common functionality.

    All config classes should inherit from this to get:
    - Validation
    - Type checking
    - YAML loading/saving
    - Immutability (frozen)
    """

    model_config = ConfigDict(
        frozen=False,  # Allow mutation for config updates
        extra="forbid",  # Disallow extra fields to catch typos
        validate_assignment=True,  # Validate on assignment
        arbitrary_types_allowed=True,  # Allow Path and other types
    )

    @classmethod
    def from_yaml(cls, path: Path | str) -> "BaseConfig":
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            Validated configuration instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def to_yaml(self, path: Path | str) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Path to save YAML configuration
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()

    def update(self, **kwargs) -> "BaseConfig":
        """
        Create new configuration with updated values.

        Args:
            **kwargs: Fields to update

        Returns:
            New configuration instance with updates
        """
        data = self.model_dump()
        data.update(kwargs)
        return self.__class__(**data)
