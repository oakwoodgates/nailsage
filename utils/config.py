"""Configuration loading utilities."""

from pathlib import Path
from typing import Type, TypeVar

from config.backtest import BacktestConfig
from config.risk import RiskConfig
from config.strategy import StrategyConfig

T = TypeVar("T")


class ConfigLoader:
    """
    Utility class for loading configurations.

    Provides convenience methods for loading different config types
    and handling common configuration patterns.
    """

    @staticmethod
    def load_strategy(path: Path | str) -> StrategyConfig:
        """
        Load strategy configuration from YAML file.

        Args:
            path: Path to strategy config YAML

        Returns:
            Validated StrategyConfig instance
        """
        return StrategyConfig.from_yaml(path)

    @staticmethod
    def load_backtest(path: Path | str) -> BacktestConfig:
        """
        Load backtest configuration from YAML file.

        Args:
            path: Path to backtest config YAML

        Returns:
            Validated BacktestConfig instance
        """
        return BacktestConfig.from_yaml(path)

    @staticmethod
    def load_risk(path: Path | str) -> RiskConfig:
        """
        Load risk configuration from YAML file.

        Args:
            path: Path to risk config YAML

        Returns:
            Validated RiskConfig instance
        """
        return RiskConfig.from_yaml(path)

    @staticmethod
    def get_default_config_dir() -> Path:
        """
        Get default configuration directory.

        Returns:
            Path to configs directory
        """
        # Get package root (nailsage directory)
        package_root = Path(__file__).parent.parent.parent
        return package_root / "configs"

    @staticmethod
    def load_default_backtest() -> BacktestConfig:
        """
        Load default backtest configuration.

        Returns:
            Default BacktestConfig
        """
        config_dir = ConfigLoader.get_default_config_dir()
        default_path = config_dir / "backtest_default.yaml"

        if default_path.exists():
            return BacktestConfig.from_yaml(default_path)
        else:
            # Return default instance
            return BacktestConfig()

    @staticmethod
    def load_default_risk() -> RiskConfig:
        """
        Load default risk configuration.

        Returns:
            Default RiskConfig
        """
        config_dir = ConfigLoader.get_default_config_dir()
        default_path = config_dir / "risk_default.yaml"

        if default_path.exists():
            return RiskConfig.from_yaml(default_path)
        else:
            # Return default instance
            return RiskConfig()


def load_config(path: Path | str, config_class: Type[T]) -> T:
    """
    Generic configuration loader.

    Args:
        path: Path to config YAML file
        config_class: Configuration class to instantiate

    Returns:
        Loaded and validated configuration instance

    Example:
        >>> from config.strategy import StrategyConfig
        >>> config = load_config("configs/my_strategy.yaml", StrategyConfig)
    """
    return config_class.from_yaml(path)
