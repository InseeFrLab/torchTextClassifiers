"""Generic factories for different classifier types."""

from typing import Dict, Any, Optional, Type, Callable
from .classifiers.base import BaseClassifierConfig

# Registry of config factories for different classifier types
CONFIG_FACTORIES: Dict[str, Callable[[dict], BaseClassifierConfig]] = {}


def register_config_factory(classifier_type: str, factory_func: Callable[[dict], BaseClassifierConfig]):
    """Register a config factory for a classifier type."""
    CONFIG_FACTORIES[classifier_type] = factory_func


def create_config_from_dict(classifier_type: str, config_dict: dict) -> BaseClassifierConfig:
    """Create a config object from dictionary based on classifier type."""
    if classifier_type not in CONFIG_FACTORIES:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")
    
    return CONFIG_FACTORIES[classifier_type](config_dict)


# Register FastText factory
def _register_fasttext_factory():
    """Register FastText config factory."""
    try:
        from .classifiers.fasttext.core import FastTextFactory
        register_config_factory("fasttext", FastTextFactory.from_dict)
    except ImportError:
        pass  # FastText module not available


# Auto-register available factories
_register_fasttext_factory()