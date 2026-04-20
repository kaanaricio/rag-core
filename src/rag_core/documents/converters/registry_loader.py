from __future__ import annotations

import importlib
import logging

from .base import BaseConverter
from .registry_specs import CONVERTER_CLASS_MODULES, CONVERTER_SPECS, ConverterSpec

logger = logging.getLogger(__name__)

_converters: dict[str, BaseConverter] | None = None


def _build_converter(spec: ConverterSpec) -> BaseConverter:
    try:
        module = importlib.import_module(spec.module_name, package=__package__)
        converter_cls = getattr(module, spec.class_name)
        return converter_cls()
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize {spec.key} converter: {exc}") from exc


def get_registered_converters() -> dict[str, BaseConverter]:
    global _converters
    if _converters is not None:
        return _converters

    converters: dict[str, BaseConverter] = {}
    for spec in CONVERTER_SPECS:
        try:
            converters[spec.key] = _build_converter(spec)
        except Exception as exc:
            if spec.required:
                raise
            logger.warning("Skipping unavailable %s converter: %s", spec.key, exc)

    _converters = converters
    return converters


def load_converter_class(name: str):
    module_name = CONVERTER_CLASS_MODULES[name]
    module = importlib.import_module(module_name, package=__package__)
    return getattr(module, name)
