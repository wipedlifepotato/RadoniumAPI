"""
Internationalization (i18n) for the Rdkit SMILES API.

Supported languages: en (default), ru, uk, es.

Translations live outside the Python code, in one INI file per language::

    locales/en.ini
    locales/ru.ini
    locales/uk.ini
    locales/es.ini

Every file contains a single ``[messages]`` section::

    [messages]
    invalid_smiles = Invalid SMILES string
    ...

A language file is read lazily and cached, so only the requested language is
loaded into memory. If a key is missing in the selected language, the value
from ``en.ini`` is used instead.

The base directory can be overridden with the ``RADONIUM_LOCALES_DIR``
environment variable (useful for tests and alternative translations).
"""

from __future__ import annotations

import configparser
import os
from pathlib import Path
from typing import Dict, Optional

from fastapi import Query

# Supported languages
SUPPORTED_LANGUAGES = {"ru", "uk", "en", "es"}
DEFAULT_LANGUAGE = "en"

_BASE_DIR = Path(__file__).resolve().parent


def _locales_dir() -> Path:
    """Return the directory containing the translation INI files."""
    override = os.environ.get("RADONIUM_LOCALES_DIR")
    if override:
        return Path(override).expanduser()
    return _BASE_DIR / "locales"


# Cache of loaded languages: {lang: {key: value}}
_CACHE: Dict[str, Dict[str, str]] = {}


def _read_language_file(lang: str) -> Dict[str, str]:
    """Read a single language INI file. Returns an empty dict on error."""
    path = _locales_dir() / f"{lang}.ini"
    if not path.is_file():
        return {}

    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    try:
        with path.open("r", encoding="utf-8") as fh:
            parser.read_file(fh)
    except (OSError, configparser.Error):
        return {}

    if not parser.has_section("messages"):
        return {}
    return {
        key: value.strip()
        for key, value in parser.items("messages")
        if value is not None
    }


def get_translations(lang: str) -> Dict[str, str]:
    """Return the translations for one language (lazy loading with caching)."""
    if lang not in SUPPORTED_LANGUAGES:
        lang = DEFAULT_LANGUAGE
    if lang not in _CACHE:
        _CACHE[lang] = _read_language_file(lang)
    return _CACHE[lang]


def reload_translations(lang: Optional[str] = None) -> None:
    """
    Reset the translation cache.

    ``lang=None`` reloads every language on the next access; otherwise only
    the given language is dropped from the cache.
    """
    if lang is None:
        _CACHE.clear()
    else:
        _CACHE.pop(lang, None)


def get_text(key: str, lang: str = DEFAULT_LANGUAGE, **kwargs) -> str:
    """
    Look up a translation by key and language code.

    If the translation is missing, the key itself is returned. When keyword
    arguments are supplied, the string is formatted with ``str.format``.
    """
    if lang not in SUPPORTED_LANGUAGES:
        lang = DEFAULT_LANGUAGE

    message = get_translations(lang).get(key)
    if message is None and lang != DEFAULT_LANGUAGE:
        message = get_translations(DEFAULT_LANGUAGE).get(key)
    if message is None:
        message = key

    if kwargs:
        try:
            return message.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            return message
    return message


def get_lang(
    lang: str = Query(DEFAULT_LANGUAGE, description="Response language: en, ru, uk, es")
) -> str:
    """
    FastAPI dependency that reads the language from the query parameter.
    """
    if lang not in SUPPORTED_LANGUAGES:
        return DEFAULT_LANGUAGE
    return lang
