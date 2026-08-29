"""Compatibility shim for tooling that still invokes ``setup.py`` directly.

Project metadata and every dependency are defined in ``pyproject.toml``.
"""

from setuptools import setup


setup()
