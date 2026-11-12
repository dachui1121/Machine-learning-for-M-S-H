#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical DACHUI V7.0 story-replay entry point.

This thin wrapper intentionally delegates to dachui_publication_ready.py so there is
only one formal pipeline implementation. The legacy pre-Round4 run_pipeline.py was
removed from the canonical path to prevent accidental use of the old fixed-model flow.
"""
from __future__ import annotations

from dachui_publication_ready import main


if __name__ == "__main__":
    raise SystemExit(main())
