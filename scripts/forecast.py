#!/usr/bin/env python
"""Entry point for the forecast driver (adds src/ to path, then delegates)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from perthwx.forecast import main  # noqa: E402

if __name__ == "__main__":
    main()
