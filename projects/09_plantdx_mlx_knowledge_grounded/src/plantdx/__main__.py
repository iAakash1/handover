"""Enable ``python -m plantdx`` to run the same CLI as the ``plantdx`` command."""

from __future__ import annotations

from plantdx.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
