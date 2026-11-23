"""Write (or check) the OpenAPI schema the front-end types are generated from.

    python -m app.openapi web/src/api/openapi.json          # write
    python -m app.openapi web/src/api/openapi.json --check  # fail if out of date (CI)

Then, in web/: npm run gen:api
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .main import create_app


def render() -> str:
    return json.dumps(create_app().openapi(), indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.openapi")
    parser.add_argument("path", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    text = render()
    if args.check:
        current = args.path.read_text(encoding="utf-8") if args.path.exists() else ""
        if current != text:
            print(f"{args.path} is out of date: run `python -m app.openapi {args.path}` then `npm run gen:api`")
            return 1
        print(f"{args.path} is up to date")
        return 0
    args.path.write_text(text, encoding="utf-8", newline="\n")
    print(f"-> {args.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
