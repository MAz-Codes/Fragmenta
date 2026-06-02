"""Allow ``python -m app.core.loop_quantizer …`` to invoke the CLI."""

from .cli import main

if __name__ == "__main__":
    main()
