import logging, os, pathlib
from rich.logging import RichHandler


def setup_logger(name: str, log_dir: str, level: int = logging.INFO, to_console: bool = True) -> logging.Logger:
    """Create or fetch a named logger with rotating file output.

    - Prevent duplicate handlers when called multiple times per process.
    - Do not propagate to root to avoid duplicate console logs under Ray.
    - Console output should generally be enabled only on the driver process.
    """
    path = pathlib.Path(log_dir) / f"{name}.log"
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Do not propagate to root; Ray often has its own root handlers → duplication.
    logger.propagate = False

    # Add file handler only once per logger/path
    needs_file_handler = True
    for h in logger.handlers:
        try:
            if isinstance(h, logging.handlers.RotatingFileHandler) and getattr(h, "baseFilename", None) == str(path):
                needs_file_handler = False
                break
        except Exception:
            pass
    if needs_file_handler:
        fh = logging.handlers.RotatingFileHandler(path, maxBytes=10_000_000, backupCount=5, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(fh)

    # Add console handler only once and only if requested
    if to_console:
        has_console = any(isinstance(h, RichHandler) for h in logger.handlers)
        if not has_console:
            ch = RichHandler(rich_tracebacks=True, markup=True)
            ch.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(ch)

    return logger
