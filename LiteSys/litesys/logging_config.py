import logging
from .utils import TextColors

class ColorFormatter(logging.Formatter):
    COLOR_MAP = {
        logging.DEBUG: "blue",
        logging.INFO: "green",
        logging.WARNING: "yellow",
        logging.ERROR: "red",
        logging.CRITICAL: "magenta",
    }

    def format(self, record):
        level_color = self.COLOR_MAP.get(record.levelno, "white")
        record.msg = TextColors.colorize(record.msg, level_color)
        return super().format(record)

def setup_logger(name="infini_ai_litesys", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        formatter = ColorFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    return logger
