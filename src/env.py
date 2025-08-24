from logging import getLogger, StreamHandler, Formatter, WARNING
from global_config import LOGGING_LEVEL
from sys import stdout

# Sleek colorized formatter
class ColorFormatter(Formatter):
    COLORS = {
        "DEBUG": "\033[36m",    # Cyan
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",    # Red
        "CRITICAL": "\033[41m", # Red background
    }
    RESET = "\033[0m"

    def format(self, record):
        if record.levelname == "INFO":
            # Plain message, no level label, no color
            return record.getMessage()
        else:
            color = self.COLORS.get(record.levelname, self.RESET)
            return (
                f"{color}{record.levelname:<8}{self.RESET} "
                f"{record.getMessage()}"
            )

# Mute noisy libs globally
getLogger("matplotlib").setLevel(WARNING)
getLogger("PIL").setLevel(WARNING)
getLogger("asyncio").setLevel(WARNING)
# add more if needed

# Your dedicated app logger
logger = getLogger("myapp")   # give it a name
logger.setLevel(LOGGING_LEVEL)
logger.propagate = False      # don't bubble up to root

if not logger.handlers:
    handler = StreamHandler(stdout)
    handler.setLevel(LOGGING_LEVEL)
    handler.setFormatter(ColorFormatter())
    logger.addHandler(handler)
