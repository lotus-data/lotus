import logging
import lotus.dtype_extensions
import lotus.models
import lotus.vector_store
import lotus.nl_expression
import lotus.templates
import lotus.utils
from lotus.sem_ops import (
    load_sem_index,
    sem_agg,
    sem_extract,
    sem_filter,
    sem_index,
    sem_join,
    sem_map,
    sem_partition_by,
    sem_search,
    sem_sim_join,
    sem_cluster_by,
    sem_dedup,
    sem_topk,
)
import sys
from lotus.web_search import web_search, WebSearchCorpus
from lotus.settings import settings  # type: ignore[attr-defined]

LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(process)d:%(threadName)s] %(name)s:%(funcName)s:%(lineno)d - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    stream=sys.stdout
)

# Small color-config for logging.
class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[36m",    # Cyan
        logging.INFO: "\033[32m",     # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",    # Red
        # Critical is unused... but kept here for completeness.
        logging.CRITICAL: "\033[1;31m" # Bold Red
    }
    RESET = "\033[0m"
    
    def format(self, record):
        color = self.COLORS.get(record.levelno, self.COLORS[logging.INFO])
        formatter = logging.Formatter(f"{color}{LOG_FORMAT}{self.RESET}", datefmt=DATE_FORMAT)
        return formatter.format(record)

for handler in logging.root.handlers:
    handler.setFormatter(ColorFormatter())

# In the future, consider loguru, but requires a wider refactor.
logger = logging.getLogger(__name__)

__all__ = [
    "sem_map",
    "sem_filter",
    "sem_agg",
    "sem_extract",
    "sem_join",
    "sem_partition_by",
    "sem_topk",
    "sem_index",
    "load_sem_index",
    "sem_sim_join",
    "sem_cluster_by",
    "sem_search",
    "sem_dedup",
    "settings",
    "nl_expression",
    "templates",
    "logger",
    "models",
    "vector_store",
    "utils",
    "dtype_extensions",
    "web_search",
    "WebSearchCorpus",
]
