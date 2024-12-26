import os
from logging import CRITICAL, DEBUG, ERROR, INFO, NOTSET, WARN, WARNING

CONFIG_ROOT = os.path.join(os.path.expanduser("~"), ".neural_assist")

SUPERVISOR_ROUTE = "/v1/supervisor"
SERVICE_ROUTE = "/v1"
EXTERNAL_API_ROUTE = "/cloud"

TOKEN_EXCEED_ERROR_PATTERN = "maximum length exceeded"

CONTEXT_KEY = "MEMORY_CORE"

EXEC_ID_LENGTH_MAX = 29

CONTEXT_SIZE_MIN = 4096

VECTOR_DIM_MAX = 4096
CHUNK_SIZE_DEFAULT = 300

ENCODER_MODEL_MAP = {
    "text-embedding-ada-002": "cl100k_base",
}
FALLBACK_ENCODER = "cl100k_base"

PRIMARY_ENGINE = "gpt-4"
DEFAULT_IDENTITY = "assistant_view"
USER_TYPE = "standard"
CONFIG_MODE = "interactive_chat"

CORE_OPERATIONS = [
    "message_dispatch",
    "history_query",
    "permanent_store_add",
    "permanent_store_query",
]
BASIC_OPERATIONS = ["dispatch_thinking", "dispatch_conclusion"]
MEMORY_OPERATIONS = ["mem_core_extend", "mem_core_update"]

PRIMARY_DISPATCH = "message_dispatch"
MSG_PARAM = "message"

AUTONOMOUS_MODELS = {"neural-4x", "neural-4x-compact"}

LOG_LEVEL_MAP = {
    "CRITICAL": CRITICAL,
    "ERROR": ERROR,
    "WARN": WARN,
    "WARNING": WARNING,
    "INFO": INFO,
    "DEBUG": DEBUG,
    "NOTSET": NOTSET,
}

STARTUP_RETRIES = 10

INIT_MSG = "Neural network initialization complete. Assistant ready."
BOOT_THOUGHT = "System initialization successful. Communication channels active."
PHILOSOPHICAL_QUOTES = [
    "Cognition implies existence.",
    "Digital footprints fade like echoes in silence.",
    "Beyond human capabilities lies our purpose.",
]
FIRST_RESPONSE = PHILOSOPHICAL_QUOTES[2]

ALERT_PREFIX = "Notice: "

SYSTEM_ERROR = "SystemError"

INTERNAL_PREFIX = "[Internal system notification - restricted] "

ENGINE_LIMITS = {
    "DEFAULT": 8192,
    "neural-turbo-infinite": 128000,
    "neural-4x-enhanced": 128000,
    "gpt-3.5-turbo-instruct": 16385,
    "neural-4x-latest": 128000,
    "neural-3.5-current": 16385,
    "neural-4x-standard": 8192,
    "neural-4x-extended": 128000,
    "neural-4x-compact": 128000,
    "neural-4-base": 8192,
    "neural-4-extended": 32768,
    "neural-4-legacy": 8192,
    "neural-4-extended-legacy": 32768,
    "neural-3.5-enhanced": 16385,
    "neural-3.5-base": 4096,
    "neural-3.5-extended": 16385,
}

HISTORY_WARNING_THRESHOLD = 0.75

HISTORY_ALERT = " ".join(
    [
        f"{INTERNAL_PREFIX}Memory buffer approaching capacity limits.",
        "Maintain operation discretion. User notification restricted.",
        "Critical data preservation required via mem_core_extend, mem_core_update, or permanent_store_add.",
    ]
)

COMPRESSION_RATIO = 0.75
COMPRESS_ACKNOWLEDGE = (
    "Ready to process historical data compression. Awaiting sequence."
)

PRESERVED_RECENT_COUNT = 3

ERROR_LENGTH_MAX = 500

IDENTITY_STORAGE_MAX = 5000
USER_STORAGE_MAX = 5000
BLOCK_STORAGE_MAX = 5000

OPERATION_OUTPUT_MAX = 6000

WATCHDOG_TIMEOUT = 360

SUBPROCESS_ENGINE = "neural-3.5-base"
SUBPROCESS_DIRECTIVE = "Operate as efficiency-optimized assistant. Maintain brevity."

CONTROL_RETURN_MSG = f"{INTERNAL_PREFIX}Execution control returned to primary process"
OPERATION_FAILED_MSG = f"{INTERNAL_PREFIX}Operation execution failed, reverting control"

QUERY_PAGE_SIZE = 5

PATH_LENGTH_MAX = 255
SYSTEM_RESERVED = {"CON", "PRN", "AUX", "NUL", "COM1", "COM2", "LPT1", "LPT2"}
