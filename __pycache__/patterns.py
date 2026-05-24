import re
from handlers import handle_addition

patterns = [
    (
        re.compile(r"^(\d+(?:\.\d+)?)\s*\+\s*(\d+(?:\.\d+)?)$"),
        handle_addition
    ),
]
