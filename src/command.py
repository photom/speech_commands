SILENCE_KEYWORD_IDX = 0
UNKNOWN_KEYWORD_IDX = 1

KNOWN_KEYWORDS = {
#    'silence': 0,
#    'unknown': 1,  # others
    'on': 2,
    'off': 3,
    'one': 4,
    'two': 5,
    'three': 6,
    'marvin': 7,
    'sheila': 8,
#    'stop': 4,
#    'yes': 6,
#    'no': 7,
#    'up': 2,
#    'down': 3,
#    'go': 8,
#    'right': 8,
#    'left': 10,
#    'tree': 12,
#    'zero': 13,
#    'four': 17,
#    'five': 18,
#    'six': 19,
#    'seven': 20,
#    'eight': 21,
#    'nine': 22,
#    'bed': 23,
#    'bird': 24,
#    'cat': 25,
#    'dog': 26,
#    'happy': 27,
#    'house': 28,
#    'marvin': 29,
#    'wow': 31,
}

COMMANDS = [
    ('marvin', ),
    ('sheila', ),
    ('one', 'on'),
    ('one', 'off'),
    ('two', 'on'),
    ('two', 'off'),
    ('three', 'on'),
    ('three', 'on'),
]

# KEYWORDS (exclude silence)
NUM_CLASSES = len(KNOWN_KEYWORDS) + 2


def get_keyword_index(keyword):
    if keyword in KNOWN_KEYWORDS:
        # target word
        # decrement index offset
        return KNOWN_KEYWORDS[keyword]
    else:
        # unknown
        return UNKNOWN_KEYWORD_IDX
