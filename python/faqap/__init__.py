from .fw import minimize
import sys

MIN_PYTHON = (3, 5)
if sys.version_info < MIN_PYTHON:
    sys.exit("faqap requires Python %s.%s or later.\n" % MIN_PYTHON)
