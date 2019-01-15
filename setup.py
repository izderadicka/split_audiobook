import sys
import os
import re
from setuptools import setup

if sys.version_info < (3,5):
    sys.exit("Python 3.5+ is required")

def read_version():
    with open(os.path.join(os.path.dirname(__file__), "split_audiobook.py")) as f:
        text = f.read(1024)
        m= re.search(r'__version__\s*=\s*"(.+)"', text)
        if m:
            return m.group(1) 
        raise Exception("Version not available in script file")

setup(
    name="split_audiobook",
    version=read_version(),
    author="Ivan",
    author_email="ivan.zderadicka@gmail.com",
    url="https://github.com/izderadicka/split_audiobook",
    scripts=["split_audiobook.py"]
)