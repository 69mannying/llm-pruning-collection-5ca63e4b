import io
from setuptools import setup, find_packages
import pathlib

with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        line.strip()
        for line in requirements_txt
        if line.strip() and not line.startswith('#') and '-e' not in line
    ]


setup(
    name='llmshearing',
    packages=["llmshearing"],
    version='0.1',
    description='LLM Shearing',
    author='Mengzhou Xia',
    url='https://github.com/princeton-nlp/LLM-Shearing',
    install_requires=install_requires,
    entry_points={
        "console_scripts": [],
    },
    package_data={},
    classifiers=["Programming Language :: Python :: 3"],
)

# auto-gptq cannot be installed