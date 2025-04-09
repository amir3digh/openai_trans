from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="openaitrans",
    version="0.1.0",
    author="Amirhossein",
    author_email="amirhosseinseddigh@gmail.com",
    description="A powerful AI-based text translator using ChatGPT API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/amir3digh/openaitrans",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "openai>=1.0.0",
        "pydantic>=2.0.0",
        "tiktoken>=0.3.0",
    ],
) 