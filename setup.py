"""
Setup script for the clubhouse package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="clubhouse",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Kafka integration with Confluent Schema Registry",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/clubhouse",
    packages=find_packages(include=["clubhouse", "clubhouse.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "confluent-kafka>=2.0.0",
        "fastavro>=1.5.0",
        "requests>=2.25.0",
        "python-schema-registry-client>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.0.0",
            "mypy>=1.0.0",
            "flake8>=6.0.0",
        ],
    },
    package_data={
        "clubhouse": ["schemas/*.avsc"],
    },
    entry_points={
        "console_scripts": [
            "clubhouse=clubhouse.__main__:main",
        ],
    },
)
