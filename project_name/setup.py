from setuptools import setup, find_packages

setup(
    name="project_name",
    version="0.1.0",
    author="Developer",
    author_email="developer@example.com",
    description="A Python project with Kafka integration",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/username/project_name",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pydantic>=2.0.0",
        "confluent-kafka>=2.0.0",
        "typing-extensions>=4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.0.0",
            "mypy>=1.0.0",
        ],
    },
)
