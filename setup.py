import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="toolemu",
    version="0.1.0-local",  # Local version bundled with this repo
    author="Honghua Dong, Yangjun Ruan",
    author_email="honghuad@cs.toronto.edu, yjruan@cs.toronto.edu",
    description="A language model (LM)-based emulation framework for identifying the risks of LM agents with tool use",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bplaut/safety-persists-llm-agents",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8, <3.13",
)
