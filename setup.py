import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyDojoML",
    version="0.0.1",
    author="Victor Velev",
    author_email="velev.victor@yahoo.com",
    description="A General Purpose Machine Learning Library for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VIVelev/PyDojo",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
