import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="facerec",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A facial recognition software",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cy53rg/facerec",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.18.5',
        'matplotlib>=3.2.1',
        'opencv-python>=4.2.0.34',
        'scikit-learn>=0.23.1',
        'tensorflow>=2.2.0'
    ],
)
