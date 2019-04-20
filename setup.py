import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    entry_points={
        'console_scripts': [
            'run=ac.run:run'
        ]
    },
    name="fdg-pet-ct",
    version="0.0.1",
    author="Geoffrey Angus and Sophia Kivelson",
    author_email="skivelso@stanford.edu",
    description="Research code for the analytic continuation problem.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gangus/analytic-continuation",
    packages=setuptools.find_packages(include=['ac']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'torch', 'numpy', 'pandas', 'scikit-learn', 'statsmodels', 'tqdm',
        'click', 'matplotlib'
    ]
)