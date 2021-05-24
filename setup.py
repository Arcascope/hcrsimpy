from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements=["scipy>=1.3"]
setup(
    name="HCRSimPY",
    version="1.0.2",
    author="Kevin Hannay",
    author_email="khannay24@gmail.com",
    description="A package to simulate and analyze human circadian rhythms.",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/khannay/HCRSimPY",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    python_requires='>=3.6',
)
