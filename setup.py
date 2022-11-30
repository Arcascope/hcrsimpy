from setuptools import setup, find_packages, Extension

with open("README.md", "r") as readme_file:
    readme = readme_file.read()
    
fastclock_module = Extension('fastclock',
                    sources = ['./HCRSimPy/fastclock/rk4.c', 
                               './HCRSimPy/fastclock/fastclock.c', 
                               './HCRSimPy/fastclock/spmodel.c', 
                               './HCRSimPy/fastclock/utils.c'],
                    extra_compile_args = [''],
                    language = "c")

requirements = ["scipy>=1.3"]
setup(
    name="HCRSimPY",
    version="1.0.7",
    author="Kevin Hannay",
    author_email="khannay24@gmail.com",
    description="A package to simulate and analyze human circadian rhythms.",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/khannay/HCRSimPY",
    packages=find_packages(),
    install_requires=requirements,
    headers=['./HCRSimPy/fastclock/rk4.h', 
                               './HCRSimPy/fastclock/spmodel.h', 
                               './HCRSimPy/fastclock/utils.h'],
    ext_modules= [fastclock_module],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    python_requires='>=3.8',
)
