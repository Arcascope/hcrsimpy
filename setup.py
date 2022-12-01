from setuptools import setup, find_packages, Extension

with open("README.md", "r") as readme_file:
    readme = readme_file.read()
    
fastclock_module = Extension('fastclock',
                    sources = ['./hcrsimpy/fastclock/rk4.c', 
                               './hcrsimpy/fastclock/fastclock.c', 
                               './hcrsimpy/fastclock/spmodel.c', 
                               './hcrsimpy/fastclock/utils.c'],
                    extra_compile_args = [''],
                    language = "c")

requirements = ["scipy>=1.3"]
setup(
    name="hcrsimpy",
    version="1.0.7",
    author="Kevin Hannay",
    author_email="khannay24@gmail.com",
    description="A package to simulate and analyze human circadian rhythms.",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/khannay/hcrsimpy",
    packages=find_packages(),
    install_requires=requirements,
    headers=['./hcrsimpy/fastclock/rk4.h', 
                               './hcrsimpy/fastclock/spmodel.h', 
                               './hcrsimpy/fastclock/utils.h'],
    ext_modules= [fastclock_module],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    python_requires='>=3.8',
)
