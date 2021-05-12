# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

with open("README.md", "r") as f:
    readme = f.read()

requirements = [
    "pythainlp>=2.0",
    "bpemb",
    "gensim>=4.0.0",
    "nltk"
    #"transformers",
    #"thai2transformers"
]

extras = {
    "fasttext":[
        "gensim>=4.0.0",
        "PyICU"
    ],
    "wangchanberta":[
        "thai2transformers",
        "torch==1.4.0"
    ],
    "full":[
        "thai2transformers",
        "torch==1.4.0",
        "gensim>=4.0.0",
        "PyICU"
    ]
}


setup(
    name="thaitextaug",
    version="0.0.1.dev6",
    description="Thai Text Augmentation",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Wannaphong Phatthiyaphaibun",
    author_email="wannaphong@yahoo.com",
    url="https://github.com/wannaphong/thaitextaug",
    packages=find_packages(exclude=["tests", "tests.*"]),
    test_suite="tests",
    python_requires=">=3.6",
    #package_data={
    #    "thaitextaug": [
    #        "data/*",
    #    ],
    #},
    include_package_data=True,
    install_requires=requirements,
    extras_require=extras,
    license="Apache Software License 2.0",
    zip_safe=False,
    keywords=[
        "thaitextaug",
        "NLP",
        "natural language processing",
        "text analytics",
        "text processing",
        "localization",
        "computational linguistics",
        "ThaiNLP",
        "Thai NLP",
        "Thai language",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: Thai",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing",
        "Topic :: Text Processing :: General",
        "Topic :: Text Processing :: Linguistic",
    ],
    project_urls={
        "Documentation": "https://github.com/wannaphong/thaitextaug",
        "Tutorials": "https://github.com/wannaphong/thaitextaug/tree/main/notebooks",
        "Source Code": "https://github.com/wannaphong/thaitextaug",
        "Bug Tracker": "https://github.com/wannaphong/thaitextaug/issues",
    },
)