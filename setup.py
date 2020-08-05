import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "Semantic-Segmentation-with-ENet",
    version = "1.0",
    author = "Ashish Ramayee Asokan",
    author_email = "raashish020@gmail.com",
    description = "Semantic Segmentation Model using E-Net",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/AshishAsokan/Semantic-Segmentation-with-ENet",
    packages = setuptools.find_packages(),
    classifiers = [
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires = '>=3.6',
)