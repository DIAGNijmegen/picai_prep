import setuptools

if __name__ == '__main__':
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

    setuptools.setup(
        version='1.0',
        author_email='Joeran.Bosma@radboudumc.nl',
        long_description=long_description,
        long_description_content_type="text/markdown",
        url='https://github.com/DIAGNijmegen/picai_prep',
        project_urls={
            "Bug Tracker": "https://github.com/DIAGNijmegen/picai_prep/issues"
        },
        license='MIT',
        packages=['picai_prep'],
        package_data={'picai_prep': ['resources/*.json']},
    )
