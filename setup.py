import setuptools

if __name__ == '__main__':
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

    setuptools.setup(
        version='1.1',
        author_email='Joeran.Bosma@radboudumc.nl',
        long_description=long_description,
        long_description_content_type="text/markdown",
        url='https://github.com/DIAGNijmegen/picai_prep',
        project_urls={
            "Bug Tracker": "https://github.com/DIAGNijmegen/picai_prep/issues"
        },
        license='Apache License, Version 2.0',
        packages=['picai_prep', 'picai_prep.resources', 'picai_prep.examples.dcm2mha', 'picai_prep.examples.mha2nnunet'],
    )
