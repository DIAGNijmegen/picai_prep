# Steps to set up the environment for testing

Set up conda environment:
```
conda create --name picai_prep python=3.9
```

Activate environment:
```
conda activate picai_prep
```

Install dependencies:
```
pip install -e .
pip install -r requirements_dev.txt
```

Perform tests:
```
pytest
mypy src
flake8 src
```