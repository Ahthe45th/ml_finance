## Installing mlfinance From Source

If you've not yet made a virtual environment, skip to <a href="#tutorial-on-virtual-environments">Tutorial on Virtual Environments</a>. Otherwise, run this in the command line:

```bash
git clone https://github.com/Ahthe45th/ml_finance.git
cd ml_finance
pip install -e . -r requirements.txt
```

And you're ready to start developing

<br> </br>

---
## Tutorial on Virtual Environments


### Option 1: venv
First, install virtualenv
```python
pip3 install virtualenv
```

Make a folder for virtualenv (name it related to the script you're making)
```bash
mkdir finance
```


Next, download a python executable; or, more simply, run the following in your python interactive terminal. It should give you a path to the system executable:
```python
>>> import sys
>>> sys.executable
/Library/Frameworks/Python.framework/Versions/3.9/bin/python3
```

Then, go to your terminal and run the following. Remember to copy the period at the end of this command; it is very important
```bash
cd finance
python -m virtualenv -p="/Library/Frameworks/Python.framework/Versions/3.9/bin/python3" .
```

Then, activate your virtual environment
```bash
source bin/activate
```
When you run python and install packages with pip, it will only install inside of the "finance" folder. If you want to quit the virtual environment, run
```bash
deactivate
```


<br> </br>
### Option 2: conda
Coming soon...

<br> </br>

---
# Next Steps

Coming soon...

# Overview

 - mlfinance: machine learning applied to stock prediction
 - node_modules: @Ahthe45th
 - .gitignore: tells git which files to ignore when pushing to the project
 - mlfinance.bash: installs [twint](https://github.com/twintproject/twint) requirements
 - package-lock.json: @Ahthe45th
 - package.json: @Ahthe45th
 - requirements.txt: required python dependencies for running mlfinance
 - S&P500-Info.csv: @Ahthe45th
 - setup.py: tells pip (python's package manager) how to install mlfinance
 - setup.py.orig: legacy code
 - token: @Ahthe45th