import os
from setuptools import setup, find_packages

os.system('pip install pipreqs')
currentdir = os.path.dirname(os.path.realpath(__file__))
os.system(f'pipreqs {currentdir}')
<<<<<<< HEAD

=======
#os.system()
>>>>>>> 22998ebf8b4b897ea1cb3b794cb94c88fb007676
reqs = open('requirements.txt').read().split('\n')
reqs = [x for x in reqs if 'yusuf' not in x]

setup(name='mlfinance', version='1.0', packages=find_packages(), install_requires=reqs)
