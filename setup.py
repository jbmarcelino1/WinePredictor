from setuptools import find_packages
from setuptools import setup


with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content if 'git+' not in x]

setup(name='winemodel',
      version="1.0",
      description="Wine Predict Model",
      packages=find_packages(),
      test_suite = 'tests',
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      install_requires=requirements,
      zip_safe=False)
