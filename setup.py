import setuptools
setuptools.setup(name='mosum',
version='0.0.1',
description="Moving Sum Based Procedures for Changes in the Mean",
url='#',
author='Dom Owens',
install_requires=[
    'requests',
    'importlib-metadata; python_version < "3.8"',
],  #['opencv-python'],
author_email='',
packages=setuptools.find_packages(
    where='src',  # '.' by default
    include=['*'],  # ['*'] by default
    exclude=['tests'],
),
zip_safe=False)