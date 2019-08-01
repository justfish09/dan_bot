from setuptools import find_packages, setup

install_requires = [
    'numpy==1.16.1',
    'sklearn',
    'pandas',
    'keras==2.2.4',
    'tensorflow',
    'slackclient==1.3.0',
    'vaderSentiment',
    'nltk',
    'websocket-client==0.54.0',
    'boto3',
]

setup(name='dan_bot',
      version='0.1.0',
      description='Emojifying slack comments',
      platforms=['POSIX'],
      packages=find_packages(),
      include_package_data=True,
      install_requires=install_requires,
      zip_safe=False)