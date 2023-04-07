from setuptools import setup
from codecs import open
from os import path


from se3dif import __version__


ext_modules = []

here = path.abspath(path.dirname(__file__))
requires_list = []
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    for line in f:
        requires_list.append(str(line))


setup(name='se3dif',
      version=__version__,
      description='SE(3)-DiffusionFields. A library to learn and sample from diffusion models on SE(3). We show how to train and use them for learning 6D grasp distributions.',
      author='Julen Urain',
      author_email='julen@robot-learning.de',
      packages=['se3dif', 'isaac_evaluation'],
      install_requires=requires_list,
      )