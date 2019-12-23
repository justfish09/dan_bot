from setuptools import find_packages, setup

setup(name='dan_bot',
      version='0.1.0',
      description='Emojifying slack comments',
      platforms=['POSIX'],
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      data_files=[('keras_model', ['input_data/keras_model.h5']),
                  ('tflite_model', ['input_data/tf_lite_model.tflite']),
                  ('global_model', ['input_data/dan_bot.zip'])]
      )
