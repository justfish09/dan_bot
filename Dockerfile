# for raspberry pi3B arm64

FROM debian:stretch-slim

RUN apt-get update && \
  apt-get -y dist-upgrade \
            build-essential libssl-dev libffi-dev \
            libblas3 libc6 liblapack3 libhdf5-dev gcc wget \
            python3.5 python3-dev python3-pip cython3 \
            python3-numpy python3-sklearn python3-pandas python3-h5py && \
  apt-get clean

RUN wget https://dl.google.com/coral/python/tflite_runtime-1.14.0-cp35-cp35m-linux_aarch64.whl && \
  pip3 install tflite_runtime-1.14.0-cp35-cp35m-linux_aarch64.whl && \
  rm tflite_runtime-1.14.0-cp35-cp35m-linux_aarch64.whl

COPY requirements_pi.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY dan_bot/ dan_bot

COPY setup.py setup.py

MKDIR input_data

RUN python3 dan_bot/s3_client.py

RUN python3 setup.py install

CMD ["/bin/bash"]
