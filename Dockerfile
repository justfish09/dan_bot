# for raspberry pi3B arm64

FROM debian:buster-slim

RUN apt-get update && \
  apt-get -y dist-upgrade \
            build-essential libssl-dev libffi-dev \
            libblas3 libc6 liblapack3 libhdf5-dev gcc wget \
            python3.7 python3-dev python3-pip cython3 \
            python3-numpy python3-sklearn python3-pandas python3-h5py && \
  apt-get clean

RUN wget https://dl.google.com/coral/python/tflite_runtime-1.14.0-cp37-cp37m-linux_aarch64.whl && \
  pip3 install tflite_runtime-1.14.0-cp37-cp37m-linux_aarch64.whl && \
  rm tflite_runtime-1.14.0-cp37-cp37m-linux_aarch64.whl

COPY requirements_pi.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY dan_bot/ dan_bot

COPY setup.py setup.py

RUN mkdir input_data

ARG slack_key
ARG aws_id
ARG aws_key
ARG aws_bucket

ENV DAN_BOT_KEY=$slack_key
ENV DAN_BOT_AWS_ID=$aws_id
ENV DAN_BOT_AWS_KEY=$aws_key
ENV DAN_BOT_BUCKET=$aws_bucket

RUN python3 dan_bot/s3_client.py

RUN python3 setup.py install

CMD ["python3", "dan_bot/app.py"]
