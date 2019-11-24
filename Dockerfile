# for raspberry pi

FROM azogue/py36_base:rpi3

RUN apt install libhdf5-dev

COPY setup.py setup.py

COPY dan_bot/ dan_bot/

RUN python setup.py install

CMD ["python", "dan_bot/app.py"]
