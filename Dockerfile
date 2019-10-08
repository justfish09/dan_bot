FROM tiangolo/python-machine-learning:python3.6

COPY setup.py setup.py

COPY dan_bot/ dan_bot/

RUN python setup.py install

CMD ["python", "dan_bot/app.py"]
