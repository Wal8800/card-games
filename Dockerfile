FROM tensorflow/tensorflow:2.6.1-gpu

COPY ./requirements.gamerunner.txt /requirements.txt

RUN pip install --no-cache -r /requirements.txt