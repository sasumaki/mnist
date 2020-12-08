FROM python:3.7-slim
RUN apt-get update -y && apt-get install libgl1-mesa-glx -y
RUN apt-get install libglib2.0-0 -y
COPY ./requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . /app
WORKDIR /app


CMD python train.py


