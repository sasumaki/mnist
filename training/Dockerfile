FROM python:3.7-slim
COPY . /app
WORKDIR /app
RUN apt-get update -y && apt-get install libgl1-mesa-glx -y
RUN apt-get install libglib2.0-0 -y
RUN pip install -r requirements.txt

CMD python train.py


