FROM python:3.7-slim
COPY . /app
WORKDIR /app
RUN apt-get update -y && apt-get install libgl1-mesa-glx -y
RUN apt-get install libglib2.0-0 -y
RUN pip install -r requirements.txt
EXPOSE 5000

ENV MODEL_NAME Mnist
ENV SERVICE_TYPE MODEL
ENV PERSISTENCE 0

CMD exec seldon-core-microservice $MODEL_NAME --service-type $SERVICE_TYPE --persistence $PERSISTENCE
