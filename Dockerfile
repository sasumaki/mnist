FROM python:3.7-slim
COPY . /app
WORKDIR /app
RUN apt-get update -y && apt-get install libgl1-mesa-glx -y
RUN apt-get install libglib2.0-0 -y
RUN pip install -r requirements.txt
EXPOSE 5000

ENV MODEL_NAME Mnist
ENV API_TYPE REST
ENV SERVICE_TYPE MODEL
ENV PERSISTENCE 0
ENV INCLUDE_METRICS_IN_CLIENT_RESPONSE True

CMD exec seldon-core-microservice $MODEL_NAME $API_TYPE --service-type $SERVICE_TYPE --persistence $PERSISTENCE


