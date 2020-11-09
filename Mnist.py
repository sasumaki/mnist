import numpy as np
import cv2
import joblib
import numpy as np
import seldon_core
from seldon_core.user_model import SeldonComponent
from seldon_core.user_model import SeldonResponse
from typing import Dict, List, Union, Iterable
import os
import logging
import onnx
from urllib.parse import urlparse
from seldon_core.utils import getenv
import onnxruntime as rt
import time
# from minio import Minio
# from minio.error import ResponseError
logger = logging.getLogger(__name__)

class Mnist(SeldonComponent):
  def __init__(self, model_uri: str = None,  method: str = "predict", modelUri: str = None, type: str = None):
    
    super().__init__()
    self.model_uri = model_uri
    self.method = method
    self.ready = False


    # s3Client = Minio(
    #     's3.amazonaws.com',
    #     access_key=os.getenv("AWS_ACCESS_KEY_ID", ""),
    #     secret_key=os.getenv("AWS_SECRET_ACCESS_KEY", ""),
    #     secure=True,
    # )
    model_file = os.path.join(self.model_uri, "model.onnx")
    
    # try:
    # response = minio.get_object('foo', 'bar')
    # finally:
    #   response.close()
    #   response.release_conn()

    self._model = model_file
    self.session = rt.InferenceSession(self._model, None)
    self.input_name = session.get_inputs()[0].name
    self.output_name = session.get_outputs()[0].name
    self.ready = True
    print("init and model loading done!!!")

  def init_metadata(self):
    meta = {
        "versions": [self.model_uri]
    }

    return meta

  def predict(self, X, features_names):
    start_time = time.time()
    X = np.reshape(X, (1,1,28,28))
    

    res = self.session.run([self.output_name], {self.input_name: X.astype('float32')})

    runtime_metrics = {"type": "TIMER", "key": "prediction_time", "value": (time.time() - start_time) * 1000}

    return SeldonResponse(data=res, metrics=runtime_metrics)

   

  def metrics(self):
    return [
        {"type":"COUNTER","key":"request_counter","value":1}, # a counter which will increase by the given value
        {"type":"GAUGE","key":"mygauge","value":100}, # a gauge which will be set to given value
        {"type":"TIMER","key":"mytimer","value":20.2}, # a timer which will add sum and count metrics - assumed millisecs
      ]

  def tags(self,X):
    return {"model_uri": self.model_uri}
