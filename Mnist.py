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

logger = logging.getLogger(__name__)

class Mnist(SeldonComponent):
  def __init__(self, model_uri: str = None,  method: str = "predict", modelUri: str = None, type: str = None):
    
    super().__init__()
    self.model_uri = model_uri
    self.method = method
    self.ready = False
    
    model_file = os.path.join(seldon_core.Storage.download(self.model_uri), "model.onnx")
    self._model = model_file
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
    session = rt.InferenceSession(self._model, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    res = session.run([output_name], {input_name: X.astype('float32')})

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
