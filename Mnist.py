import numpy as np
import cv2
import joblib
import numpy as np
import seldon_core
from seldon_core.user_model import SeldonComponent
from typing import Dict, List, Union, Iterable
import os
import logging
import onnx
from urllib.parse import urlparse
from seldon_core.utils import getenv
import onnxruntime as rt

logger = logging.getLogger(__name__)

class Mnist(SeldonComponent):
  """
  Model template. You can load your model parameters in __init__ from a location accessible at runtime
  """
  def __init__(self, model_uri: str = None,  method: str = "predict", modelUri: str = None, type: str = None):
    """
    Add any initialization parameters. These will be passed at runtime from the graph definition parameters defined in your seldondeployment kubernetes resource manifest.
    """
    super().__init__()
    print(model_uri, modelUri, type)
    self.model_uri = model_uri
    self.method = method
    self.ready = False
    self.load()

  def load(self):
    print("load")
    print(self.model_uri)
    
    model_file = os.path.join(seldon_core.Storage.download(self.model_uri), "model.onnx")
    print("model file", model_file)
    self._model = model_file
    self.ready = True
    print("init and model loading done!!!")

  def predict(self, X, features_names):
    print("REQRUEST!!!", X)
    # gray = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)
    # gray = cv2.resize(gray, (28,28)).astype(np.float32)/255
    X = np.reshape(X, (1,1,28,28))
    session = rt.InferenceSession(self._model, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name


    res = session.run([output_name], {input_name: X.astype('float32')})
    print(res)
 
    return res