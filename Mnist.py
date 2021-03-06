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
from minio import Minio
import tempfile
import yaml

logger = logging.getLogger(__name__)

class Mnist(SeldonComponent):
  def __init__(self, model_uri: str = None,  method: str = "predict", modelUri: str = None, type: str = None): 
    super().__init__()
    self.model_uri = model_uri
    self.method = method
    self.ready = False
    self.out_dir = tempfile.mkdtemp()
    self.local_folder = self._download_model(self.model_uri, self.out_dir)
    self._model_file =  os.path.join(self.local_folder, "model.onnx")

    self.session = rt.InferenceSession(self._model_file)
    self.input_name = self.session.get_inputs()[0].name
    self.output_name = self.session.get_outputs()[0].name
    self.ready = True
    print("init and model loading done!!!")
 
  def init_metadata(self):
    print(self.model_uri, self.out_dir)
    print(os.listdir(self.out_dir))
    print(os.listdir(self.local_folder))

    file_path = os.path.join(self.local_folder, "metadata.yaml")
    try:
      with open(file_path, "r") as f:
        self.mdata = yaml.safe_load(f.read())
        return self.mdata
   
    except FileNotFoundError:
      print(f"metadata file {file_path} does not exist")
      logger.debug(f"metadata file {file_path} does not exist")
      return {}
    
    except yaml.YAMLError:
      print( f"metadata file {file_path} present but does not contain valid yaml")
      logger.error(
        f"metadata file {file_path} present but does not contain valid yaml"
      )
      return {}
    
    return {}

  def predict(self, X, features_names):
    start_time = time.time()
    res = self.session.run([self.output_name], {self.input_name: X.astype('float32')})

    runtime_metrics = [{"type": "TIMER", "key": "prediction_time", "value": ((time.time() - start_time) * 1000)}]

    return SeldonResponse(data=res, metrics=runtime_metrics)

  def tags(self):
    try:
      tag = {
        "model_uri": self.model_uri,
        "metadata": self.mdata
        }
      return tag
    except:
      return {
        "model_uri":  self.model_uri,
        "metadata": "no metadata"
        }
    
  def _create_minio_client(self):
        # Adding prefixing "http" in urlparse is necessary for it to be the netloc
        url = urlparse(os.getenv("AWS_ENDPOINT_URL", "http://s3.amazonaws.com"))
        use_ssl = (
            url.scheme == "https"
            if url.scheme
            else bool(getenv("USE_SSL", "S3_USE_HTTPS", "false"))
        )
        return Minio(
            url.netloc,
            access_key=os.getenv("AWS_ACCESS_KEY_ID", ""),
            secret_key=os.getenv("AWS_SECRET_ACCESS_KEY", ""),
            region=os.getenv("AWS_REGION", ""),
            secure=use_ssl,
        )
  def _download_model(self, uri, temp_dir: str = None):
    client = self._create_minio_client()
    bucket_args = uri.replace("s3://", "", 1).split("/", 1)
    bucket_name = bucket_args[0]
    bucket_path = bucket_args[1] if len(bucket_args) > 1 else ""
    objects = client.list_objects(bucket_name, prefix=bucket_path, recursive=True)
    count = 0
    for obj in objects:
        # Replace any prefix from the object key with temp_dir
        subdir_object_key = obj.object_name.replace(bucket_path, "", 1).strip("/")
        # fget_object handles directory creation if does not exist
        if not obj.is_dir:
            if subdir_object_key == "":
                subdir_object_key = obj.object_name
            client.fget_object(
                bucket_name,
                obj.object_name,
                os.path.join(temp_dir, subdir_object_key),
            )
        count = count + 1
    if count == 0:
        raise RuntimeError(
            "Failed to fetch model. \
            The path or model %s does not exist."
            % (uri)
        )
    return temp_dir
