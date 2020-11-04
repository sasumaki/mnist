from seldon_core.seldon_client import SeldonClient
import numpy as np
import json
import tensorflow_datasets as tfds
import random
import warnings
def test():
  ds = tfds.load(name="mnist", split="test", as_supervised=True)


  sc = SeldonClient(deployment_name="mnist-model", namespace="seldon-system", gateway_endpoint="localhost:8081", gateway="istio")
  print(sc.config)
  test_size = 100
  corrects = 0
  data = ds.take(test_size).cache()
  for image, label in data:

    # plt.imshow(data.reshape(28,28))
    # plt.colorbar()
    # plt.show()
    r = sc.predict(data=np.array(image), gateway="istio",transport="rest")
    print(r.msg)
    assert(r.success==True)

    res = r.response['data']['tensor']['values']
    print(res)
    prediction = int(np.argmax(np.array(res).squeeze(), axis=0))
    print("predicted: ", prediction, "Truth: ", int(label))
    if prediction == int(label):
      corrects = corrects + 1
  warnings.warn("Sanity checks")
  warnings.warn(corrects/test_size)
  print(corrects/test_size)
  assert(corrects/test_size > 0.9)