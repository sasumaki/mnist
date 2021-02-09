import os
import sys
import argparse
import json
import importlib
import logging
import yaml
from aiga_train import JSONLogger
  

def boolify(s):
    if s == 'True':
        return True
    if s == 'False':
        return False
    raise ValueError("huh?")

def autoconvert(s):
    for fn in (boolify, int, float):
        try:
            return fn(s)
        except ValueError:
            pass
    return s

def parse_params(parameters):
  parsed_parameters = {}
  for param in parameters:
    name = param.get("name")
    value = param.get("value")
    # TODO: figure out ohow to pass types
    parsed_parameters[name] = autoconvert(value)


  return parsed_parameters

def main():
  sys.path.append(os.getcwd())
  print("wrapping this")
  parser = argparse.ArgumentParser()

  parser.add_argument("model", type=str, default="train", help="Name of the user interface.")
  parser.add_argument("--train_method", type=str, default=os.environ.get("TRAIN_METHOD", "train"))
  parser.add_argument("--parameters", type=str, default="[]")


  args=parser.parse_args()

  print(args)
  parsed = parse_params(json.loads(args.parameters))

  train_method = args.train_method
  client_file = args.model
  client_model = importlib.import_module(client_file)

  user_method = getattr(client_model, train_method)


  logger = logging.getLogger(client_file)
  logger.setLevel(logging.DEBUG)

  fh = JSONLogger.JSONLogger('spam.log')
  fh.setLevel(logging.DEBUG)

  logger.addHandler(fh)

  user_method(**parsed)


  metadata_yaml = r"./outputs/training_metadata.yaml"
  if "metadata_file" in parsed:
    metadata_yaml = parsed["metadata_file"]


  a_file = open("spam.log")
  print('lets read file')
  lines = a_file.readlines()
  runtime_logs = []
  for line in lines:
      runtime_logs.append(line)
  a_file.close()

  with open(metadata_yaml,'r') as yamlfile:
      cur_yaml = yaml.safe_load(yamlfile)
      cur_yaml.update({ "parameters": parsed })
      cur_yaml.update({ "runtime_logs": runtime_logs })

      print(cur_yaml)

  with open(metadata_yaml,'w') as yamlfile:
    yaml.safe_dump(cur_yaml, yamlfile) # Also note the safe_dump
  
