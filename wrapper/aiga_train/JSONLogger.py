import json
import logging


class JSONLogger(logging.FileHandler):
  def __init__(self, filename):
    logging.FileHandler.__init__(self, filename)

  def emit(self, record):
    msg = record.msg
    try:
      json.loads(msg)
      super(JSONLogger, self).emit(record)
    except:
      pass