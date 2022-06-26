import argparse
import os
import regex as re
import multiprocessing
import threading
from functools import partial
from contextlib import contextmanager

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, default='./format.yaml', help='format yaml path')
    parser.add_argument('--mode', type=int, default=1, help='for set the mode of execution')
    opt = parser.parse_args()
    return opt

def files(path):
  for file in os.listdir(path):
    if os.path.isfile(os.path.join(path, file)):
      yield file

def folders(path):
  for file in os.listdir(path):
    if not os.path.isfile(os.path.join(path, file)):
      yield file

def conditional_folders(path, condition):
  '''
  return folders with name matched the condition pattern
  '''
  for file in os.listdir(path):
    if not os.path.isfile(os.path.join(path, file)) and bool(re.search(condition, file)):
      yield file

def conditional_files(path, condition):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)) and bool(re.search(condition, file)):
            yield file
