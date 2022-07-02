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


def keyword_from_path(full_path):
    """
    The function is used for return key word for searching grids by using the full file path or file name.
    return the keywords for searching in the data sheet: core_name, slide_name, grid_no
    """
    base_name = os.path.basename(full_path).split('.')[0]
    full_path_data = base_name.split('_')
    if len(full_path_data) == 5:
        core_name = full_path_data[0]
        slide_name = full_path_data[1]
        grid_no = int(full_path_data[-1])
    elif len(full_path_data) == 6:
        core_name = full_path_data[0]
        slide_name = full_path_data[1] + '_' + full_path_data[2]
        grid_no = int(full_path_data[-1])
    else:
        raise f'File name {base_name} not matching the required format.'
    return core_name, slide_name, grid_no

def keyword_from_folder(folder_name):
    """
    The function is used for return key word for searching grids by using the full file path or file name.
    return the keywords for searching in the data sheet: core_name, slide_name, grid_no
    """
    base_name = os.path.basename(folder_name).split('.')[0]
    full_path_data = base_name.split('_')
    if len(full_path_data) == 3:
        core_name = full_path_data[0]
        slide_name = full_path_data[1]
    elif len(full_path_data) == 4:
        core_name = full_path_data[0]
        slide_name = full_path_data[1] + '_' + full_path_data[2]
    else:
        raise f'File name {base_name} not matching the required format.'
    return core_name, slide_name

def get_target(target):
    if bool(re.search('genus', target.lower())):
        return 8
    elif bool(re.search('species', target.lower())):
        return 10
    raise f'Invalid target: {target}'