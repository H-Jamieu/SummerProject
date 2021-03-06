import csv
import shutil
import autogluon.core as ag
from autogluon.vision import ImageDataset, ImagePredictor

'''
The file is intended to include some debug only functions to fix ad-hoc errors
'''

error_log = 'error_log.csv'


def extract_error_images(error_log):
    '''

    :param error_log: file name of the error log containing the error images
    :return: none

    The function will copy images with detected problems into a temporary directory waiting for cropping by hand.
    '''
    data_path = 'Data/'
    errors = open(error_log)
    csvreader = csv.reader(errors)
    for row in csvreader:
        src = data_path + row[1] + '/images/' + row[0]
        dst = '../tmp/' + row[0]
        shutil.copy(src, dst)

def test_autogluon():
    model_list = ImagePredictor.list_models()
    print(model_list)

test_autogluon()
#extract_error_images(error_log)