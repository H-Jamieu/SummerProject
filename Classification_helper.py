import os

import pandas as pd
import time
import torch
import csv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import seaborn as sn
from tqdm import tqdm

PATH = './Models/0.91_acc.pth'

''' This is the script for saving model and classification result: '''
''' Functions to be implemented:
    1. A validation function to test accuracy on the test dataset
    2. A recording function to recording the classification result of original files. Outputs (filename,label)
    3. A visualization function to visualize the classification results
'''


# Return a dictionary mapping class string with number label
def get_class_meaning(target):
    """
    Map number to actual classes
    :param target: Species or genus.
    :return: Dictionary. eg: {0, 'Genus_Specie'}
    """
    guideline_path = target + '_guide.csv'
    guide_data = pd.read_csv(guideline_path, header=None)
    guide_index = guide_data.set_index(1).to_dict(orient='dict')
    mydict = guide_index[0]
    return mydict


def extract_class_label(target):
    """
    Get a list of classes
    :param target: Species or genus.
    :return: List. List of genus or species.
    """
    guideline_path = target + '_guide.txt'
    # guideline_data = pd.read_csv(guideline_path, header=None)
    f = open(guideline_path, 'r', encoding='ascii')

    all_classes = f.read().splitlines()
    return all_classes


# Classify image and write the results into
def verify_model(model, test_loader, device, target, data_size, model_name):
    """
    Get the classified label of each image. Return a classification results csv file in the Results Directory.
    :param model: A pre-trained CNN model.
    :param test_loader: An instance of Data_loader class. Defines the source of image.
    :param device: Cpu or cuda.
    :param target: Species or genus.
    :param data_size: Size of the data, i.e. how many images.
    :return: Accuracy of the classification, float.
    """
    since = time.time()

    class_dict = extract_class_label(target)
    running_corrects = 0
    all_labels =[]
    all_paths = []
    all_predictions = []
    with torch.no_grad():
        # iterate over batch
        for images, labels, paths in tqdm(test_loader):
            all_paths.append(paths)
            all_labels.append(labels)

            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            all_predictions.append(predictions)
            # collect the correct predictions for each class

            running_corrects += torch.sum(predictions == labels.data)

    # Horrible efficiency
    outfile = './Results/' + str(int(since)) + '_' + target + '_' + model_name + '_result.csv'
    all_predictions = all_predictions
    with open(outfile, 'a', encoding='ascii', errors='ignore', newline='') as f_guide:
        writer = csv.writer(f_guide)
        for label_batch, prediction_batch, path_batch in zip(all_labels, all_predictions, all_paths):
            for label, prediction, path in zip(label_batch, prediction_batch, path_batch):
                row = [path]
                label = int(label.numpy())
                label_t = class_dict[label]
                row.append(label_t)
                prediction = int(prediction.cpu().data.numpy())
                pred_t = class_dict[prediction]
                row.append(pred_t)
                writer.writerow(row)
    f_guide.close()
    accu = running_corrects.double() / data_size
    print('Current test Acc: {:4f}'.format(accu))
    acc4 = str(float("{0:.4f}".format(accu)))
    new_out = './Results/' + str(int(since)) + '_' + target + '_' + model_name + acc4 + '_result.csv'
    os.rename(outfile, new_out)
    return accu


def plot_prediction(test_file, target = 'species',show_plot = True):
    """
    This function is visualizing confusion matrix of classification result.
    :param test_file: the file containing the classification result of tested model. It is expected to be a output from
    verify_model function with format of [image_path, predicted_label, real_label].
    :return: None
    """
    pred = pd.read_csv(test_file, header=None)
    y_pred, y_real = pred.iloc[:, 2], pred.iloc[:, 1]
    accCount = 0
    for y1, y2 in zip(y_pred, y_real):
        if y1 == y2:
            accCount += 1
    print(accCount / len(y_pred))
    classes = extract_class_label(target)
    cm = confusion_matrix(y_pred, y_real)
    df_cm = pd.DataFrame(cm, index=classes,
                         columns=classes)
    df_cm.to_csv(test_file.split('.csv')[0]+'_cm.csv')
    if show_plot == True:
        plt.figure()
        sn.heatmap(df_cm, annot=True)
        plt.show()


def result_visualization(img_path):
    """
    Visualizing classification results: intended function:
    1. Visualizing classification result by image path （one at a time）
    2. Visualizing a group of images by list of image path
    3. Visualizing random 9 images.
    :return: graph showing image and classification results.
    """

    return 0


#plot_prediction('./Results/1643863508species_result_0.88_resnet152.csv')

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = models.resnet152(pretrained=True)
