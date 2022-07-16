import argparse
from Train import test_model
import torch
from torchvision import models
from Classification_helper import plot_prediction

'''
    Default model in case no parameter of model provided.
    1. Get the model path
    2. Decode the model
    3. Doing classification and output the result
'''
MODEL_BASE = 'Models/'
MODEL_PATH = '1643965937_0.92_genus_vgg16.pth'
RESULT_BASE = 'Results/'
RESULT_PATH = '1644219241_genus_vgg16_result.csv'

def determine_model(arg_model, arg_pretrain, arg_classes):
    if arg_model.lower() == 'resnet18':
        model = models.resnet18(pretrained=arg_pretrain)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, arg_classes)
    elif arg_model.lower() == 'resnet34':
        model = models.resnet34(pretrained=arg_pretrain)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, arg_classes)
    elif arg_model.lower() == 'resnet101':
        model = models.resnet101(pretrained=arg_pretrain)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, arg_classes)
    elif arg_model.lower() == 'resnet152':
        model = models.resnet152(pretrained=arg_pretrain)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, arg_classes)
    elif arg_model.lower() == 'vgg16':
        model = models.vgg16(pretrained=arg_pretrain)
        model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=arg_classes)
    elif arg_model.lower() == 'vgg19':
        model = models.vgg19(pretrained=arg_pretrain)
        model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=arg_classes)
    elif arg_model.lower() == 'efficientnet':
        model = models.efficientnet_b4(pretrained=arg_pretrain)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, arg_classes)
    elif arg_model.lower() == 'convnext_tiny':
        model = models.convnext_tiny(pretrained=arg_pretrain)
        model.num_classes = arg_classes
    elif arg_model.lower() == 'convnext_large':
        model = models.convnext_large(pretrained=arg_pretrain)
        model.num_classes = arg_classes
    elif arg_model.lower() == 'regnet_y_16gf':
        model = models.regnet_y_16gf(pretrained=arg_pretrain)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, arg_classes)
    elif arg_model.lower() == 'vit_b_16':
        model = models.vit_b_16(pretrained=arg_pretrain)
        model.num_classes = arg_classes
    elif arg_model.lower()=='efficient_v2':
        model = models.efficientnet_v2_l(pretrained=arg_pretrain)
        model.num_classes = arg_classes
    else:
        model = models.resnet50(pretrained=arg_pretrain)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, arg_classes)
    return model

parser = argparse.ArgumentParser()
parser.add_argument("--model_path",
                    type=str,
                    default=MODEL_PATH,
                    help="Model path for your classification task.")
parser.add_argument("--model",
                    type=str,
                    default='resnet50',
                    help="Model for your classification task.")
parser.add_argument("--pretrained",
                    type=bool,
                    default=False,
                    help="Indicator if you want the model to be pre-trained.")
parser.add_argument("--classes",
                    type=int,
                    default=31,
                    help="The count of classes for classification.")
parser.add_argument("--mode",
                    type=str,
                    default='test',
                    help="The mode for running the programme. Test for doing inference and cm for getting the "
                         "confusion matrix.")
parser.add_argument("--result_path",
                    type=str,
                    default=RESULT_PATH,
                    help="The count of classes for classification.")
parser.add_argument("--show_plot",
                    type=bool,
                    default=False,
                    help="The count of classes for classification.")
parser.add_argument("--target",
                    type=str,
                    default='species',
                    help="The count of classes for classification.")

args = parser.parse_args()
if args.mode == 'cm':
    # for producing confusion matrix
    result_dir = RESULT_BASE+args.result_path
    plot_prediction(result_dir, args.target, args.show_plot)
else:
    # for classification using the models
    model_info = args.model
    model = determine_model(model_info, args.pretrained, args.classes)
    target = args.target
    model_dir = MODEL_BASE+args.model_path
    print(model_dir)
    test_model(model, model_dir, target, model_info, args.mode)