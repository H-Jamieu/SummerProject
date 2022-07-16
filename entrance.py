import argparse
import torch
from torchvision import models
from Train import single_train

"""
This is an entrance for receiving terminal command lines and 
calling training functions according to specified model and hyper-parameters.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--lr",
                    type=float,
                    default=0.01,
                    help="learning rate")
parser.add_argument("--batch_size",
                    type=int,
                    default=64,
                    help="batch size for training")
parser.add_argument("--epochs",
                    type=int,
                    default=25,
                    help="training epoches")
parser.add_argument("--model",
                    type=str,
                    default='resnet50',
                    help="The model for training. For example: resnet152.")
parser.add_argument("--target",
                    type=str,
                    default='species',
                    help="The target for training, species or genus.")
parser.add_argument("--classes",
                    type=int,
                    default=31,
                    help="The count of classes for classification.")
parser.add_argument("--pretrained",
                    type=bool,
                    default=False,
                    help="Indicator if you want the model to be pre-trained.")
parser.add_argument("--criterion",
                    type=str,
                    default='CrossEntropyLoss',
                    help="The loss function for training.")
parser.add_argument("--optimizer",
                    type=str,
                    default='SGD',
                    help="The optimizer for training, for example SGD.")
parser.add_argument("--momentum",
                    type=float,
                    default=0.9,
                    help="The step size for optimizer.")
parser.add_argument("--scheduler",
                    type=str,
                    default='steplr',
                    help="The scheduler for training.")
parser.add_argument("--step_size",
                    type=int,
                    default=7,
                    help="The step size for the scheduler, should be integer.")
parser.add_argument("--gamma",
                    type=float,
                    default=0.1,
                    help="The gamma for the scheduler, should be float.")
parser.add_argument("--lambd",
                    type=float,
                    default=0.1,
                    help="The lambda for the optimizer, should be float.")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using ', device, '.')


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


def determine_optimizer(model, arg_optimizer, arg_lr, arg_momentum, arg_lambda):
    if arg_optimizer.lower() == 'ASGD':
        optimizer = torch.optim.ASGD(model.parameters(), lr=arg_lr, lambd=arg_lambda)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=arg_lr, momentum=arg_momentum)
    return optimizer


def determine_criterion(arg_criterion):
    return torch.nn.CrossEntropyLoss()


def determine_scheduler(optimizer, arg_scheduler, arg_step_size, arg_gamma):
    if arg_scheduler.lower() == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=arg_step_size, gamma=arg_gamma)
    elif arg_scheduler.lower() == 'exponentiallr':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=arg_gamma)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=arg_step_size, gamma=arg_gamma)
    return scheduler


model = determine_model(args.model, args.pretrained, args.classes)
model = model.to(device)
optimizer = determine_optimizer(model,args.optimizer,args.lr, args.momentum, args.lambd)
criterion = determine_criterion(args.criterion)
scheduler = determine_scheduler(optimizer, args.scheduler, args.step_size, args.gamma)

try:
    model, save_path = single_train(model=model, target=args.target, batch_size=args.batch_size,
                                n_epochs=args.epochs, criterion=criterion, optimizer=optimizer,
                                scheduler=scheduler)
    save_path = save_path+'_'+args.target+'_'+args.model+'.pth'
    torch.save(model.state_dict(), save_path)
except Exception as e:
    print(str(e))
    exit('Training failed.')

# arg = sys.argv[1:]
# l = ['model', 'target', 'n_class', 'phase', 'pretrained', 'batch_size', 'n_epochs', 'criterion', 'optimizer',
#      'learning_rate',
#      'momentum', 'scheduler', 'step_size', 'gamma']
#
# if len(arg) == 0:
#     print('Using default settings for training. Model: Resnet18.')
#     model = models.resnet18(pretrained=True)
#     std_call_train(model)
#
# elif len(arg) == 14:
#     modeldict = dict(zip(l, arg))
#     input = input(str(modeldict)[1:-1].replace('\'', '').replace(', ', '\n') +
#                   '\nContinue with the above model and hyper-parameters? ([y]/n): ')
#     if input == 'y':
#         model, optimizer, criterion, scheduler = [None, None, None, None]
#
#         """------------------------------------load model------------------------------------"""
#         if 'resnet' in modeldict['model']:
#             if modeldict['model'] == 'resnet18':
#                 model = models.resnet18(pretrained=True if modeldict['pretrained'] == 't' else False)
#
#             elif modeldict['model'] == 'resnet34':
#                 model = models.resnet34(pretrained=True if modeldict['pretrained'] == 't' else False)
#             elif modeldict['model'] == 'resnet50':
#                 model = models.resnet50(pretrained=True if modeldict['pretrained'] == 't' else False)
#             elif modeldict['model'] == 'resnet101':
#                 model = models.resnet101(pretrained=True if modeldict['pretrained'] == 't' else False)
#             elif modeldict['model'] == 'resnet152':
#                 model = models.resnet152(pretrained=True if modeldict['pretrained'] == 't' else False)
#             else:
#                 exit(modeldict['model'] + 'is not supported yet. Command \'python entrance.py help\' for information.')
#             model.fc = torch.nn.Linear(model.fc.in_features, int(modeldict['n_class']))
#         elif 'vgg' in modeldict['model']:
#             if modeldict['model'] == 'vgg16':
#                 model = models.vgg16(pretrained=True if modeldict['pretrained'] == 't' else False)
#                 model.classifier[6] = torch.nn.Linear(4096, int(modeldict['n_class']))
#             else:
#                 exit(modeldict['model'] + 'is not supported yet. Command \'python entrance.py help\' for information.')
#         else:
#             exit(modeldict['model'] + ' is not supported yet. Command \'python entrance.py help\' for information.')
#         print(modeldict['model'] + ' has been loaded.')
#
#         """-----------------------------------load optimizer------------------------------------"""
#         if modeldict['optimizer'] == 'sgd':
#             try:
#                 optimizer = torch.optim.SGD(params=model.parameters(), lr=float(modeldict['learning_rate']),
#                                             momentum=float(modeldict['momentum']))
#             except Exception as e:
#                 print(str(e))
#                 exit(
#                     'Please check if learning_rate and momentum have been entered correctly. Command \'python entrance.py help\' for information.')
#         else:
#             exit(modeldict['optimizer'] + ' is not supported yet. Command \'python entrance.py help\' for information.')
#         print(modeldict['optimizer'] + ' has been loaded.')
#
#         """------------------------------------load criterion------------------------------------"""
#         if modeldict['criterion'] == 'cel':
#             criterion = torch.nn.CrossEntropyLoss()
#         else:
#             exit(modeldict['criterion'] + ' is not supported yet. Command \'python entrance.py help\' for information.')
#         print(modeldict['criterion'] + ' has been loaded.')
#
#         """------------------------------------load scheduler------------------------------------"""
#         if modeldict['scheduler'] == 'steplr':
#             try:
#                 scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=int(modeldict['step_size']),
#                                                             gamma=float(modeldict['gamma']))
#             except Exception as e:
#                 print(str(e))
#                 exit('Please check if step_size and gamma have been entered correctly. '
#                      'Command \'python entrance.py help\' for information.')
#         else:
#             exit(modeldict['scheduler'] + ' is not supported yet. Command \'python entrance.py help\' for information.')
#         print(modeldict['scheduler'] + ' has been loaded.')
#
#         """------------------------------------train or test model------------------------------------"""
#         if modeldict['phase'] == 'train':
#             print('-----start training-----')
#             try:
#                 model = single_train(model=model, target=modeldict['target'], batch_size=int(modeldict['batch_size']),
#                                      n_epochs=int(modeldict['n_epochs']), criterion=criterion, optimizer=optimizer,
#                                      scheduler=scheduler)
#             except Exception as e:
#                 print(str(e))
#                 exit('Training failed.')
#         elif model['phase'] == 'test':
#             print('To be implemented')
#     else:
#         exit('Cancelled')
# elif len(arg) == 1 and arg[0] == 'help':
#     print('help')
# else:
#     exit('Use the command line \'python entrance.py help\' for information.')
