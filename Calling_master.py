import os
import pandas as pd
import torch
import time
import copy
from torch.utils.data import Dataset
from torchvision import transforms, models
from torch.utils.data import DataLoader
from PIL import Image
from Classification_helper import verify_model

CLASSDICT = {
    'species': 31,
    'genues': 16
}

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# The path to load model
PATH = './Models/'
MODELPATH = './Models/0.8695_acc.pth'
DEFAULTWD = os.getcwd()


class CustomImageDataset(Dataset):
    """
    DataLoader class. Sub-class of torch.utils.data Dataset class. It will load data from designated files.
    """

    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        """
        Initial function. Creates the instance.
        :param annotations_file: The file containing image directory an labels (for train and validation)
        :param img_dir: The directory containing target images.
        :param transform: Transformation applied to images. Should be a torchvision.transform type.
        :param target_transform:
        """
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        Controls what returned from data loader
        :param idx: Index of image.
        :return: image: The image array.
        label: label of training images.
        img_path: path to the image.
        """
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # image = read_image(img_path)
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, img_path


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    """
    Training function of models. The model is trained on this function.
    :param model: the decided model to be trained.
    :param criterion: loss function of the model
    :param optimizer: optimizer of training function. Should be subclass of torch.optim
    :param scheduler: scheduler of the training function. Should be subclass of torch.scheduler
    :param dataloaders: passed data loader
    :param dataset_sizes: size of the data set
    :param num_epochs: number of epohes we want to train
    :return:
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    last_corrects = 0
    cnt = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels, _ in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                # Early stopping
                if running_corrects < last_corrects:
                    cnt += 1
                    if cnt >= 3:
                        print('Early stop triggered. Best accuracy: {:.4f} with {} epochs.'.format(best_acc, epoch + 1))
                        break
                else:
                    cnt = 0
        last_corrects = running_corrects
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    if best_acc >= 0.75:
        model.load_state_dict(best_model_wts)
        model_save_path = PATH + str(time.time()) + '_' + str(best_acc) + '.pth'
        torch.save(model.state_dict(), model_save_path)
    return model


# Load train, test and validation data by phase. Phase = train, val and test. Target = genus and species
def load_data(phase, target, d_transfroms, batch_size=16):
    data_path = './Metadata/' + target + '_' + phase + '.csv'
    src_path = './Plaindata'
    data_out = CustomImageDataset(data_path, src_path, d_transfroms)
    data_size = len(data_out)
    data_loader = DataLoader(data_out, batch_size=batch_size, shuffle=True)
    return data_loader, data_size


def std_call_train(config, model, checkpoint_dir=None, data_dir=None):
    os.chdir(DEFAULTWD)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 31)
    model = model.to(device)
    dataloaders = {}
    dataset_sizes = {}
    batch_size = config['batch_size']
    dataloaders['train'], dataset_sizes['train'] = load_data('train', target, data_transforms, batch_size)
    dataloaders['val'], dataset_sizes['val'] = load_data('val', target, data_transforms, batch_size)
    optimizer_ft = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config['momentum'])
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    model_ft = train_model(model, criterion, optimizer_ft,
                           exp_lr_scheduler, dataloaders, dataset_sizes, num_epochs=25)
    return model_ft


def single_train(model, target, batch_size, n_epochs, criterion, optimizer, scheduler):
    os.chdir(DEFAULTWD)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    dataloaders['train'], dataset_sizes['train'] = load_data('train', target, data_transforms, batch_size)
    dataloaders['val'], dataset_sizes['val'] = load_data('val', target, data_transforms, batch_size)
    model_ft = train_model(model=model, criterion=criterion, optimizer=optimizer,
                           scheduler=scheduler, dataloaders=dataloaders,
                           dataset_sizes=dataset_sizes, num_epochs=n_epochs)
    return model_ft


def tune_train(config, model, target):
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, CLASSDICT[target])
    model = model.to(device)


def test_model(model, pre_trained_path, data, data_size, device, target):
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 31)
    model.load_state_dict(torch.load(pre_trained_path, map_location=torch.device(device)))

    verify_model(model, data, device, target, data_size)


'''
Sample Mean and Std in (r,g,b) from validation set
    mean: 0.119743854 0.12807578 0.23815322
    std: 0.113375455 0.112062685 0.22721237
Could increase accuracy
'''

data_transforms = transforms.Compose([transforms.Resize([256, 256]),
                                      transforms.CenterCrop([224, 224]),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])])
target = 'species'
dataloaders = {}
dataset_sizes = {}
batch_size: int = 16
# dataloaders['train'], dataset_sizes['train'] = load_data('train', target, data_transforms, batch_size)
# dataloaders['val'], dataset_sizes['val'] = load_data('val', target, data_transforms, batch_size)
dataloaders['test'], dataset_sizes['test'] = load_data('test', target, data_transforms, batch_size)
#
# print(dataset_sizes)
#
# train_features, train_labels, train_path = next(iter(dataloaders['train']))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = torch.nn.Linear(num_ftrs, CLASSDICT[target])
# model_ft.load_state_dict(torch.load(MODELPATH, map_location=torch.device(device)))
#
# verify_model(model_ft, dataloaders['test'], device, target, dataset_sizes['test'])

# test_model(model_ft, MODELPATH, dataloaders['test'], dataset_sizes['test'], device, target)

# Observe that all parameters are being optimized
optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
'''Need a parameter searching grid'''
# optimizer_ft = torch.optim.ASGD(model_ft.parameters(), lr=0.001,lambd=0.0002)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

# result = tune.run(
#     partial(std_call_train,
#     model=model_ft),
#     resources_per_trial={"cpu": 20, "gpu": 1},
#     config=config)

# test_model(model_ft, './Models/0.91_acc.pth', dataloaders['test'], dataset_sizes['test'], device, target)
criterion = torch.nn.CrossEntropyLoss()
model_ft = single_train(model_ft, 'species', 16, 25, criterion, optimizer_ft, exp_lr_scheduler)
