import os
import pandas as pd
import torch
import time
import copy
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from Classification_helper import verify_model
from tqdm import tqdm

# data_transforms = transforms.Compose([transforms.Resize([512, 512]),
#                                       transforms.CenterCrop([448, 448]),
#                                       transforms.ToTensor(),
#                                       transforms.Normalize(
#                                           mean=[0.12, 0.128, 0.238],
#                                           std=[0.113, 0.112, 0.227])])

data_transforms = transforms.Compose([transforms.Resize([256, 256]),
                                      transforms.CenterCrop([224, 224]),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])])

'''
Sample Mean and Std in (r,g,b) from validation set
    mean: 0.119743854 0.12807578 0.23815322
    std: 0.113375455 0.112062685 0.22721237
Could increase accuracy
'''

CLASSDICT = {
    'species': 31,
    'genus': 15
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
        :param annotations_file: The file containing image directory and labels (for train and validation)
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


def train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device, tolerance=5):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    # Sluggish factor is indicating how many rounds the loss doesn't improved
    sluggish = 0
    timelist=[]
    for epoch in range(1, num_epochs+1):
        epochsince = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        cnt = 0
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels, _ in tqdm(dataloaders[phase]):
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
                cnt += 1
                if cnt % 10 == 0:
                    print("", end=f"\rCompleted: {cnt} Batches")
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            # if phase == 'train':
            #     trainloss.append(epoch_loss)
            # else:
            #     valloss.append(epoch_loss)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':
                # tune.report(loss=epoch_loss, accuracy=epoch_acc)
                sluggish += 1
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    sluggish = 0

        if sluggish >= tolerance:
            print(f'Best validation loss did not improved over {tolerance} epochs. Break.')
            break

        timelist.append(time.time()-epochsince)
        print(f'Time for epoch {epoch}: {(timelist[-1] // 60):.0f}m {(timelist[-1] % 60):.0f}s.')
        remainingtime = (num_epochs-epoch)*(sum(timelist)/len(timelist))
        print(f'Estimated remaining time: {(remainingtime // 60):.0f}m {(remainingtime % 60):.0f}s.')
        finishingtime = time.localtime(time.time()+remainingtime)
        print(f'Estimated finishing time: {finishingtime.tm_year}/{finishingtime.tm_mon}/{finishingtime.tm_mday} {finishingtime.tm_hour}:{finishingtime.tm_min}')


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    model_save_path = PATH+str(int(time.time()))+'_'+str(best_acc.cpu().data.numpy().round(decimals=2))
    #torch.save(model.state_dict(), model_save_path)
    return model, model_save_path


# Load train, test and validation data by phase. Phase = train, val and test. Target = genus and species
def load_data(phase, target, d_transfroms, batch_size=16):
    data_path = './Metadata/' + target + '_' + phase + '.csv'
    src_path = ''
    data_out = CustomImageDataset(data_path, src_path, d_transfroms)
    data_size = len(data_out)
    data_loader = DataLoader(data_out, batch_size=batch_size, shuffle=True)
    return data_loader, data_size

"""----------------------------------to be modified----------------------------------"""
'''
    std_call_train included the standard testing interface for the model for test only.
    The default model is resnet18 and batch_size is 96.
    The default settings is only acceptable for running for testing in new environment.
'''
def std_call_train(model, batch_size = 96):
    os.chdir(DEFAULTWD)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 31)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    dataloaders, dataset_sizes = [{}, {}]
    dataloaders['train'], dataset_sizes['train'] = load_data('train', target='species',
                                                             d_transfroms=data_transforms, batch_size=batch_size)
    dataloaders['val'], dataset_sizes['val'] = load_data('val', target='species',
                                                         d_transfroms=data_transforms, batch_size=batch_size)
    optimizer_ft = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    model_ft = train_model(model, criterion, optimizer_ft,
                          exp_lr_scheduler, num_epochs=25,dataloaders=dataloaders,dataset_sizes=dataset_sizes,device=device)
    return model_ft

def single_train(model, target, batch_size, n_epochs, criterion, optimizer, scheduler):
    os.chdir(DEFAULTWD)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    dataloaders, dataset_sizes = [{}, {}]
    dataloaders['train'], dataset_sizes['train'] = load_data('train', target, data_transforms, batch_size)
    dataloaders['val'], dataset_sizes['val'] = load_data('val', target, data_transforms, batch_size)
    model_ft = train_model(model=model, criterion=criterion, optimizer=optimizer,
                          scheduler=scheduler, num_epochs=n_epochs, dataloaders=dataloaders, dataset_sizes=dataset_sizes, device=device)
    return model_ft


def tune_train(config, model, target):
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, CLASSDICT[target])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)


def test_model(model, pre_trained_path, target, model_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(pre_trained_path, map_location=torch.device(device)))
    model = model.to(device)
    test_data, data_size = load_data('test', target, data_transforms)
    verify_model(model, test_data, device, target, data_size, model_name)


def loading_data():
    return 0


