import os
import pickle

import numpy as np
from torch.utils.data.sampler import Sampler
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import torchvision
import torchvision.models as models

class UnifLabelSampler(Sampler):
    """Samples elements uniformely accross pseudolabels.
        Args:
            N (int): size of returned iterator.
            images_lists: dict of key (target), value (list of data with this target)
    """

    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        nmb_non_empty_clusters = 0
        for i in range(len(self.images_lists)):
            if len(self.images_lists[i]) != 0:
                nmb_non_empty_clusters += 1

        size_per_pseudolabel = int(self.N / nmb_non_empty_clusters) + 1
        res = np.array([])

        for i in range(len(self.images_lists)):
            # skip empty clusters
            if len(self.images_lists[i]) == 0:
                continue
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel)
            )
            res = np.concatenate((res, indexes))

        np.random.shuffle(res)
        res = list(res.astype('int'))
        if len(res) >= self.N:
            return res[:self.N]
        res += res[: (self.N - len(res))]
        return res

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return len(self.indexes)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def learning_rate_decay(optimizer, t, lr_0):
    for param_group in optimizer.param_groups:
        lr = lr_0 / np.sqrt(1 + lr_0 * param_group['weight_decay'] * t)
        param_group['lr'] = lr


class Logger(object):
    """ Class to update every epoch to keep trace of the results
    Methods:
        - log() log and save
    """

    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with open(os.path.join(self.path), 'wb') as fp:
            pickle.dump(self.data, fp, -1)

# Early Stopping class
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None, num_images_per_class=None):
        """
        Args:
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            num_images_per_class (int, optional): Number of images to be used from each class. If None, all images are used.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.classes = []

        # Assuming each subfolder in the directory represents a label
        for label_dir in sorted(os.listdir(data_dir)):
            label_path = os.path.join(data_dir, label_dir)
            if os.path.isdir(label_path):
                self.classes.append(label_dir)
                img_files = os.listdir(label_path)
                if num_images_per_class is not None:
                    img_files = img_files[:min(num_images_per_class, len(img_files))]
                for img_file in img_files:
                    self.images.append(os.path.join(label_path, img_file))
                    self.labels.append(self.classes.index(label_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = Image.open(img_name).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class DataLoaderFactory:
    def __init__(self, dataset, train_split=0.7, validation_split=0.15, batch_size=16, num_workers=10, shuffle=True):
        """
        Args:
            dataset (Dataset): The custom dataset to be split into training, validation, and test datasets.
            train_split (float): Proportion of the dataset to include in the training split (between 0 and 1).
            validation_split (float): Proportion of the dataset to include in the validation split (between 0 and 1).
            batch_size (int): How many samples per batch to load.
            num_workers (int): How many subprocesses to use for data loading.
            shuffle (bool): Whether to shuffle the dataset at every epoch.
        """
        self.dataset = dataset
        self.train_split = train_split
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        # Split the dataset into training, validation, and test sets
        total_size = len(dataset)
        train_size = int(total_size * train_split)
        validation_size = int(total_size * validation_split)
        test_size = total_size - train_size - validation_size
        self.train_dataset, self.validation_dataset, self.test_dataset = random_split(dataset, [train_size, validation_size, test_size])

    def get_train_loader(self):
        """Returns the DataLoader for the training dataset."""
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        return train_loader

    def get_validation_loader(self):
        """Returns the DataLoader for the validation dataset."""
        validation_loader = DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return validation_loader

    def get_test_loader(self):
        """Returns the DataLoader for the test dataset."""
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return test_loader

# Example usage
# First, create an instance of your custom dataset
# dataset = CustomDataset(data_dir='path/to/data', transform=your_transforms)

# Then, create an instance of DataLoaderFactory
# data_loader_factory = DataLoaderFactory(dataset, train_split=0.7, validation_split=0.15, batch_size=32, num_workers=4, shuffle=True)

# # Get the train, validation, and test DataLoaders
# train_loader = data_loader_factory.get_train_loader()
# validation_loader = data_loader_factory.get_validation_loader()
# test_loader = data_loader_factory.get_test_loader()

def print_dataset_info(dataset_dir, dataset_name="Dataset"):
    """
    Prints information about the dataset.

    Args:
    - dataset_dir (str): Path to the dataset directory.
    - dataset_name (str): Name of the dataset.
    """
    if not os.path.isdir(dataset_dir):
        raise ValueError(f"Directory does not exist: {dataset_dir}")

    # Count the number of images in each class
    class_counts = []
    classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    for class_dir in classes:
        class_dir_path = os.path.join(dataset_dir, class_dir)
        num_images = len([name for name in os.listdir(class_dir_path) if os.path.isfile(os.path.join(class_dir_path, name))])
        class_counts.append(num_images)

    class_counts = np.array(class_counts)

    # Print dataset information
    print(f"Dataset {dataset_name}")
    print(f"Number of classes: {len(classes)}")
    print(f"Total images     : {int(class_counts.sum())}")
    print(f"Average images   : {class_counts.mean():6.2f}")
    print(f"Std images       : {class_counts.std():6.2f}")
    print(f"Maximum images   : {int(class_counts.max())}")
    print(f"Minimum images   : {int(class_counts.min())}")
    print(f"Class_counts   : {[(c, class_c) for (c, class_c) in zip(classes, class_counts)]}")


# Define Sobel filters as convolutional layers
class SobelConv(nn.Module):
    def __init__(self):
        super(SobelConv, self).__init__()
        self.sobel_x = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
        self.sobel_y = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)

        # Sobel filter weights for x and y direction
        sobel_kernel_x = torch.tensor([[-1., 0., 1.],
                                    [-2., 0., 2.],
                                    [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_kernel_y = torch.tensor([[-1., -2., -1.],
                                    [ 0.,  0.,  0.],
                                    [ 1.,  2.,  1.]]).view(1, 1, 3, 3)

        # Expand the weights to apply the same filter to each input channel
        self.sobel_x.weight = nn.Parameter(sobel_kernel_x.repeat(3, 1, 1, 1), requires_grad=False)
        self.sobel_y.weight = nn.Parameter(sobel_kernel_y.repeat(3, 1, 1, 1), requires_grad=False)

    def forward(self, x):
        x_x = self.sobel_x(x)
        x_y = self.sobel_y(x)
        # Compute magnitude across color channels and combine
        magnitude = torch.sqrt(x_x**2 + x_y**2)
        # Sum across the color channels or take the maximum
        return magnitude.sum(1, keepdim=True)


# Define Sobel filters as convolutional layers
class SobelConv2(nn.Module):
    def __init__(self):
        super(SobelConv, self).__init__()
        self.sobel_x = nn.Conv2d(2, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=2)
        self.sobel_y = nn.Conv2d(2, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=2)

        # Sobel filter weights for x and y direction
        sobel_kernel_x = torch.tensor([[-1., 0., 1.],
                                    [-2., 0., 2.],
                                    [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_kernel_y = torch.tensor([[-1., -2., -1.],
                                    [ 0.,  0.,  0.],
                                    [ 1.,  2.,  1.]]).view(1, 1, 3, 3)

        # Expand the weights to apply the same filter to each input channel
        self.sobel_x.weight = nn.Parameter(sobel_kernel_x.repeat(2, 1, 1, 1), requires_grad=False)
        self.sobel_y.weight = nn.Parameter(sobel_kernel_y.repeat(2, 1, 1, 1), requires_grad=False)

    def forward(self, x):
        x_x = self.sobel_x(x)
        x_y = self.sobel_y(x)
        # Compute magnitude across color channels and combine
        magnitude = torch.sqrt(x_x**2 + x_y**2)
        # Sum across the color channels or take the maximum
        return magnitude.sum(1, keepdim=True)



def show_batch(images, labels, class_names=None, figsize=(10, 10)):
    """
    Visualize a batch of images.

    Args:
    - images (torch.Tensor): A batch of images (assumed to be in the format BxCxHxW).
    - labels (list or torch.Tensor): The labels corresponding to each image.
    - class_names (list, optional): List of class names corresponding to the labels.
    - figsize (tuple): Size of the figure to be displayed.
    """
    images = images.numpy()
    fig, axes = plt.subplots(len(images) // 4, 4, figsize=figsize)
    axes = axes.flatten()

    for i, (img, label) in enumerate(zip(images, labels)):
        ax = axes[i]
        img = torchvision.transforms.functional.to_pil_image(img)
        ax.imshow(img)
        ax.axis('off')
        if class_names:
            label = class_names[label]
        ax.set_title(label)

    plt.tight_layout()
    plt.show()

def load_model(arch, num_classes, checkpoint_path, sobel=False, deepcluster_weights=False, checkpoint_key='state_dict'):
    """
    Load a model with a specified architecture and optional modifications.

    Args:
    - arch (str): Architecture of the model (e.g., 'resnet18', 'resnet50').
    - num_classes (int): Number of classes for the final layer.
    - checkpoint_path (str): Path to the model checkpoint.
    - sobel (bool): Whether to apply Sobel filter at the first layer.
    - deepcluster_weights (bool): Whether to use a specific setting for deep clustering.
    - checkpoint_key (str): Key to access the state_dict in the checkpoint.

    Returns:
    - model (torch.nn.Module): Loaded PyTorch model.
    """
    # Retrieve the model class dynamically from torchvision.models
    if hasattr(models, arch):
        model_class = getattr(models, arch)
        model = model_class()
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    # model = get_model(arch, weights=None)

    # Apply Sobel filter if required
    if sobel:
        if arch not in ['resnet18', 'resnet50']:  # Extend this list as needed
            raise ValueError(f"Sobel filter not implemented for architecture: {arch}")
        
        sobel_conv = SobelConv()  # Ensure SobelConv is defined elsewhere
        model.conv1 = nn.Sequential(
            sobel_conv,
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False))

    num_ftrs = model.fc.in_features

    if deepcluster_weights:
        model.fc = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(num_ftrs, num_classes))
    else:
        model.fc = nn.Linear(num_ftrs, num_classes)

    # Load the checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint[checkpoint_key])

    except FileNotFoundError:
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    except KeyError:
        raise KeyError(f"State dict key '{checkpoint_key}' not found in the checkpoint")

    return model

def my_model(arch, weights=None, sobel=False):
    # model = torchvision.models.resnet18(weights=weights)
    # model = get_model(arch, weights=weights)
    if arch == "resnet18":
        model = my_resnet18(weights=weights, sobel=sobel)
    elif arch == "resnet50":
        model = my_resnet50(weights=weights, sobel=sobel)
    # elif arch == "googlenet":
    #     model = my_googlenet(weights=weights, sobel=sobel)
    # elif arch == "densenet121":
    #     model = my_densenet121(weights=weights, sobel=sobel)

    return model

def my_resnet18(weights=None, sobel=False):
    # Load ResNet18
    model = torchvision.models.resnet18(weights=weights)

    if sobel:
        # Replace the first convolutional layer of ResNet-18
        sobel_conv = SobelConv()
        model.conv1 = nn.Sequential(
            sobel_conv,
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        )

    def new_forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten
        if self.fc:
            x = self.fc(x)
        return x

    model.forward = new_forward.__get__(model)

    return model


def my_resnet50(weights=None, sobel=False):
    # Load ResNet18
    model = torchvision.models.resnet50(weights=weights)

    if sobel:
        # Replace the first convolutional layer of ResNet-18
        sobel_conv = SobelConv()
        model.conv1 = nn.Sequential(
            sobel_conv,
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        )

    def new_forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten
        if self.fc:
            x = self.fc(x)
        return x

    model.forward = new_forward.__get__(model)
    return model

def format_time(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:.2f}"

def calculate_mean_std(loader):
    # Variables for mean, standard deviation, and number of batches
    mean = 0.0
    std = 0.0
    nb_samples = 0.0
    
    for data, _ in loader:
        # Batch data is of shape (B, C, H, W)
        batch_samples = data.size(0)
        
        # Rearrange the data to be in shape of (B, H*W, C)
        data = data.view(batch_samples, data.size(1), -1)
        
        # Update mean and standard deviation
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        
        # Update total number of images
        nb_samples += batch_samples
    
    # Final mean and standard deviation
    mean /= nb_samples
    std /= nb_samples
    
    return mean, std