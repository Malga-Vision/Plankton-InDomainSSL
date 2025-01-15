import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torchvision import models, transforms
from utils import CustomDataset, SobelConv
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="""Extract features.""")

    parser.add_argument('--data', type=str, help='path to train dataset')
    parser.add_argument('--images_per_class', type=int, default=None, help='number of images per class')
    parser.add_argument('--test_data', type=str, help='path to test dataset')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--arch', type=str, help='architecture')
    parser.add_argument('--pretrain_method', type=str, help='Name of the pretrained method')
    parser.add_argument('--model', type=str, default="", help='path to pretrained model')
    parser.add_argument('--deepcluster', action='store_true', help='DeepCluster weights')
    parser.add_argument('--sobel', action='store_true', help='Sobel filter')
    parser.add_argument('--imagenet', action='store_true', help='ImageNet pretrained model')
    parser.add_argument('--workers', default=2, type=int,
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', type=int, default=50, help='number of total epochs to run (default: 50)')
    # parser.add_argument('--head', type=int, default=1000, help='size fc layer (default: 1000)')
    parser.add_argument('--seed', type=int, default=31, help='random seed')
    parser.add_argument('--verbose', action='store_true', help='chatty')
    return parser.parse_args()


class MyModel(nn.Module):
    def __init__(self, backbone, head, device):
        super(MyModel, self).__init__()
        # Initialize the backbone and head of the model
        self.backbone = backbone
        self.head = head
        self.device = device

    def forward(self, x):
        # Forward pass through the backbone
        x = self.backbone(x)
        # Forward pass through the head
        output = self.head(x)
        return output
    
    def extract_features(self, dataloader: DataLoader) -> torch.Tensor:
        features: list = []
        labels:   list = []
        self.eval()
        with torch.no_grad():
            try:
                for _inputs, _labels in dataloader:
                    _inputs: torch.Tensor = _inputs.to(self.device)
                    _labels: torch.Tensor = _labels.to(self.device)
                    fx = self.backbone(_inputs)
                    features.append(fx.squeeze(-2, -1))
                    labels.append(_labels)
            except ValueError: # 'Too many values to unpack... ' when dataset's return_index is True
                for _inputs, _labels, _ in dataloader:
                    _inputs: torch.Tensor = _inputs.to(self.device)
                    _labels: torch.Tensor = _labels.to(self.device)
                    fx = self.backbone(_inputs)
                    features.append(fx.squeeze(-2, -1))
                    labels.append(_labels)
        return (
            torch.cat(features, dim=0).cpu().numpy(),
            torch.cat(labels, dim=0).cpu().numpy(),
        )

def get_deepcluster_model(architecture, sobel, model_path):
    model = models.get_model(architecture, weights=None)
    if sobel:
        model = replace_first_layer_with_sobel(model)
        # print(model)
    if model_path:
        load_model_weights(model, model_path)
    return model

def replace_first_layer_with_sobel(model):
    sobel_conv = SobelConv()
    model.conv1 = nn.Sequential(
        sobel_conv,
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False))
    return model


def load_model_weights(model, model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint, strict=False)

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # if args.imagenet:
    print("Imagenet normalize")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # else:
    # print("WHOI22 normalize")
    # normalize = transforms.Normalize(mean=[0.7487, 0.7487, 0.7487], std=[0.1738, 0.1738, 0.1738])

    train_dataset = CustomDataset(data_dir=args.data, 
                            transform=transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                normalize
                            ]),
                            num_images_per_class=args.images_per_class)

    # Print total length of the dataset
    print(f'Total length of the dataset: {len(train_dataset)}')

    # Print number of classes
    print(f'Number of classes: {len(train_dataset.classes)}')

    print(f'Load model from {args.model}')

    # Print number of images per class
    if args.verbose:
        class_counts = {}
        for _, label in train_dataset:
            class_name = train_dataset.classes[label]
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1

        for class_name, count in class_counts.items():
            print(f'Number of images in class {class_name}: {count}')

    class_names = train_dataset.classes

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    if 'imagenet' in args.pretrain_method:
        model = models.get_model(args.arch, weights="DEFAULT")

    elif 'deepcluster' in args.pretrain_method:
        model = get_deepcluster_model(args.arch, args.sobel, args.model)

    elif 'dinov1' in args.pretrain_method:
        ########### Load the weights from the torch hub checkpoint ###########
        model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')

        ########### Load the weights from the custom checkpoint ###########
        # model = models.get_model(args.arch, weights=None)
        # num_features = model.fc.in_features
        # checkpoint_key = 'teacher'

        # if os.path.isfile(args.model):
        #     state_dict = torch.load(args.model, map_location="cpu")
        #     if checkpoint_key is not None and checkpoint_key in state_dict:
        #         print(f"Take key {checkpoint_key} in provided checkpoint dict")
        #         state_dict = state_dict[checkpoint_key]
        #     # remove `module.` prefix
        #     state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        #     # remove `backbone.` prefix induced by multicrop wrapper
        #     state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        #     msg = model.load_state_dict(state_dict, strict=False)
        #     print('Pretrained weights found at {} and loaded with msg: {}'.format(args.model, msg))       

    pretrain_methods = ['mocov1', 'mocov2', 'simclr']
    if args.pretrain_method in pretrain_methods:
        model = models.get_model(args.arch, weights=None)
        load_model_weights(model, args.model)

    method_label=args.pretrain_method

    # Modify the backbone as needed, e.g., remove the fully connected layer
    backbone = nn.Sequential(*list(model.children())[:-1])

    if args.verbose:
        print(backbone)

    head = None #nn.Linear(num_features, len(train_dataset.classes))  # num_classes is the number of output classes

    # Initialize the model
    new_model = MyModel(backbone, head, device)

    new_model.to(device)

    # Extract features
    train_features, train_labels = new_model.extract_features(train_dataloader)
    if args.verbose:
        print(train_features.shape)
        print(train_labels.shape)

    # Create folders if not present
    if not os.path.exists(f'features/{args.dataset_name}/train/'):
            os.makedirs(f'features/{args.dataset_name}/train/')

    if not os.path.exists(f'features/{args.dataset_name}/test/'):
            os.makedirs(f'features/{args.dataset_name}/test/')

    if not os.path.exists(f'labels/{args.dataset_name}/train/'):
            os.makedirs(f'labels/{args.dataset_name}/train/')

    if not os.path.exists(f'labels/{args.dataset_name}/test/'):
            os.makedirs(f'labels/{args.dataset_name}/test/')

    # Save the train features and labels
    np.save(f'features/{args.dataset_name}/train/features_{args.arch}_{args.dataset_name}_{method_label}.npy', train_features)
    np.save(f'labels/{args.dataset_name}/train/labels_{args.arch}_{args.dataset_name}_{method_label}.npy', train_labels)

    test_dataset = CustomDataset(data_dir=args.test_data, 
                            transform=transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                normalize
                            ]))

    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Extract features
    test_features, test_labels = new_model.extract_features(test_dataloader)
    if args.verbose:
        print(test_features.shape)
        print(test_labels.shape)

    # Save the test features and labels
    np.save(f'features/{args.dataset_name}/test/features_{args.arch}_{args.dataset_name}_{method_label}.npy', test_features)
    np.save(f'labels/{args.dataset_name}/test/labels_{args.arch}_{args.dataset_name}_{method_label}.npy', test_labels)

if __name__ == "__main__":
    args = parse_args()
    main(args)