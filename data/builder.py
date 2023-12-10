import os
from torchvision import datasets
from data.dataset import img_loader
from data.transforms import data_transforms, simple_transform
from data.dataset import CustomizedImageFolder
from utils.func import mean_and_std, print_dataset_info


def generate_dataset(cfg):
    if cfg.data.mean == 'auto' or cfg.data.std == 'auto':
        mean, std = auto_statistics(
            cfg.base.data_path,
            cfg.data.input_size,
            cfg.train.batch_size,
            cfg.train.num_workers
        )
        cfg.data.mean = mean
        cfg.data.std = std

    train_transform, test_transform = data_transforms(cfg)

    data_splits = generate_dataset_from_folder(
        cfg.base.data_path,
        train_transform,
        test_transform
    )

    print_dataset_info(data_splits)
    return data_splits


def auto_statistics(data_path, input_size, batch_size, num_workers):
    print('Calculating mean and std of training set for data normalization.')
    transform = simple_transform(input_size)
    train_path = os.path.join(data_path, 'train')
    train_dataset = datasets.ImageFolder(train_path, transform=transform)

    return mean_and_std(train_dataset, batch_size, num_workers)


def generate_dataset_from_folder(data_path, train_transform, test_transform):
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')
    val_path = os.path.join(data_path, 'val')

    train_dataset = CustomizedImageFolder(train_path, train_transform, loader=img_loader)
    test_dataset = CustomizedImageFolder(test_path, test_transform, loader=img_loader)
    val_dataset = CustomizedImageFolder(val_path, test_transform, loader=img_loader)

    return train_dataset, test_dataset, val_dataset
