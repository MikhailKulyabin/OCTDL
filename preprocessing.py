import os
from os import listdir
from os.path import isfile
from os.path import join
from pathlib import Path
import cv2
from math import ceil
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_folder', type=str, default='./OCTDL_dataset', help='path to dataset folder')
parser.add_argument('--labels_path', type=str, default='./OCTDL_dataset/labels.csv', help='path to labels.csv')
parser.add_argument('--output_folder', type=str, default='./dataset_1', help='path to output folder')
parser.add_argument('--crop_ratio', type=int, default=1, help='central crop ratio of image')
parser.add_argument('--image_dim', type=int, default=512, help='final dimensions of image')
parser.add_argument('--val_ratio', type=float, default=0.15, help='validation size')
parser.add_argument('--test_ratio', type=float, default=0.25, help='test size')
parser.add_argument('--padding', type=bool, default=False, help='padding to square')
parser.add_argument('--crop', type=bool, default=False, help='crop')
parser.add_argument('--resize', type=bool, default=False, help='resize')

labels = ['AMD', 'DME', 'ERM', 'NO', 'RAO', 'RVO', 'VID']
folders = ['train', 'val', 'test']


def main():
    args = parser.parse_args()
    root_folder = Path(args.dataset_folder)
    output_folder = Path(args.output_folder)
    val_ratio = args.val_ratio
    test_ratio = args.test_ratio
    train_ratio = 1 - val_ratio - test_ratio
    dim = (args.image_dim, args.image_dim)
    crop_ratio = args.crop_ratio
    padding_bool = args.padding
    crop_bool = args.crop
    resize_bool = args.resize
    labels_path = args.labels_path
    
    df = pd.read_csv(labels_path)
    
    for folder in folders:
        for label in labels:
            Path(os.path.join(output_folder, folder, label)).mkdir(parents=True, exist_ok=True)
    
    for label in tqdm(labels):
        df_label = df[df['disease'] == label][['file_name', 'disease', 'patient_id']]
        patients_list = df_label.patient_id.unique()
        train_patients, test_patients = train_test_split(patients_list, test_size=1 - train_ratio)
        val_patients, test_patients = train_test_split(test_patients, test_size=test_ratio / (test_ratio + val_ratio))
        df_label_train = df_label[df_label['patient_id'].isin(train_patients)]
        df_label_val = df_label[df_label['patient_id'].isin(val_patients)]
        df_label_test = df_label[df_label['patient_id'].isin(test_patients)]
        
        print(label, len(df_label_train), len(df_label_val), len(df_label_test))
    
        for i in range(0, len(df_label_train)):
            file_name = df_label_train.iloc[i, 0] + '.jpg'
            file_label = df_label_train.iloc[i, 1]    
            preprocessing(root_folder, output_folder, file_name, 'train', crop_ratio, dim, file_label, padding_bool, crop_bool, resize_bool)
 
        for i in range(0, len(df_label_test)):
            file_name = df_label_test.iloc[i, 0] + '.jpg'
            file_label = df_label_test.iloc[i, 1]    
            preprocessing(root_folder, output_folder, file_name, 'test', crop_ratio, dim, file_label, padding_bool, crop_bool, resize_bool)    
    
        for i in range(0, len(df_label_val)):
            file_name = df_label_val.iloc[i, 0] + '.jpg'
            file_label = df_label_val.iloc[i, 1]    
            preprocessing(root_folder, output_folder, file_name, 'val', crop_ratio, dim, file_label, padding_bool, crop_bool, resize_bool) 


def preprocessing(root_folder, output_folder, file, folder, crop_ratio, dim, label, padding_bool, crop_bool, resize_bool):
    img = cv2.imread(os.path.join(root_folder, label, file))
    if padding_bool:
        img = padding(img)
    if crop_bool:
        img = center_crop(img, (img.shape[1] * crop_ratio, img.shape[0] * crop_ratio))
    if resize_bool:
        img = cv2.resize(img, dim, interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(os.path.join(output_folder, folder, label, Path(file).name), img, [cv2.IMWRITE_JPEG_QUALITY, 100])


def padding(img):
    """Returns padded to square image
    Args:
    img: image to be center cropped
    """
    height = img.shape[0]
    width = img.shape[1]
    if width == height:
        return img
    elif width > height:
        left = 0
        right = 0
        bottom = ceil((width - height) / 2)
        top = ceil((width - height) / 2)
        result = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return result
    else:
        left = ceil((height - width) / 2)
        right = ceil((height - width) / 2)
        bottom = 0
        top = 0
        result = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return result


def center_crop(img, dim):
    """Returns center cropped image
    Args:
    img: image to be center cropped
    dim: dimensions (width, height) to be cropped
    """
    width, height = img.shape[1], img.shape[0]
    # process crop width and height for max available dimension
    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2)
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return crop_img


def scale_image(img, factor=1):
    """Returns resize image by scale factor.
    This helps to retain resolution ratio while resizing.
    Args:
    img: image to be scaled
    factor: scale factor to resize"
    """
    width = int(img.shape[1] * factor)
    height = int(img.shape[0] * factor)
    dsize = (width, height)
    output = cv2.resize(img, dsize, interpolation=cv2.INTER_LANCZOS4)
    return output


if __name__ == "__main__":
    main()
