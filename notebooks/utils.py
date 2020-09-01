#!/usr/bin/env python3
# -*- coding: utf-8 -*-


##########################################
# PROGRAMMER: Pierre-Antoine Ksinant     #
# DATE CREATED: 26/07/2020               #
# REVISED DATE: -                        #
# PURPOSE: General toolbox for the study #
##########################################


##################
# Needed imports #
##################

import cv2, detectron2, json, base64
import numpy as np
import pandas as pd
import imgaug as ia
import imgaug.augmenters as iaa
import imgaug.augmentables.segmaps as iaas
from os import listdir
from io import BytesIO
from sys import maxsize
from PIL import Image
from detectron2.structures import BoxMode
from scipy.ndimage import label, generate_binary_structure


#######################################
# Get folder's images characteristics #
#######################################

def get_folder_images_characteristics(folder_path):
    """ Get folder's images characteristics """
    
    # Get images list:
    images_list = [x for x in listdir(folder_path) if x.endswith('.jpg')]
    
    # Get number of images:
    nb_images = len(images_list)
    
    # Determine variability of folder's images size:
    width_min, width_max = maxsize, -maxsize
    height_min, height_max = maxsize, -maxsize
    
    for img in images_list:
        image = Image.open(folder_path + img)
        width, height = image.size
        if width < width_min:
            width_min = width
        if width > width_max:
            width_max = width
        if height < height_min:
            height_min = height
        if height > height_max:
            height_max = height
            
    # Return results:
    return nb_images, width_min, width_max, height_min, height_max


######################################################
# Transform run-length encoding string to mask array #
######################################################

def rle_to_mask(rle, width=1600, height=256):
    """ Transform run-length encoding string to mask array """
    
    # Get all rle elements from rle string:
    rle_list = rle.split()
    
    # Convert all elements into integers:
    rle_integers = [int(x) for x in rle_list]
    
    # Create pairs in previous list:
    rle_pairs = np.array(rle_integers).reshape(-1, 2)
    
    # Initialize mask array:
    mask_array = np.zeros(width*height, dtype=np.uint8)
    
    # Populate mask array:
    for index, length in rle_pairs:
        index -= 1
        mask_array[index:index+length] = 1
    
    # Reshape mask array:
    mask = mask_array.reshape(width, height).T
    
    # Return result:
    return mask


######################################################
# Transform mask array to run-length encoding string #
######################################################

def mask_to_rle(mask):
    """ Transform mask array to run-length encoding string """
    
    # Flatten mask:
    flat_mask = mask.T.flatten()
    flat_mask = np.concatenate([[0], flat_mask, [0]])
    
    # Work on flattened mask:
    working_mask = np.where(flat_mask[1:] != flat_mask[:-1])[0] + 1
    working_mask[1::2] -= working_mask[::2]
    
    # Constitute rle string:
    rle = ' '.join(str(x) for x in working_mask)
    
    # Return result:
    return rle


###############################################
# Calculate number of defects in a mask array #
###############################################

def calculate_nb_defects_in_mask(train_df):
    """ Calculate number of defects in a mask array """
    
    # Define a defect type binary structure:
    defect_type_structure = generate_binary_structure(2, 2)
    
    # Initialize number of defects column:
    enhanced_train_df = train_df.copy()
    enhanced_train_df['NbDefects'] = 0
    
    # Go through dataframe:
    for idx, row in enhanced_train_df.iterrows():
        mask = rle_to_mask(row['EncodedPixels'])
        if np.sum(mask) > 0:
            labeled_mask, nb_defects = label(mask, structure=defect_type_structure)
            enhanced_train_df['NbDefects'].loc[idx] = nb_defects
    
    # Return result:
    return enhanced_train_df


############################
# Basic image augmentation #
############################

def basic_image_augmentation(image_name, image_dataframe):
    """ Basic image augmentation """
    
    # Basic augmentation sequence:
    seq = iaa.Sequential([iaa.Crop(px=(0, 16)),
                          iaa.Fliplr(0.5),
                          iaa.GaussianBlur(sigma=(0, 3.0)),
                          iaa.Rotate((-45, 45))])
    
    # Open image:
    image_path = '../data/train_images/' + image_name
    image = cv2.imread(image_path)
    
    # Get image's mask:
    image_rle = image_dataframe[(image_dataframe['ImageId']==image_name)]['EncodedPixels'].values[0]
    image_mask = rle_to_mask(image_rle)
    
    # Create segmentation map:
    segmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)
    segmap = np.where(image_mask==1, 1, segmap)
    segmap = iaas.SegmentationMapsOnImage(segmap, shape=image.shape)
    
    # Apply basic augmentation sequence:
    image_aug, segmap_aug = seq(image=image, segmentation_maps=segmap)
    
    # Extract mask:
    image_aug_mask = segmap_aug.get_arr()
    
    # Get rle:
    image_aug_rle = mask_to_rle(image_aug_mask)
    
    # Return results:
    return image_aug, image_aug_mask, image_aug_rle


##############################################
# Custom function to register custom dataset #
##############################################

def get_severstal_dicts(dataset_path):
    """ Get Severstal dictionaries """
    
    # Open CSV description file:
    dataset_file_path = dataset_path + 'annotations.csv'
    dataset = pd.read_csv(dataset_file_path)
    
    # Initialize list of dictionaries for dataset:
    dataset_dicts = []
    
    # Go through dataset:
    for idx, row in dataset.iterrows():
        # Initialize record:
        record = {}
        # Populate first elements of record:
        file_name = dataset_path + row['ImageId']
        width, height = Image.open(file_name).size
        record['file_name'] = file_name
        record['height'] = height
        record['width'] = width
        record['image_id'] = idx
        # Initialize last element of record:
        annotations = []
        # Get mask:
        rle = row['EncodedPixels']
        mask = rle_to_mask(rle)
        # Determine contours:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Populate last element of record:
        for contour in contours:
            contour_array = contour.reshape(-1, 2)
            px = [i[0] for i in contour_array]
            py = [j[1] for j in contour_array]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]
            x0, y0 = int(np.min(px)), int(np.min(py))
            x1, y1 = int(np.max(px)), int(np.max(py))
            if (len(poly)%2)==0 and len(poly)>=6:
                annotation = {'bbox': [x0, y0, x1, y1],
                              'bbox_mode': BoxMode.XYXY_ABS,
                              'segmentation': [poly],
                              'category_id': row['ClassId'] - 1,
                              'iscrowd': 0}
                annotations.append(annotation)
        record['annotations'] = annotations
        # Add to list of dictionaries:
        dataset_dicts.append(record)
    
    # Return results:
    return dataset_dicts


######################################
# Generate input json for TorchServe #
######################################

def gen_input_json_ts(img_path, json_path):
    """ Generate input json for TorchServe """
    
    # Open image and get filename extension:
    img = Image.open(img_path)
    
    # Construct input json file:
    img_bio = BytesIO()
    img.save(img_bio, format='JPEG')
    img_bytes = img_bio.getvalue()
    img_string = base64.b64encode(img_bytes).decode('utf-8')
    img_b64 = 'data:image/jpg;base64,' + img_string
    input_dict = {'data': img_b64}
    
    # Dump input json file:
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(input_dict, f, ensure_ascii=False, indent=4)
        
        
################################################
# Generate image from TorchServe json response #
################################################

def gen_img_json_ts(json_path, img_path):
    """ Generate image from TorchServe json response """
    
    # Open json file:
    with open(json_path, 'r') as f:
        json_dict = json.load(f)
        
    # Construct image:
    json_data = json_dict['data']
    img_b64 = json_data.split(',', 1)[1]
    img_bio = BytesIO(base64.b64decode(img_b64))
    img = Image.open(img_bio)
    
    # Save image:
    img.save(img_path)