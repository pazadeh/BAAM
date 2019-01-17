# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 14:29:18 2018

@author: asgharpn
"""

from imgaug import augmenters as iaa
import numpy as np
import h5py
import scipy.io as sio
import glob
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import sys




image_path = '/media/asgharpn/daten2017-03/Bone_Machine_learning/Learning_dataset/projected_not_squared_yz_test_01_without_2mice_04/*.mat'  
num_rotations    = 15
trans_percent    = float(2)
rot_angle        = 1
num_translations = 11
trans_percent    = float(2)
num_classes      = 4

if num_rotations % 2 == 0:
    sys.exit("The number of rotations should be an odd number to prevenet the translation with 0 percent")
if num_translations % 2 == 0:
    sys.exit("The number of translation should be an odd number to prevenet the translation with 0 percent")


addrs = glob.glob(image_path)

train_addrs = addrs


img = sio.loadmat(train_addrs[4])
img = img["slice_new"]


# Creating the hdf5 files
images_path = '/media/asgharpn/daten2017-03/Bone_Machine_learning/Learning_dataset/projected_augmented_not_squared_yz_test_01_without_2mice_04/'
ip_path =images_path +'dataset_prjected.hdf5'  
ip_rotated_path = images_path+'dataset_prjected_rotated.hdf5'  
ip_rotated_flipped_h_path  = images_path+'dataset_prjected_rotated_flipped_h.hdf5'  
ip_rotated_flipped_v1_path = images_path+'dataset_prjected_rotated_flipped_v1.hdf5'  
ip_rotated_flipped_v2_path = images_path+'dataset_prjected_rotated_flipped_v2.hdf5'  
ip_rotated_translated_path = images_path+'dataset_prjected_rotated_translated.hdf5'  

ip_augmented_path          = images_path+'dataset_projected_augmented.hdf5'


images_names =  [None]*len(train_addrs)
images = np.zeros([len(addrs),img.shape[0],img.shape[1]],dtype=np.int32)

ip_shape = (len(addrs),img.shape[0],img.shape[1])
ip = h5py.File(ip_path, mode='w')
ip.create_dataset("images", ip_shape,np.int32)
labels = [[1,0,0,0] if 'day0' in addr else 
         [0,1,0,0] if 'day5'  in addr else  
         [0,0,1,0] if 'day10' in addr else  
         [0,0,0,1] if 'day15' in addr else 'bad naming of images'
         for addr in addrs]
for i in range(len(train_addrs)):
    # print how many images are saved every 1000 images
    img = sio.loadmat(train_addrs[i])
    img = img["slice_new"]
    ip["images"][i, ...] = img[None]
    images_names[i] = train_addrs[i]
ip.create_dataset("Index1", (len(addrs),num_classes), np.float32)
ip["Index1"][...] = labels 

num_rotations = num_rotations
num_translations = num_translations-1
counter_first = 0
lim_counter_first = ((num_rotations)*2*(1+num_translations))
print("Augmenting %d images \n" % (lim_counter_first*len(addrs)))
pbar = tqdm(total=lim_counter_first)


# Rotations in increments of rot_angle degree
ip_rotated_shape = (num_rotations*2,len(addrs),img.shape[0],img.shape[1])
ip_rotated = h5py.File(ip_rotated_path, mode='w')
ip_rotated.create_dataset("images", ip_rotated_shape,np.int32)        
ip_rotated.create_dataset("Index1", (ip_rotated_shape[0],len(addrs),num_classes), np.float32)
j=0
for i in range(-int(num_rotations/rot_angle),int(math.ceil(num_rotations/rot_angle))+1):
    if (i!=0):
        aug_rotate =  iaa.Sequential([
                      iaa.Affine(rotate=i*rot_angle),]) 
        images_aug = aug_rotate.augment_images(ip['images'][...])
        ip_rotated["images"][j, ...] = images_aug[None]
        ip_rotated["Index1"][j, ...] = ip["Index1"]
        j=j+1
        counter_first = counter_first+1
        pbar.update(1)
print('Rotation finished')        
#Horizontal_flip   

#ip_rotated_flipped_h_shape = (num_rotations,len(addrs),img.shape[0],img.shape[1])
#ip_rotated_flipped_h = h5py.File(ip_rotated_flipped_h_path, mode='w')
#ip_rotated_flipped_h.create_dataset("images", ip_rotated_flipped_h_shape,np.int32)
#ip_rotated_flipped_h.create_dataset("Index1", (ip_rotated_flipped_h_shape[0],len(addrs),num_classes), np.float32)

#j=0
#aug_flip_h = iaa.Sequential([iaa.Fliplr(0.9)])
#for i in range(num_rotations):
#        images_aug = aug_flip_h.augment_images(ip_rotated["images"][j])
#        ip_rotated_flipped_h["images"][i, ...] = images_aug[None]
#        ip_rotated_flipped_h["Index1"][i, ...] = ip_rotated["Index1"][i, ...]
#        j=j+1
#        counter_first = counter_first+1
#        pbar.update(1)
#print('Horizotal flipping finished') 
##flipped_vertical_1
#
#ip_rotated_flipped_v1_shape = (num_rotations,len(addrs),img.shape[0],img.shape[1])
#ip_rotated_flipped_v1 = h5py.File(ip_rotated_flipped_v1_path, mode='w')
#ip_rotated_flipped_v1.create_dataset("images", ip_rotated_flipped_v1_shape,np.int32)
#ip_rotated_flipped_v1.create_dataset("Index1", (ip_rotated_flipped_v1_shape[0],len(addrs),num_classes), np.float32)
#j=0
#aug_flip_h = iaa.Sequential([iaa.Flipud(0.9)])
#for i in range(num_rotations):
#        images_aug = aug_flip_h.augment_images(ip_rotated["images"][i])
#        ip_rotated_flipped_v1["images"][i, ...] = images_aug[None]
#        ip_rotated_flipped_v1["Index1"][i, ...] = ip_rotated["Index1"][i, ...]
#        j=j+1
#        counter_first = counter_first+1
#        pbar.update(1)
#print('Vertical flipping 1 finished') 
##flipped_vertical_2
#
#ip_rotated_flipped_v2_shape = (num_rotations,len(addrs),img.shape[0],img.shape[1])
#ip_rotated_flipped_v2 = h5py.File(ip_rotated_flipped_v2_path, mode='w')
#ip_rotated_flipped_v2.create_dataset("images", ip_rotated_flipped_v2_shape,np.int32)
#ip_rotated_flipped_v2.create_dataset("Index1", (ip_rotated_flipped_v2_shape[0],len(addrs),num_classes), np.float32)
#j=0
#aug_flip_h = iaa.Sequential([iaa.Flipud(0.9)])
#for i in range(num_rotations):
#        images_aug = aug_flip_h.augment_images(ip_rotated_flipped_h["images"][i])
#        ip_rotated_flipped_v2["images"][i, ...] = images_aug[None]
#        ip_rotated_flipped_v1["Index1"][i, ...] = ip_rotated["Index1"][i, ...]
#        j=j+1
#        counter_first = counter_first+1
#        pbar.update(1)
#print('Vertical flipping 2 finished') 
# translation in increments of 2 percent

ip_rotated_translated_shape = (num_translations,num_rotations*2,len(addrs),img.shape[0],img.shape[1])
ip_rotated_translated = h5py.File(ip_rotated_translated_path, mode='w')
ip_rotated_translated.create_dataset("images", ip_rotated_translated_shape,np.int32)
ip_rotated_translated.create_dataset("Index1", (ip_rotated_translated_shape[0],ip_rotated_translated_shape[1],len(addrs),num_classes), np.float32)

j=0

for k in range(-int(num_translations/2),int(math.ceil(num_translations/2))+1):   
    if (k != 0):    
        aug_translate = iaa.Affine(translate_percent={"x": (-(trans_percent/100)*k, (trans_percent/100)*k), "y": (-(trans_percent/100)*k, (trans_percent/100)*k)})
        for i in range(num_rotations*2):
            images_aug = aug_translate.augment_images(ip_rotated["images"][i])
            ip_rotated_translated["images"][j,i, ...] = images_aug[None]
            ip_rotated_translated["Index1"][j,i, ...] = ip_rotated["Index1"][i, ...]
            counter_first = counter_first+1
            pbar.update(1)
        j=j+1

print('Translation finished')         



ip_augmented_shape = (((num_rotations)*2*(1+num_translations)*len(addrs)),img.shape[0],img.shape[1])
ip_augmented = h5py.File(ip_augmented_path, mode='w')
ip_augmented.create_dataset("data1", ip_augmented_shape,np.int32)
ip_augmented.create_dataset("Index1", (ip_augmented_shape[0],num_classes), np.float32)
ip["Index1"][...] = labels 

print("Number of images after augmentation: \n")
print ip_augmented_shape[0]

print("Creating the data set \n")
counter = 0
pbar = tqdm(total=ip_augmented_shape[0])
for i in range(ip_rotated_shape[0]):
    for j in range(ip_rotated_shape[1]):
        ip_augmented['data1'][counter] = ip_rotated['images'][i][j]
        ip_augmented['Index1'][counter]= ip_rotated['Index1'][i][j]
        counter = counter+1
        pbar.update(1)

    
#for i in range(ip_rotated_flipped_h_shape[0]):
#    for j in range(ip_rotated_flipped_h_shape[1]):
#        ip_augmented['data1'][counter] = ip_rotated_flipped_h['images'][i][j]
#        ip_augmented['Index1'][counter]= ip_rotated_flipped_h['Index1'][i][j]
#        counter = counter+1
#        pbar.update(1)
#        
#for i in range(ip_rotated_flipped_v1_shape[0]):
#    for j in range(ip_rotated_flipped_v1_shape[1]):
#        ip_augmented['data1'][counter] = ip_rotated_flipped_v1['images'][i][j]
#        ip_augmented['Index1'][counter]= ip_rotated_flipped_v1['Index1'][i][j]
#        counter = counter+1
#        pbar.update(1)
#        
#for i in range(ip_rotated_flipped_v2_shape[0]):
#    for j in range(ip_rotated_flipped_v2_shape[1]):
#        ip_augmented['data1'][counter] = ip_rotated_flipped_v2['images'][i][j]
#        ip_augmented['Index1'][counter]= ip_rotated_flipped_v2['Index1'][i][j]     
#        counter = counter+1
#        pbar.update(1)
print("first finishe \n")        
for i in range(ip_rotated_translated_shape[0]):
    for j in range(ip_rotated_translated_shape[1]):
        for k in range(ip_rotated_translated_shape[2]):
            ip_augmented['data1'][counter] = ip_rotated_translated['images'][i][j][k]
            ip_augmented['Index1'][counter]= ip_rotated_translated['Index1'][i][j][k]
            counter = counter+1
            pbar.update(1)
                
plt.imshow(ip_augmented["data1"][ip_augmented_shape[0]-1])

# Checkin if all images are actually filled and not zero
dset = ip_augmented["data1"]
for i in range(ip_augmented_shape[0]-1):
    image = dset[i]
    all_zero = not np.any(image)
    if all_zero:
        print i
        print('There is an empty image in the dataset')
        sys.exit("Error: There is an empty image in the dataset!")   
print('Dataset is created and there is no empty images') 

   
ip.close()
ip_rotated.close()
#ip_rotated_flipped_h.close()
#ip_rotated_flipped_v1.close()  
#ip_rotated_flipped_v2.close()
ip_rotated_translated.close()

ip_augmented.close()         