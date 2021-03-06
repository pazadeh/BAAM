"""
shuffleing the augmentated data to create train, validation and test data set
"""

import numpy as np
import h5py
from tqdm import tqdm
from random import shuffle



images_path        = 'augmentated dataset location'
ip_augmented_path  = images_path+'dataset_projected_augmented.hdf5'
f_or               = h5py.File(ip_augmented_path,'r')
keys = f_or.keys()
dset_images = f_or[keys[1]]
dset_indexes= f_or[keys[0]]
shuffle_data = True
num_classes = 4


image_size_y= 733
image_size_z= 161
# read addresses and labels from the 'train' folder with a hot vector
image_order = np.arange(dset_images.shape[0])
if shuffle_data:
    c = list(zip(image_order))
    shuffle(c)
    image_order = zip(*c)

image_order =np.asarray(image_order)
# Divide the hata into 80% train, 10% validation, and 10% test 
train_order = image_order[0][0:int(0.80*image_order.shape[1])]
valid_order = image_order[0][int(0.80*image_order.shape[1]):int(0.90*image_order.shape[1])]
test_order  = image_order[0][int(0.90*image_order.shape[1]):]

# check the order of data and chose proper data shape to save images
train_shape = (len(train_order), image_size_y*image_size_z)
val_shape   = (len(valid_order), image_size_y*image_size_z)
test_shape  = (len(test_order),  image_size_y*image_size_z)

#Setting the path to shuffled and divided files
shuffled_images_path ='location of train, test, validation datasets'
bone_cr10p_top_train_set_path = shuffled_images_path+'bone_projected_train_set.hdf5'  
bone_cr10p_top_test_set_path  = shuffled_images_path+'bone_projected_test_set.hdf5'  
bone_cr10p_top_valid_set_path = shuffled_images_path+'bone_projected_valid_set.hdf5'  

# open three files and create earrays
bone_cr10p_top_train_set = h5py.File(bone_cr10p_top_train_set_path, mode='w')
bone_cr10p_top_test_set  = h5py.File(bone_cr10p_top_test_set_path , mode='w')
bone_cr10p_top_valid_set = h5py.File(bone_cr10p_top_valid_set_path, mode='w')

bone_cr10p_top_train_set.create_dataset("data1", train_shape,np.int32)
bone_cr10p_top_train_set.create_dataset("Index1", (len(train_order),num_classes), np.float32)
bone_cr10p_top_test_set.create_dataset("data1", test_shape, np.int32)
bone_cr10p_top_valid_set.create_dataset("Index1", (len(valid_order),num_classes), np.float32)
bone_cr10p_top_valid_set.create_dataset("data1", val_shape, np.int32)
bone_cr10p_top_test_set.create_dataset("Index1", (len(test_order),num_classes), np.float32)

counter =0 
pbar = tqdm(total=image_order.shape[1])
for i in train_order:
    img   = dset_images[i]
    label = dset_indexes[i]
    # add any image pre-processing here
    img = img.flatten()
    bone_cr10p_top_train_set["data1"][counter, ...] = img[None]
    bone_cr10p_top_train_set["Index1"][counter, ...] = label
    counter = counter+1
    pbar.update(1)
    
counter =0 
   
for i in valid_order:
    img   = dset_images[i]
    label = dset_indexes[i]
    # add any image pre-processing here
    img = img.flatten()
    bone_cr10p_top_valid_set["data1"][counter, ...] = img[None]
    bone_cr10p_top_valid_set["Index1"][counter, ...] = label
    counter = counter+1
    pbar.update(1)
counter =0 


f_or.close()
bone_cr10p_top_train_set.close()
bone_cr10p_top_valid_set.close()
bone_cr10p_top_test_set.close()
