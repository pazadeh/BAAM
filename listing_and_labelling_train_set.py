# -*- coding: utf-8 -*-
"""
This code is used ONLY for creating the the HDF5 file from the training set
to calculate saliency maps and maximum activations! We do NOT use the created
file for training!!! 
"""


import glob
import scipy.io as sio
import numpy as np
import h5py


shuffle_data = True  # shuffle the addresses before saving

# address to where you want to save the hdf5 file

image_path =      '/media/asgharpn/daten2017-03/Bone_Machine_learning/Learning_dataset/projected_not_squared_yz_test_01_without_2mice_10/*.mat'
HDF5_files_path = '/media/asgharpn/daten2017-03/Bone_Machine_learning/Learning_dataset/projected_not_squared_yz_test_01_without_2mice_10/HDF5_2Mice_10/'
bone_evaluation_projected_set_path = HDF5_files_path+'bone_projected_train_set.hdf5'  

dt = h5py.special_dtype(vlen=bytes)

# Number of classes
num_classes = 4
# read addresses and labels from the 'train' folder with a hot vector
image_names = []
addrs = glob.glob(image_path)
labels = [[1,0,0,0] if 'day0' in addr else 
         [0,1,0,0] if 'day5'  in addr else  
         [0,0,1,0] if 'day10' in addr else  
         [0,0,0,1] if 'day15' in addr else 'bad naming of images'
         for addr in addrs]
             

image_size_y = 733
image_size_z = 161


for addr in addrs:
    image_names.append(addr.replace(
    "/media/asgharpn/daten2017-03/Bone_Machine_learning/Test_dataset/projected_not_squared_yz_test_01_2mice_05/",""))


train_addrs = addrs
train_labels = labels


# check the order of data and chose proper data shape to save images
train_shape = (len(train_addrs), image_size_y*image_size_z)#761,761)

# open three files and create earrays
bone_evaluation_projected_set = h5py.File(bone_evaluation_projected_set_path, mode='w')


bone_evaluation_projected_set.create_dataset("data1", train_shape,np.int32)
bone_evaluation_projected_set.create_dataset("Index1", (len(train_addrs),num_classes), np.float32)
bone_evaluation_projected_set.create_dataset("Image_names1", (len(train_addrs),),dtype=dt)
bone_evaluation_projected_set["Index1"][...] = train_labels
bone_evaluation_projected_set["Image_names1"][...] = image_names

for i in range(len(train_addrs)):
    # print how many images are saved every 1000 images
    if i % 1000 == 0 and i > 1:
        print('Train data: {}/{}'.format(i, len(train_addrs)))
    print i
    addr = train_addrs[i]
    img = sio.loadmat(train_addrs[i])
    img = img["slice_new"]
    # add any image pre-processing here
    img = img.flatten()
    # save the image and calculate the mean so far
    bone_evaluation_projected_set["data1"][i, ...] = img[None]



# close the hdf5 files
bone_evaluation_projected_set.close()
