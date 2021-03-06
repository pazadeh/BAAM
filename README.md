# BAAM

# Decoding the Rejuvenating Effects of Mechanical Loading on Skeletal Maturation using in Vivo Imaging and Deep Learning

For a detailed clarification of the method and results of this project, please take a look at our paper available at:

https://doi.org/10.1016/j.actbio.2020.02.007

Bone Age Assesment Model (BAAM) :Predicting bone age based on 3D µCT images of bone using a deep neural netwrork. 
The codes are tailored for data used in BAAM paper but could be easily used for any classification puposase of 3D images.

Here you can find the following seris of codes:
1. Image preproccessing (MATLAB):

1.1. Size homogenization: Not all the images have originaly the same size. This code will make them homogenized based on the the smallest dimensions in the dataset. This code is created to run on GPU because of the massivie improvement in calculation time when workin on big 3D images.

1.2. Projector: This code will perform and intensity projection on the images with same size.

2. Data Preparation (Python):

2.1  Data Annotation: Creating hot vectors for images to include their class. Here we extract this from the names of the files.
2.2. Augmentation: For increasing the number of images in training data set using standard augmenatation techniques such as rotation, translation and ...
2.3. Schuffel: For shuffeling the data into train, validation and test datasets and creating HDf5 file format for optimal big structure.
  
3. Read and write libraries (Python): For transforming the data from different dataset into correct format in the process of training including batching and ...
   
4. Training (Python-Tensorflow): For training the BAAM network including the architecture of the network and storing the trained network

5. Evaluating (Python-Tensorflow): For applying the trained network on another dataset.

6. Saliency Map Visualitation (Python-Tensorflow): For Computing and Visualizing saliency maps of the images passing through the network.
