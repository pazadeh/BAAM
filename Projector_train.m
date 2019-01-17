%% Projector
%  This code makes the image a square by its biggest dimension and also
%  projects all the slices into one by adding up all the slices of the
%  image

% It is important to know that this codes projects everything Y-Z plane and
% if we want projection on other planes this has to be changed.

clear 
clc
resmapled_images_path = '/media/asgharpn/daten2017-03/Bone_Machine_learning/Learning_dataset/resampled_borders_test_01';
projected_images_path_yz = '/media/asgharpn/daten2017-03/Bone_Machine_learning/Learning_dataset/projected_not_squared_yz_test_01';
projected_images_path_xz = '/media/asgharpn/daten2017-03/Bone_Machine_learning/Learning_dataset/projected_not_squared_xz_test_01';

cd(resmapled_images_path)
files = dir('*.mat');
addpath('/media/asgharpn/daten2017-03/Bone_Machine_learning/Functions/');

max_x_dim = 602;
max_y_dim = 733;
max_z_dim = 161;

for i=1:length(files)
    cd(resmapled_images_path)
    file = load(files(i).name);
    image = file.image_c;
    size_image =size(image);
    number_x_stacks = size(image,1);
    image = double(image);
    cd(projected_images_path_yz)
    slice = image(1,:,:);
    size_slice = size(image);
    slice_new = zeros(max_y_dim,size(image,3));
    number_z_stacks = size(image,3);
    for j=1:number_x_stacks
        slice = squeeze(image(j,:,:));
        slice_new(1:max_y_dim,1:size_slice(3)) = slice_new(1:max_y_dim,1:size_slice(3)) + slice;
    end
    
    save(files(i).name,'slice_new','-mat'); 
    slice_check = zeros(602,size(image,3));

    cd(projected_images_path_xz)
    number_y_stacks = size(image,2);
    for j=1:number_y_stacks
        slice = squeeze(image(:,j,:));
        slice_check(1:size_slice(1),1:size_slice(3)) = slice_check(1:size_slice(1),1:size_slice(3)) + slice;
    end
    save(files(i).name,'slice_check','-mat'); 
    i/length(files)    
    
end
