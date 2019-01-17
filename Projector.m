%% Projector
%  This code projects all the slices into one by adding up all the slices of the
%  image

% It is important to know that this codes projects everything X-Z and Y-Z planes.
% Based on the application you should pick which one you like


clear 
clc
resmapled_images_path = ' location of resampled images';
projected_images_path_yz = ' Location of projected images in YZ direction ';
projected_images_path_xz = '/Location of projected images in XZ direction ';

cd(resmapled_images_path)
files = dir('*.mat');
addpath('Location of functions');

% size base on your images should be changed
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
