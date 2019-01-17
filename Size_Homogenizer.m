%% This code homogenizes the size of the images. 
% This happens based on the give (or found) maximal sizes in X and Y and
% minimal size in the Z direction from the top



clear 
clc
cd('image directory')
files = dir('*.mat');
image_sizes = [];
addpath('where your functions are');


original_images_path  = ' original image directory ';
resampled_images_path = ' resampled image directory ';


% Should be picked based on desired size of images
min_z_dim =161+1;
max_x_dim =602;
max_y_dim =733;

for i=1:length(files)
    % Reading the image
    cd(original_images_path)
    img = load(files(i).name);
    image_c = struct2cell(img(1));
    image_c = image_c{1,1};
    image = gpuArray(image_c);    
    image = double(image);
    
    % Cropping th Z direction
    image(:,:,(min_z_dim:size(image,3))) = [];

    % Finding the stripes of pixels in left and right of the imahe
    x_strip_left  = image(:,1,:);
    x_strip_right = image(:,size(image,2),:);
    % Adding the strips to the image
    for j=1:(fix((max_y_dim-size(image,2))/2))
        image = cat(2,x_strip_left,image);
        image = cat(2,image,x_strip_right);
    end
    
    if size(image,2)<max_y_dim
       image = cat(2,image,x_strip_right); 
    end
     if size(image,2)>max_y_dim
       image(1,:,:)= []; 
     end
    
    % Finding the stripes of pixels on top and bottom of the image
    y_strip_top   = image(1,:,:);
    y_strip_bot   = image(size(image,1),:,:);
    % Adding the strips to the image
    for j=1:(fix((max_x_dim-size(image,1))/2))
        image = cat(1,y_strip_top,image);
        image = cat(1,image,y_strip_bot);
    end
    if size(image,1)<max_x_dim
       image = cat(1,image,y_strip_bot); 
    end
     if size(image,1)>max_x_dim
       image(:,1,:)= []; 
    end

    
    cd(resampled_images_path)
    % Correcting the name of the image
    file_name = strcat(erase(files(i).name,'.mat'),'.mat');
    % Gathering the image and bringing it to int16 format
    image =int16(image);
    image_c = gather(image);
    % Saving the image
    save(file_name,'image_c','-mat');
    i/length(files)

    
end

