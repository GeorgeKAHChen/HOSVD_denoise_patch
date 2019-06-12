%========================================================
%
%       HOSVD Main
%       BY KazukiAmakawa(with Image Patch Method)
%
%========================================================
clc;
clear;
%========================================================
%Setting Parameter
para_sigma       = 10;
para_betta       = 0.32;
para_gamma       = 0.45;
para_patch_size  = 6;
para_patch_stack = 25;
para_iteration   = 5;
test_switch      = 1;
patch_method     = 0;
%========================================================
%Read Initial File
img = double(imread('aznyan.jpg'));
img = img / 255;
img_size = size(size(img));
if img_size(1, 2) == 3
    img = rgb2gray(img);
end

%========================================================
%Add noise
randn('seed', 0 );
rand ('seed', 0 );
image_with_noise = img + randn(size(img)) * para_sigma / 255; 

%========================================================
%Algorithm
[result_img, PSNR] = Image_HOSVD_Denoising(255 * image_with_noise, 255 * img, para_sigma, para_betta, para_gamma, para_patch_size, para_patch_stack, para_iteration, patch_method);
result_img = result_img / 255;

%========================================================
%Show Result
if test_switch == 1
    figure
    imshow(img)
    %hold on;
    figure
    imshow(image_with_noise)
    %hold on;
    figure
    imshow(result_img)
    hold on
end
%fprintf('PSNR: %d\n', PSNR);
