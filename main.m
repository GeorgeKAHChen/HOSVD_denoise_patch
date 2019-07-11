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
para_sigma       = 50
if para_sigma == 50
    para_betta       = 0.1
    para_gamma       = 0.35
    para_patch_size  = 9
    para_patch_stack = 35
elseif para_sigma == 30
    para_betta       = 0.1
    para_gamma       = 0.3
    para_patch_size  = 7
    para_patch_stack = 25
elseif para_sigma == 10
    para_betta       = 0.16
    para_gamma       = 0.28
    para_patch_size  = 7
    para_patch_stack = 25
end
%para_gamma       = 0.67;

para_iteration   = 100;
test_switch      = 0
patch_method     = 31
if patch_method == 34
    para_patch_stack = 100
end
if patch_method == 35
    para_patch_stack = 20
end
find_parameter   = 0
%Method list
%patch_method = 1: Original method NNM patch search 
%patch_method = 2x: Pre-trained Gaussian Mixture Model method
%patch_method = 3x: Pre-trained GMM and K-means method
%patch_method = x1: BFS after classification search
%patch_method = x2: None search after classification
%patch_method = 33: None search after classification
%========================================================
%Read Initial File
%img = double(imread('figure/Barbara.png'));
img = double(imread('figure/House.png'));
%img = double(imread('figure/Lena.png'));
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
if find_parameter
    betta_arr = (0.14: 0.02: 0.3);
    gamma_arr = (0.2: 0.02: 0.3);
    para_ops  = zeros(length(betta_arr), length(gamma_arr));

    for betta_iter = 1: length(betta_arr)
        for gamma_iter = 1: length(gamma_arr)
        
            new_img = zeros(size(image_with_noise));
            for i = 1: size(img, 1)
                for j = 1: size(img, 2)
                    new_img(i, j) = image_with_noise(i, j);
                end
            end

            para_betta = betta_arr(betta_iter);
            para_gamma = gamma_arr(gamma_iter);
            [result_img, PSNR] = Image_HOSVD_Denoising(255 * new_img, 255 * img, para_sigma, para_betta, para_gamma, para_patch_size, para_patch_stack, para_iteration, patch_method);
            para_betta
            para_gamma
            PSNR
            para_ops(betta_iter, gamma_iter) = PSNR;
        end
    end	
else
    image_with_noise   = imresize(image_with_noise, size(image_with_noise) * 2);
    size(img)
    size(image_with_noise)
    [result_img, PSNR] = Image_HOSVD_Denoising(255 * image_with_noise, 255 * img, para_sigma, para_betta, para_gamma, para_patch_size, para_patch_stack, para_iteration, patch_method);
end

%========================================================
%Show Result
if test_switch == 1
    result_img = result_img / 255;
    figure
    imshow(img)
    figure
    imshow(image_with_noise)
    figure
    imshow(result_img)
    hold on
end
%fprintf('PSNR: %d\n', PSNR);
