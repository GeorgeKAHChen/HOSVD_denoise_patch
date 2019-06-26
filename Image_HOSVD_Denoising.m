function [resultImage,PSNR] = Image_HOSVD_Denoising(noiseImage, originalImage, sigma, betta, gamma, patchSize, patchStackSize, iterationCount, patch_method)

if nargin < 8
    iterationCount = 10;
end
if nargin < 7
    patchStackSize = 35;
end
if nargin < 6
    patchSize = 8;
end
if nargin < 5
    gamma = 0.50;
end
if nargin < 4
    betta = 0.28;
end
if nargin < 3
    resultImage = noiseImage;
    PSNR = -1;
    fprintf('At least need to define noiseImage, originalImage and sigma.\n');
    return;
end

par.sigma = sigma;
par.patchSize = patchSize;
par.patchStackSize = patchStackSize;
par.betta = betta;
par.gamma = gamma;
par.iterationCount = iterationCount;
par.step = min(6, par.patchSize-1);
par.originalImage = originalImage;
par.noiseImage = noiseImage;
par.patch_method = patch_method;

[resultImage,PSNR] = HOSVD_Denoising(par);
