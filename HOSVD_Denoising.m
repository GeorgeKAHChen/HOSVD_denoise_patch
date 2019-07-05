function [resultImage, PSNR, SSIM] = HOSVD_Denoising(par)
show_patch = 0;

startTime = clock;
noiseImage = par.noiseImage;
originalImage = par.originalImage;
patchSize = par.patchSize;
[imageHeight, imageWidth, ~] = size(noiseImage);

N = imageHeight-patchSize+1;
M = imageWidth-patchSize+1;
rows = 1:N;
columns = 1:M;
fprintf('PSNR of the noisy image = %f \n', csnr(noiseImage, originalImage, 0, 0) );

resultImage = noiseImage;
initialSigma = par.sigma;
old_PSNR = 0;

for iter = 1 : par.iterationCount
    resultImage = resultImage + par.betta*(noiseImage - resultImage);
    diff = resultImage-noiseImage;
    vd = initialSigma^2-(mean(mean(diff.^2)));
        
    if (iter==1)
        par.sigma  = sqrt(abs(vd));
        blk_arr = [];
    else
        par.sigma  = sqrt(abs(vd))*par.gamma;
    end

    %Start Change
    [blk_arr, allPatches] = Block_matching( resultImage, par, noiseImage, 1, blk_arr, iter, initialSigma);

    updAllPatches = zeros( size(allPatches) );
    Weights =   zeros( size(allPatches) );
    patchesStacksCount = size(blk_arr,2);
    subTou = 2.9 * sqrt(2 * size(blk_arr,1)) * par.sigma^2;

    for  stackIdx = 1 : patchesStacksCount
        sub_arr = blk_arr(:, stackIdx);
        sub_arr = sub_arr(sub_arr ~= 0);
        patches = allPatches(:, sub_arr);
        patches = reshape(patches, [par.patchSize, par.patchSize, size(patches, 2)]);
        [patches, Wi] = hosvdFilter(patches, subTou);

        patches = reshape(patches, [par.patchSize * par.patchSize, size(patches, 3)]);
        Wi = reshape(Wi, [par.patchSize * par.patchSize, size(Wi, 3)]);

        updAllPatches(:, sub_arr) = patches;
        Weights(:, sub_arr) = Wi;
    end
    %End Change



    resultImage = zeros(imageHeight,imageWidth);
    im_wei = zeros(imageHeight,imageWidth);
    k = 0;
    for x = 1:patchSize
        for y = 1:patchSize
            k = k+1;
            resultImage(rows-1+x,columns-1+y) = resultImage(rows-1+x,columns-1+y) + reshape( updAllPatches(k,:)', [N M]);
            im_wei(rows-1+x,columns-1+y) = im_wei(rows-1+x,columns-1+y) + reshape( Weights(k,:)', [N M]);
        end
    end
    resultImage = resultImage./(im_wei+eps);
    
    if isfield(par,'originalImage')
        PSNR = csnr( resultImage, originalImage, 0, 0 );
        SSIM = cal_ssim( resultImage, originalImage, 0, 0 );
        %if old_PSNR > PSNR
        %    PSNR = old_PSNR;
        %    break;
        %end
        %old_PSNR = PSNR;
    end
    
    fprintf( 'Iteration %d : sigma = %2.2f, PSNR = %2.2f, SSIM = %2.4f\n', iter, par.sigma, PSNR, SSIM );
end

%if isfield(par,'originalImage')
%   PSNR = csnr( resultImage, originalImage, 0, 0 );
%   SSIM = cal_ssim( resultImage, originalImage, 0, 0 );
%end

fprintf('Total elapsed time = %f min\n', (etime(clock,startTime)/60) );
return;

function [updPatchesStack, W] = hosvdFilter( patchesStack, subTou)
[Core, U] = hosvd(patchesStack);
CoreAbs = abs(Core);


tau = subTou ./ (CoreAbs+eps);
Core = Core .* (CoreAbs > tau);


nonZeroElements = nnz(Core);
elementsCount = numel(Core);
if nonZeroElements==elementsCount
    wei = 1/elementsCount;
else
    wei = elementsCount/(elementsCount + nonZeroElements);
end

updPatchesStack = wei * tprod(Core, U);
W = wei * ones( size(updPatchesStack) );
return;


