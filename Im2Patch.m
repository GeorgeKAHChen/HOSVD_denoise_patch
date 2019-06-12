function X = Im2Patch( image, par )
patchSize = par.patchSize;
N = size(image,1)-patchSize+1;
M = size(image,2)-patchSize+1;
L = N*M;
X = zeros(patchSize*patchSize, L, 'single');
k = 0;
for x = 1:patchSize
    for y = 1:patchSize
        k = k+1;
        blk = image(x:end-patchSize+x,y:end-patchSize+y);
        X(k,:) = blk(:)';
    end
end