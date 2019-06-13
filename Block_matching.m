function  [pos_arr, X0] = Block_matching(im, par, noiseImage, Class, origin_pos)
searchRadius      = 21;
patchSize         = par.patchSize;
patchSize2        = patchSize^2;
step              = par.step;
%trashHold = 3 * par.sigma^2 * patchSize^2;

N                 = size(im,1)-patchSize+1;
M                 = size(im,2)-patchSize+1;
rows              = [1:step:N];
rows              = [rows rows(end)+1:N];
columns           = [1:step:M];
columns           = [columns columns(end)+1:M];
L                 = N*M;
X                 = zeros(patchSize*patchSize, L, 'single');

k    =  0;
for i  = 1:patchSize
    for j  = 1:patchSize
        k    =  k+1;
        blk  =  im(i:end-patchSize+i,j:end-patchSize+j);
        X(k,:) =  blk(:)';
    end
end

X0 = X;
if Class == 0    
    pos_arr = origin_pos;
else
    if par.patch_method == 1
        I     =   (1:L);
        I     =   reshape(I, N, M);
        N1    =   length(rows);
        M1    =   length(columns);
        X         =  X';
        pos_arr   =  zeros(par.patchStackSize, N1*M1 );
        for  i  =  1 : N1
            for  j  =  1 : M1
                row     =   rows(i);
                col     =   columns(j);
                off     =  (col-1)*N + row;
                off1    =  (j-1)*N1 + i;
                        
                rmin    =   max( row-searchRadius, 1 );
                rmax    =   min( row+searchRadius, N );
                cmin    =   max( col-searchRadius, 1 );
                cmax    =   min( col+searchRadius, M );
                 
                idx     =   I(rmin:rmax, cmin:cmax);
                idx     =   idx(:);
                patchesInSearchArea       =   X(idx, :);        
                currentPatch       =   X(off, :);
                
                dis = 0;
                for k = 1:patchSize2
                    dis   =  dis + (patchesInSearchArea(:,k) - currentPatch(k)).^2;
                end
                dis   =  dis./patchSize2;
                [val,ind]   =  sort(dis);
                %ind(val >= trashHold) = [];
                pos_arr(:,off1)  =  idx( ind(1:par.patchStackSize) );        
            end
        end

    elseif par.patch_method == 2
        [par1, model] = GMMInitial(par.sigma, im);
        [X1, Sigma_arr] = GMMim2patch(im, noiseImage, par1);
        pos_arr = GMM(Sigma_arr, X1, par1, model, par);
    end
end

