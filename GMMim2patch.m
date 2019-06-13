function  [Y SigmaArr]  =  GMMim2patch( E_Img,N_Img, par )
% Based on code written by Shuhang Gu (cssgu@comp.polyu.edu.hk)
TotalPatNum =   (size(E_Img,1)-par.win+1)*(size(E_Img,2)-par.win+1);                %Total Patch Number in the image
Y           =   zeros(par.win*par.win, TotalPatNum, 'single');                      %Current Patches
N_Y         =   zeros(par.win*par.win, TotalPatNum, 'single');                      %Patches in the original noisy image
k           =   0;

for i  = 1:par.win
    for j  = 1:par.win
        k           =  k+1;
        E_patch     =  E_Img(i:end-par.win+i,j:end-par.win+j);
        N_patch     =  N_Img(i:end-par.win+i,j:end-par.win+j);        
        Y(k,:)      =  E_patch(:)';
        N_Y(k,:)    =  N_patch(:)';
    end
end
SigmaArr = par.siglamda*sqrt(abs(repmat(par.nSig^2,1,size(Y,2))-mean((N_Y-Y).^2)));          %Estimated Local Noise Level