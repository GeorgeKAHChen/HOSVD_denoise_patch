function [MY,ks,group,nSig,PF] = GmmCluster( Sigma_arr,X,par1,model)
    nSig = mean(Sigma_arr);
    Y    = X/255;
    MY   = mean(Y);
    Y    = bsxfun(@minus,Y,MY);
    SigmaNoise = (nSig/255)^2*eye(par1.win^2);
    PF = zeros(size(model.weight,2),size(Y,2));
    for i = 1:size(model.weight,2)
        PF(i,:) = log(model.weight(i)) + loggausspdf(Y,model.Sigma(:,:,i) + SigmaNoise);
    end
    [~,ks] =    max(PF);
    group=[];
    %If the patch number of some class is too small (less than 10), 
    %we merge these patches to the other classes most likely.
    for i=1:size(model.weight,2)
        inds = find(ks == i);
        sl   = length(inds);
        if sl>=10% 
            group = [group i];
        end
    end
    [~,ks] = max(PF(group,:));
end

