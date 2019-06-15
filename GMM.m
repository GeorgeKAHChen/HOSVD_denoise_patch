function ks = GMM(Sigma_arr,X,par1,model, par)
    %[MY,ks,group,nSig,PF]
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
    

    %By KazukiAmakawa, change for out methods
    %group_size = zeros(1, max(ks));
    %length(ks)
    %for i = 1: length(ks)
    %    group_size(ks(i)) = group_size(ks(i)) + 1;
    %end
    %group_size
    %outputs_size = max(group_size)
    %outputs_size = par.patchStackSize;
    %outputs = zeros(floor((length(ks)+0.5) / par.step) + 1, outputs_size);
    %for i0 = 1:length(index)
    %    i = index(i0)
    %    LocalPatch = [];
    %    val = 1;
    %    outputs(val, i) = i;
    %    for j = 1: length(ks)
    %        if i == j
    %            continue;
    %        end
    %        if val >= outputs_size
    %            break;
    %        end
    %        if ks(i) == ks(j)
    %            val = val + 1
    %            outputs(val, i) = j
    %        end
    %    end
    %    i = i + par.step - 1;
    %    %outputs = [outputs, LocalPatch];
    %end
end

