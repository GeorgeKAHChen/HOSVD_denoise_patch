function [idx, center2, m2] = kmeans2(data, m, MaxIter,minSamples)
[n, ~]  =  size(data);
dex     =  randperm(n);
center  =  data(dex(1:m),:);

for i   = 1:MaxIter;
    nul = zeros(m,1);
    [~, idx] = min(sqdist(center', data'));
    for j = 1:m;
        dex  = find(idx == j);
        l    = length(dex);
        cltr = data(dex,:);
        if l > 1;
            center(j,:) = mean(cltr);
        elseif l == 1;
            center(j,:) = cltr;
        else
            nul(j) = 1;
        end;
    end;
    dex    =  find(nul == 0);
    m      =  length(dex);    
    center = center(dex,:);
end;
[~, idx]   =  min(sqdist(center', data'));
center2    =   [];
m2         =   0;
for j = 1:m;
    dex =  find(idx == j);
    l   =  length(dex);
    if l >= minSamples
         center2 = [center2; center(j,:)];
         m2=m2+1;
    end
end
if m2 > 1
    [~, idx] = min(sqdist(center2', data'));
elseif m2==1
    idx = ones(1,n);   
else
    m2 = m;
end
 