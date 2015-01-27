clear;
clc;



% % demo_classification(dataset,method,1000,1,2^-4,69,0.05,128,70);

dataset = 'AR_dataset';
method = 'KFLCSC';
kermethod = 'hik';
knn = 5;
% alpha = 10^-5;
nDim = 256;
tr_num = 10;
maxiter = 5000;
C = 2^9;
nu = 2^-1;
% for Iteralpha = -3:1:-3    %
%     alpha = 10^Iteralpha;
%     fprintf('alpha: %f\n', alpha);
%     for seed = 1002:1:1009
%         demo_classification(dataset,method,seed,C,nu,knn,alpha,nDim,tr_num,maxiter,kermethod);
%     end
% end
for Iteralpha = -1:1:0    %
    alpha = 10^Iteralpha;
    fprintf('alpha: %f\n', alpha);
    for seed = 1000:1:1009
        demo_classification(dataset,method,seed,C,nu,knn,alpha,nDim,tr_num,maxiter,kermethod);
    end
end

dataset = 'AR_dataset';
method = 'KFLCSC';
kermethod = 'rbf';
knn = 5;
% alpha = 10^-5;
nDim = 256;
tr_num = 10;
maxiter = 5000;
C = 2^9;
nu = 2^-1;
for Iteralpha = -5:1:0    %
    alpha = 10^Iteralpha;
    fprintf('alpha: %f\n', alpha);
    for seed = 1000:1:1009
        demo_classification(dataset,method,seed,C,nu,knn,alpha,nDim,tr_num,maxiter,kermethod);
    end
end
