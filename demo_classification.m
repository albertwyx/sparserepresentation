function demo_classification(dataset,method,seed,C,nu,knn,alpha,nDim,tr_num,maxiter,kermethod)
addpath(genpath('large_scale_svm'));
addpath(genpath('libsvm-3.1'));
addpath(genpath('sparse_coding'));
addpath(genpath('Laplacian'));
addpath(genpath('package'));
par.coding        =   method;
par.dataset       =   dataset;
par.nDim          =   nDim;                 % the eigenfaces dimension
par.nu            =   nu;
par.C             =   C;
par.knn           =   knn;
par.alpha         =   alpha;
par.seed          =   seed;
par.maxiter       =   maxiter;
par.kermethod     =   kermethod;
datasetpath = ['data/dataset/' par.dataset '.mat'];
load(datasetpath);
par.gnd = gnd;
par.fea = fea';
switch par.dataset
    case 'YaleB_32x32'
        par.tr_num = tr_num;
        par.ts_num = 100000;
        par.nclass = 38;
    case 'PIE_32x32'
        par.tr_num = tr_num;
        par.ts_num = 100000;
        par.nclass = 68;   
    case 'AR_dataset'
        par.tr_num = tr_num;
        par.ts_num = 100000;
        par.nclass = 100;   
    case 'ORL_32x32'
        par.tr_num = tr_num;
        par.ts_num = 100000;
        par.nclass = 40;   
    case 'usps_11000'
        par.tr_num = tr_num;
        par.ts_num = 100000;
        par.nclass = 10;  
    case 'COIL20'
        par.tr_num = tr_num;
        par.ts_num = 100000;
        par.nclass = 20; 
    case 'COIL100'
        par.tr_num = tr_num;
        par.ts_num = 100000;
        par.nclass = 100;   
    case 'scene_sparsecodingnew_1024'
        par.tr_num = tr_num;
        par.ts_num = 100000;
        par.nclass = 15;  
    case 'scene_kmeans_256'
        par.tr_num = tr_num;
        par.ts_num = 100000;
        par.nclass = 15;          
    case 'Caltech101_sparsecodingnew_1024'
        par.tr_num = tr_num;
        par.ts_num = 50;
        par.nclass = 101;
    case 'Caltech101_kmeans_256'
        par.tr_num = tr_num;
        par.ts_num = 50;
        par.nclass = 101;        
    case '256_ObjectCategories_sparsecodingnew_1024'
        par.tr_num = tr_num;
        par.ts_num = 15;
        par.nclass = 256;  
    case '256_ObjectCategories_kmeans_256'
        par.tr_num = tr_num;
        par.ts_num = 15;
        par.nclass = 256;          
    case 'event8_sparsecodingnew_1024'
        par.tr_num = tr_num;
        par.ts_num = 60;
        par.nclass = 8;    
    case 'event8_kmeans_256'
        par.tr_num = tr_num;
        par.ts_num = 60;
        par.nclass = 8;   
    case 'event8_kmeans_256_spm_2'
        par.tr_num = tr_num;
        par.ts_num = 60;
        par.nclass = 8;          
    case 'MNIST'
        par.tr_num = tr_num;
        par.ts_num = 10000;
        par.nclass = 10;     
        par.gnd = par.gnd+1;  
end
par.folder_results = ['data/results/' num2str(par.seed) '/' par.dataset];
if ~isdir(par.folder_results),
    mkdir(par.folder_results);
end;
par.resultmaxpathspm = [par.folder_results '/' par.coding '_trnum_' num2str(par.tr_num)];

%par.splitdatapath = [par.folder_results '\splitdata_' par.dataSet '_' num2str(par.tr_num) '.mat'];
[tr_fea tr_label ts_fea ts_label]= split_data(par.seed,par.tr_num,par.ts_num,par.fea,par.gnd);

%pca
% model   =  pca(tr_fea,par.nDim);
% tr_fea  =  model.W'*tr_fea;
% ts_fea  =  model.W'*ts_fea;

%  tr_fea = [tr_fea 255*ones(1024,2)];
%  visual(tr_fea(:,1:30),2,10);

tr_fea  =  tr_fea./( repmat(sqrt(sum(tr_fea.*tr_fea)), [size(tr_fea,1),1]) );
ts_fea  =  ts_fea./( repmat(sqrt(sum(ts_fea.*ts_fea)), [size(tr_fea,1),1]) );


switch par.coding
    case 'libsvm'
        par.resultmaxpathspm = [par.resultmaxpathspm '_C_' num2str(par.C) '.mat'];
        linear_libsvm_classification(tr_fea,tr_label,ts_fea,ts_label,par.resultmaxpathspm,par.C);  
    case 'svm'
        par.resultmaxpathspm = [par.resultmaxpathspm '_C_' num2str(par.C) '.mat'];
        linear_classification_svm(tr_fea,tr_label,ts_fea,ts_label,par.resultmaxpathspm,par.C);        
    case 'spsvm'
        par.resultmaxpathspm = [par.resultmaxpathspm '_C_' num2str(par.C) '_nu_' num2str(par.nu) '_knn_' num2str(par.knn) '_' num2str(par.alpha) '.mat'];
        linear_classification_spsvm(tr_fea,tr_label,ts_fea,ts_label,par.resultmaxpathspm,par.C,par.nu,par.knn,par.alpha);
    case 'lpsvm'
        par.resultmaxpathspm = [par.resultmaxpathspm '_C_' num2str(par.C) '_nu_' num2str(par.nu) '_knn_' num2str(par.knn) '.mat'];
        linear_classification_lpsvm(tr_fea,tr_label,ts_fea,ts_label,par.resultmaxpathspm,par.C,par.nu,par.knn);    
    case '1v1libsvm'
        par.resultmaxpathspm = [par.resultmaxpathspm '_C_' num2str(par.C) '.mat'];
        linear_libsvm_classification_1v1(tr_fea,tr_label,ts_fea,ts_label,par.resultmaxpathspm,par.C);            
    case '1v1svm'
        par.resultmaxpathspm = [par.resultmaxpathspm '_C_' num2str(par.C) '.mat'];
        linear_classification_1v1svm(tr_fea,tr_label,ts_fea,ts_label,par.resultmaxpathspm,par.C); 
    case '1v1spsvm'
        par.resultmaxpathspm = [par.resultmaxpathspm '_C_' num2str(par.C) '_nu_' num2str(par.nu) '_knn_' num2str(par.knn) '_' num2str(par.alpha) '.mat'];
        linear_classification_1v1spsvm(tr_fea,tr_label,ts_fea,ts_label,par.resultmaxpathspm,par.C,par.nu,par.knn,par.alpha);     
    case 'FS'
        par.resultmaxpathspm = [par.resultmaxpathspm '_' num2str(par.alpha) '.mat'];
%         [tr_fea, minvalue, maxvalue] = scaletrain(tr_fea, 'power');
%         ts_fea = scaletest(ts_fea, 'power', minvalue, maxvalue);
        B = tr_fea;
        rec_err = zeros(par.nclass,1);
        ID      = [];
        for pro_i = 1:size(ts_fea,2)
            Codes = feature_sign(B, ts_fea(:,pro_i), alpha); 
            for class = 1:par.nclass
                b = B (:,tr_label == class);
                s = Codes(tr_label == class);
                rec_err(class) = sum((ts_fea(:,pro_i)-b*s).^2);        
            end    
            index = find(rec_err==min(rec_err));
            ID = [ID index(1)];
        end            
       acc = zeros(par.nclass, 1);
    
       for jj = 1 : par.nclass,
           idx = find(ts_label == jj);
           curr_pred_label = ID(idx);
           curr_gnd_label = ts_label(idx);    
           acc(jj) = length(find(curr_pred_label == curr_gnd_label'))/length(idx);
       end;             
       fprintf('Mean accuracy: %f\n', mean(acc));

       save(par.resultmaxpathspm,'acc'); 
%         acc = sum(ID == ts_label')/length(ts_label);
%         fprintf('recogniton rate is %.7f\n',acc);
%         save(par.resultmaxpathspm,'acc');
    case 'BCDSC'
        par.resultmaxpathspm = [par.resultmaxpathspm '_' num2str(par.alpha) '.mat'];
%         [tr_fea, minvalue, maxvalue] = scaletrain(tr_fea, 'power');
%         ts_fea = scaletest(ts_fea, 'power', minvalue, maxvalue);
        B = tr_fea;
        rec_err = zeros(par.nclass,1);
        ID      = [];     
        A = B'*B ;
        Q = B'*ts_fea;
        Codes = zeros(size(tr_fea,2),size(ts_fea,2),'single');
        Codes =sparse_coding_fixw(single(Q),single(A),single(par.alpha),single(par.maxiter),Codes,single(ts_fea),single(B));
        for pro_i = 1:size(ts_fea,2)
            for class = 1:par.nclass
                b = B (:,tr_label == class);
                s = Codes(tr_label == class,pro_i);
                rec_err(class) = sum((ts_fea(:,pro_i)-b*s).^2);        
            end    
            index = find(rec_err==min(rec_err));
            ID = [ID index(1)];
        end          
       acc = zeros(par.nclass, 1);
    
       for jj = 1 : par.nclass,
           idx = find(ts_label == jj);
           curr_pred_label = ID(idx);
           curr_gnd_label = ts_label(idx);    
           acc(jj) = length(find(curr_pred_label == curr_gnd_label'))/length(idx);
       end;            
       fprintf('Mean accuracy: %f\n', mean(acc));
       save(par.resultmaxpathspm,'acc');
     case 'Hellinger'
        par.resultmaxpathspm = [par.resultmaxpathspm '_' num2str(par.alpha) '.mat'];
        [tr_fea, minvalue, maxvalue] = scaletrain(tr_fea, 'power');
        ts_fea = scaletest(ts_fea, 'power', minvalue, maxvalue);
        B = tr_fea;
        rec_err = zeros(par.nclass,1);
        ID      = [];     
        A = B'*B ;
        Q = B'*ts_fea;
        Codes = zeros(size(tr_fea,2),size(ts_fea,2),'single');
        Codes =sparse_coding_fixw(single(Q),single(A),single(par.alpha),single(par.maxiter),Codes,single(ts_fea),single(B));
        for pro_i = 1:size(ts_fea,2)
            for class = 1:par.nclass
                b = B (:,tr_label == class);
                s = Codes(tr_label == class,pro_i);
                rec_err(class) = sum((ts_fea(:,pro_i)-b*s).^2);        
            end    
            index = find(rec_err==min(rec_err));
            ID = [ID index(1)];
        end          
       acc = zeros(par.nclass, 1);
    
       for jj = 1 : par.nclass,
           idx = find(ts_label == jj);
           curr_pred_label = ID(idx);
           curr_gnd_label = ts_label(idx);    
           acc(jj) = length(find(curr_pred_label == curr_gnd_label'))/length(idx);
       end;            
       fprintf('Mean accuracy: %f\n', mean(acc));
       save(par.resultmaxpathspm,'acc');      
     case 'KFLCSC'
%         [tr_fea, minvalue, maxvalue] = scaletrain(tr_fea, 'power');
%         ts_fea = scaletest(ts_fea, 'power', minvalue, maxvalue);        
        par.resultmaxpathspm = [par.resultmaxpathspm '_' num2str(par.alpha)];
        B = tr_fea;
        
                   
        switch kermethod
            case 'rbf'
                gamma = 1;
                Q = sp_dist2(B',ts_fea');
                Q = exp(-gamma*Q);  
                A = sp_dist2(B',B');
                A = exp(-gamma*A);
                sumX = size(ts_fea,2);  
                par.resultmaxpathspm = [par.resultmaxpathspm kermethod '_' num2str(gamma) '.mat'];
            case 'poly'
                polyc = 1;
                polyd = 2;
                Q = (B'*ts_fea+polyc).^polyd;
                A = (B'*B+polyc).^polyd;
                P = (ts_fea'*ts_fea+polyc).^polyd;
                sumX = trace(P);
                par.resultmaxpathspm = [par.resultmaxpathspm kermethod '_' num2str(polyc) '_' num2str(polyd) '.mat'];
            case 'hik'
                distance = sp_dist3(double(B)',double(ts_fea)');
                Q = fabs_c(single(B)',single(ts_fea)');   
                Q = (distance-Q)/2;
                Q(Q<0) = 0;  
                distance = sp_dist3(double(B)',double(B)');
                A = fabs_c(single(B)',single(B)');   
                A = (distance-A)/2;
                A(A<0) = 0;                  
                sumX = sum(sum(ts_fea));  
                par.resultmaxpathspm = [par.resultmaxpathspm kermethod '.mat'];
            case 'laplacian'
                gamma = 0.1;
                Q = fabs_c(B',ts_fea');    
                Q = exp(-gamma*Q);
                A = fabs_c(B',B');    
                A = exp(-gamma*A);                
                sumX = size(ts_fea,2);
                par.resultmaxpathspm = [par.resultmaxpathspm kermethod '_' num2str(gamma) '.mat'];
        end
%         C = W'*Q;
%         A = B'*B ;
%         Q = B'*ts_fea;    
        rec_err = zeros(par.nclass,1);
        ID      = [];             
        Codes = zeros(size(tr_fea,2),size(ts_fea,2),'single');
        Codes = kernel_FLC_sparse_coding_fixw(single(Q),single(A),single(par.alpha),single(par.maxiter),Codes,single(ts_fea),single(sumX));
        for pro_i = 1:size(ts_fea,2)
            for class = 1:par.nclass
                b = B (:,tr_label == class);
                s = Codes(tr_label == class,pro_i);
                rec_err(class) = sum((ts_fea(:,pro_i)-b*s).^2);        
            end    
            index = find(rec_err==min(rec_err));
            ID = [ID index(1)];
        end          
       acc = zeros(par.nclass, 1);
    
       for jj = 1 : par.nclass,
           idx = find(ts_label == jj);
           curr_pred_label = ID(idx);
           curr_gnd_label = ts_label(idx);    
           acc(jj) = length(find(curr_pred_label == curr_gnd_label'))/length(idx);
       end;             
       fprintf('Mean accuracy: %f\n', mean(acc));
       save(par.resultmaxpathspm,'acc');      
       
       
    case 'knn'
        [tr_fea, minvalue, maxvalue] = scaletrain(tr_fea, 'power');
        ts_fea = scaletest(ts_fea, 'power', minvalue, maxvalue);
        par.resultmaxpathspm = [par.resultmaxpathspm '.mat'];
%         rec_err = zeros(par.nclass,1);
        ID      = [];  
        distance = sp_dist2(ts_fea',tr_fea');
        for pro_i = 1:size(ts_fea,2)
            d =  distance(pro_i,:);
            [dummy, idx] = sort(d, 'ascend');
            ID = [ID tr_label(idx(1))];
        end
       acc = zeros(par.nclass, 1);
    
       for jj = 1 : par.nclass,
           idx = find(ts_label == jj);
           curr_pred_label = ID(idx);
           curr_gnd_label = ts_label(idx);    
           acc(jj) = length(find(curr_pred_label == curr_gnd_label'))/length(idx);
       end;            
       fprintf('Mean accuracy: %f\n', mean(acc));

       save(par.resultmaxpathspm,'acc');
%         acc = sum(ID == ts_label')/length(ts_label);
%         fprintf('recogniton rate is %.7f\n',acc);
%         save(par.resultmaxpathspm,'acc');        
end




