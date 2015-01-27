function linear_libsvm_classification(tr_max_fea,tr_label,ts_max_fea,ts_label,resultmaxpathspm,C)


% load(traindatapath);
clabel = unique(tr_label);
nclass = length(clabel);

% [tr_max_fea, minvalue, maxvalue] = scaletrain(tr_max_fea, 'power');
fprintf('compute kernel for train\n');
% liner kernel
kernel_train = tr_max_fea'*tr_max_fea;
options = ['-t 4 -c ' num2str(C) ' -b 1 -q'];

fprintf('begin svm train\n');
for kk = 1:nclass
    tr_label_new = 2*ones(length(tr_label),1);
    idx = find(tr_label == kk);
    tr_label_new(idx) = 1;
    model(kk) = svmtrain(tr_label_new,[(1:length(tr_label))',double(kernel_train)],options);
end
clear kernel_train;   
fprintf('compute kernel for predict\n');

% ts_max_fea = scaletest(ts_max_fea, 'power', minvalue, maxvalue);
% linear kernel
kernel_test = ts_max_fea'*tr_max_fea;

clear tr_max_fea;  
fprintf('begin svm predict\n');
prob_est = zeros(nclass,length(ts_label),'single');
for kk = 1:nclass
    ts_label_new = 2*ones(length(ts_label),1);
    idx = find(ts_label == kk);
    ts_label_new(idx) = 1;
    [predict_label_new, accuracy_new,prob_estimates_new] = svmpredict(ts_label_new,[(1:length(ts_label))',double(kernel_test)],model(kk),'-b 1');
    if kk == 1
        prob_est(kk,:) = prob_estimates_new(:,1);
    else
        prob_est(kk,:) = prob_estimates_new(:,2);
    end
end
[dist,predict_label] = max(prob_est);       
clear kernel_test;

predict_label = predict_label';      
clear ts_max_fea; 
acc_recall = zeros(nclass, 1,'single');
acc_acc = zeros(nclass,1,'single');
for jj = 1 : nclass,
    idx = find(ts_label == jj);
    curr_pred_label = predict_label(idx);
    curr_gnd_label = ts_label(idx);    
    acc_recall(jj) = single(length(find(curr_pred_label == curr_gnd_label)))/single(length(idx));
    acc_acc(jj) = single(length(find(curr_pred_label == curr_gnd_label)))/single(length(find(predict_label == jj)));
end;        
accuracy_recall = mean(acc_recall);      
fprintf('recall: %f\n',accuracy_recall);
accuracy_acc = mean(acc_acc);      
fprintf('accuracy: %f\n',accuracy_acc); 
save(resultmaxpathspm,'acc_recall','acc_acc');



