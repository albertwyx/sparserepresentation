function linear_classification_1v1svm(tr_max_fea,tr_label,ts_max_fea,ts_label,resultmaxpathspm,lambda)

[tr_max_fea, minvalue, maxvalue] = scaletrain(tr_max_fea, 'power');
clabel = unique(tr_label);
nclass = length(clabel);
fprintf('begin svm train\n');
[w,class_name] = svm_multiclass_lbfgs_1v1(tr_max_fea', tr_label, lambda);

ts_max_fea = scaletest(ts_max_fea, 'power', minvalue, maxvalue);
fprintf('begin svm predict\n');
[C, Y] = svm_multiclass_fwd_1v1(ts_max_fea', w, class_name);

acc = zeros(length(class_name), 1);
    
for jj = 1 : length(class_name),
    c = class_name(jj);
    idx = find(ts_label == c);
    curr_pred_label = C(idx);
    curr_gnd_label = ts_label(idx);    
    acc(jj) = length(find(curr_pred_label == curr_gnd_label))/length(idx);
end;    

fprintf('Mean accuracy: %f\n', mean(acc));

save(resultmaxpathspm,'acc');

