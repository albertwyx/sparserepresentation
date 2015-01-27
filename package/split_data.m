function [tr_fea tr_label ts_fea ts_label] = split_data(seed,tr_num,ts_num,fea,gnd)

tr_idx = [];
ts_idx = [];
clabel = unique(gnd);
for jj = 1:length(clabel),
    idx_label = find(gnd == jj);
    num = single(length(idx_label));
    rand('seed',double(jj)+(seed-1)*length(clabel));
    idx_rand = randperm(num);
    tr_idx = [tr_idx;(idx_label(idx_rand(1:tr_num)))];
    if ts_num < (num-tr_num)
        ts_idx = [ts_idx; (idx_label(idx_rand(tr_num+1:tr_num+ts_num)))];
    else
        ts_idx = [ts_idx; (idx_label(idx_rand(tr_num+1:end)))];
    end
end;
tr_fea = fea(:,tr_idx);
ts_fea = fea(:,ts_idx);
tr_label = gnd(tr_idx);
ts_label = gnd(ts_idx);

