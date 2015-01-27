function S = local_info_lp(X,label,par)

N = size(X,2);
S = zeros(N,N);
nclass = length(unique(label));
% laplacian(X,'nn',par.knn);
% for ii = 1:N
%     id = (label == label(ii));
%     B = X(:,id);
%     d = sp_dist2(X(:,ii)',B'); 
%     [dummy, idx] = sort(d, 'ascend');
%     neighborhood = (idx(2:par.knn+1)');
%     
%     idx(neighborhood) = -10;
%     idx(idx ~= -10) = 0;
%     idx(idx == -10) = 1;
%     
% 
%     ind = (id(id == 1)+idx'-1);
%     id(id == 1) = ind;
%     BB = X(:, id == 1);
%     
%     S(id == 1,ii) = feature_sign(BB, X(:,ii), par.alpha); 
% end
for ii = 1:nclass
    id = (label == ii);
    B = X(:,id);
    S(id,id) = laplacian(B','nn',par.knn);
end