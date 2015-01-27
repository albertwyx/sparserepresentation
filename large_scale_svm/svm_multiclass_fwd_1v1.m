function [C Y] = svm_multiclass_fwd_1v1(X, w, class_name)

% function [C Y] = li2nsvm_multiclass_fwd(X, w, b, class_name):
% make multi-class prediction

Y = X*w;% + +repmat(b,[size(X,1),1]);
YY = sign(Y);
Y0_nneg = (YY+1)/2;
Y0_neg = -1*(YY-1)/2;
cnum = length(class_name);
vote = zeros(size(Y,1),cnum);
count = 0;
for i = 1:cnum
    for j = i+1:cnum
       count = count+1;       
       vote(:,i) = vote(:,i) + Y0_nneg(:,count);       
       vote(:,j) = vote(:,j) +Y0_neg(:,count);       
    end
end
[vo,C] = max(vote,[],2);

% C = oneofc_inv(Y, class_name);
% accuracy = sum(Yte==Cte)/size(Yte,1);
% fprintf('the accuracy is %f \n', accuracy);

