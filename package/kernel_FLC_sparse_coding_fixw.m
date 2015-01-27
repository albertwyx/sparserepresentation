function [H] = kernel_FLC_sparse_coding_fixw(C,B,alpha,maxiter,H,Y,sumX)
nBases = size(C,1);
B_revise = tril(B,-1)+triu(B,1);
iter = 0;
fold = 100000000;
fnew = 9999999;
% while (fold-fnew)/fold>1*1e-5 && iter<maxiter
while iter<maxiter
    iter = iter+1;
    %update H 
    for i=1:nBases
        H_tp = C(i,:)-B_revise(i,:)*H;
        H(i,:) = (max(H_tp,alpha)+min(H_tp,-alpha))/B(i,i);    
    end 
%     fold = fnew;
%     fnew1 = sumX+trace(H'*B*H)-2*trace(C*H');
%     fnew2 = 2*alpha*(sum(sum(abs(H))));
%     fnew = fnew1+fnew2;
%     fprintf('iter = %d sparsity = %.5f ',iter,length(find(H(:)~=0))/length(H(:))); 
%     fprintf('relative error =%.5f %.5f %.5f\n',fnew1/size(Y,2),fnew2,fnew);      
end
return;