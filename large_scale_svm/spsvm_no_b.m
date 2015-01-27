%to delete b or consider b
%each col of X is a sample
function [w] = spsvm_no_b( y, X, C, nu, S )
    options = optimset('Display','off','GradObj','on','LargeScale','off','TolFun',1e-5);
%     ,'MaxIter', 50
    I = eye(size(S,1));
    T = (I-S)*(I-S)';
    
    [w, l] = fminunc(@(w)(sp_svm(y, X, C, nu, T, w)), zeros(size(X, 2), 1), options);
     
    margins = y .* (X * w);
    l1 = 0.5 * (w' * w);
    l2 = C * sum(max(0, 1 - margins));
    
    %b = w(end);
%     w = w(1:end - 1);    
end
 
function [loss, gradient] = sp_svm(y, X, C, nu, T, w)
    tmp = X * w;
    margins = y .* tmp;
    loss = 0.5 * (w' * w) + C * sum(max(0, 1 - margins)) + tmp'*T*tmp*nu/2;
    gradient = w + C * (((margins < 1) .* -y)' * X)'+ nu* X'*T*X*w;
    %gradient = [gradient; C * sum(-y(margins < 1))];
end
