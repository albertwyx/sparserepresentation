function [w] = linearsvm_no_b(y, X, C)
    options = optimset('Display','off','GradObj','on','LargeScale','off','TolFun',1e-5);
%     ,'MaxIter', 200  
    [w, l] = fminunc(@(w)(svm(y, X, C, w)), zeros(size(X, 2), 1), options);
 %'MaxIter', 30
    
    margins = y .* (X * w);
    l1 = 0.5 * (w' * w);
    l2 = C * sum(max(0, 1 - margins));
    
    %b = w(end);
    %w = w(1:end - 1);    
end

function [loss, gradient] = svm(y, X, C, w)
    margins = y .* (X * w);
    loss = 0.5 * (w' * w) + C * sum(max(0, 1 - margins));
    gradient = w + C * (((margins < 1) .* -y)' * X)';
    %gradient = [gradient; C * sum(-y(margins < 1))];
end