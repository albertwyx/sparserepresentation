function [fea, minvalue, maxvalue] = scaletrain(fea, type)
% normalize features. This step empirically improves the performance
% written by Liefeng Bo on 01/04/2011 in University of Washington

minvalue = min(fea,[],2);
maxvalue = max(fea,[],2);
maxminvalue = maxvalue - minvalue;
maxminvalue(maxminvalue == 0) = 1e-16;
switch type
  case 'linear'
	for i = 1:size(fea,2)
	    fea(:,i) = (fea(:,i) - minvalue)./maxminvalue;
	end
  case 'power'
        ppp = 0.5;
        for i = 1:size(fea,2)
            fea(:,i) = sign(fea(:,i)).*abs(fea(:,i)).^ppp;
        end     
  otherwise
       disp('Unknown type');
end

