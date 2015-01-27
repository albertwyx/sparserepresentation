
function [L,U,A] = laplacian(DATA, TYPE, PARAM)  

% Calculate the graph laplacian of the adjacency graph of data set DATA.
%
% L = laplacian(DATA, TYPE, PARAM)  
% 
% DATA - NxK matrix. Data points are rows. 
% TYPE - string 'nn' or string 'epsballs'
% PARAM - integer if TYPE='nn', real number if TYPE='epsballs'
%
% Returns: L, sparse symmetric NxN matrix 
%
% Example:
%
% L = laplacian(X,'nn',6)
% L contains the Laplacian of the graph obtained from connecting
% each point of the data set to its 6 nearest neigbours.
%
%
% Author: 
%
% Mikhail Belkin 
% misha@math.uchicago.edu
%

% disp(' ');
% disp('Laplacian Egenmaps Embedding.');

% calculate the adjacency matrix for DATA
A = adjacency(DATA, TYPE, PARAM);   %A¡ª¡ªKNN distance matrix
  
U = A;

% disassemble the sparse matrix
[A_i, A_j] = find(A);

for i = 1: size(A_i)  
  % replece distances by 1
  % gaussain kernel can be used instead of 1:
  % U(A_i(i), A_j(i)) = exp(-A_v^2/t);
  U(A_i(i), A_j(i)) = 1;
end;

% disp('Computing Laplacian eigenvectors.');

D = sum(U(:,:),1);   
L = spdiags(D',0,speye(size(U,1)))-U;

