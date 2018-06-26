function K = mkKernelMatrix_RBF(prs,xx,yy)
% Generate RBF or squared-exponential kernel
%
% K = mkKernelMatrix_RBF(prs,xx,yy)
%
% Covariance matrix parametrized as:  K_ij = rho*exp(((i-j)^2/(2*d^2))
%
% INPUTS:
%     prs [1 x 2] - kernel parameters [rho, d] (struct or vector)
%                   (rho = marginal variance, d = length scale)
%      xx [n1 x m] - stimuli: each row is a point in m-dimensional input space
%      yy [n2 x m] - stimuli (optimal)
%
% OUTPUT:
%   K [n1 x n1] or [n1 x n2] - kernel matrix 
%    
%
% Updated 2015.02.24 (jwp)


% Unpack inputs
if isstruct(prs)
    rho = prs.rho;
    d = prs.d;
else
    rho = prs(1);
    d = prs(2);
end

if nargin == 2
    % Compute squared distances between each row of X and every other
    %   ||x_i-x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2*x_i'*x_j
    xnrm = sum(xx.^2,2); % squared euclidean norm of each row
    sqrdists = bsxfun(@plus,xnrm,xnrm') - 2*(xx*xx');
elseif nargin == 3
    % compute between x and y
    xnrm = sum(xx.^2,2);
    ynrm = sum(yy.^2,2);
    sqrdists = bsxfun(@plus,xnrm,ynrm') - 2*(xx*yy');
end

K = rho*exp(-.5*sqrdists/d.^2); % the kernel matrix
    