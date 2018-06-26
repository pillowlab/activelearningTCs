function [negL,dL,H] = neglogliPoissonTC(prs,x,y)
% Evaluates loglikelihood of Poisson tuning curve model
% 


mu = prs(1);
sig = prs(2);
A = prs(3);
bl = prs(4);

% Evaluate nonlinearity
lam = A*exp(-(x-mu).^2/(2*sig.^2))+bl;

% Evaluate negative log-likelihood
negL = -y'*log(lam) + sum(lam);

% Evaluate Gradient
if nargout > 1
    dlam = ones(length(x),4);
    dlam(:,1) = ((x-mu).*(lam-bl))./(sig.^2);
    dlam(:,2) = ((x-mu).^2).*(lam-bl)./(sig.^3);
    dlam(:,3) = (lam-bl)./A;
    
    dL = (-y'*bsxfun(@rdivide,dlam,lam)+sum(dlam))'; 
end

% Evaluate Hessian
if nargout > 2
    ddlam = 0;
    H = 0;
end