function logL = logliPoissonTCbd(prs,x,y,LB,UB)
% Evaluates loglikelihood of Poisson tuning curve model
% 

if any(prs(:)>UB) || any(prs(:)<LB)
    logL = -inf;
    return 
end

mu = prs(1);
sig = prs(2);
A = exp(prs(3));
bl = prs(4);

% Evaluate nonlinearity
lam = A*exp(-(x-mu).^2/(2*sig.^2))+bl;

% Evaluate negative log-likelihood
logL = y'*log(lam) - sum(lam);

