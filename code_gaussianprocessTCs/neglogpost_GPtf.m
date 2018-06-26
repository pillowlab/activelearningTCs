function [negL,dL,ddL] = neglogpost_GPtf(prs,Bproj,rsp,xct,gfun,mu,Kinv)
% Negative log-posterior for GP-poisson tuning function model
%
% [negL,dL,ddL] = neglogpost_GPtf(prs,B,rsp,gfun,mu,Kinv)
% 
% INPUT
% 
%
% OUTPUT
%   L [1 x 1] - negative log-posterior
%  dL [n x 1] - gradient
% ddL [n x n] - Hessian (2nd derivative matrix)

phi = Bproj*prs; % project parameters into space of phi

% transform by nonlinearity g
switch nargout
    case 0
        f = gfun(phi); 
    case 1 % Compute function only
        f = gfun(phi); 
    case 2 % Compute function & gradient
        [f,df] = gfun(phi);
    case 3 % Compute function, gradient & Hessian
        [f,df,ddf] = gfun(phi);
end        
        
% evaluate function
logli = rsp'*log(f) - xct'*f; % log-likelihood
logprior = -.5*(prs-mu)'*Kinv*(prs-mu); % log-posterior
negL = - (logli + logprior);

if nargout > 1 
    % evaluate gradient
    dlogli = Bproj'*(df.*(rsp./f - xct));  % log-likelihood
    dlogprior = -Kinv*(prs-mu); % log-prior
    dL = -(dlogli+ dlogprior);
end
 
if nargout > 2 
    % evaluate Hessian
    Hlogli = Bproj'*bsxfun(@times, rsp.*((f.*ddf-df.^2)./(f.^2)) - xct.*ddf,Bproj); % log-li
    Hlogprior = -Kinv; % log-prior
    ddL = -(Hlogli+Hlogprior);
end
 

