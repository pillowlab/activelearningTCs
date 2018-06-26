function [fmap,fstd,phimap,phicov,gpstat] = runMAPinference_GPtf(xx,dat,gfun,theta,phi0)
% Compute MAP estimate for tuning function under GP-Poisson model
% [fmap,fstd,phimap,phicov,gpstat] = runMAPinference_GPtf(xx,dat,gfun,theta)
%
%  Model:
%    phi(x) ~ GP(mu,K)        % GP prior over phi(x)
%      f(x) = gfun(phi(x))    % nonlinear transformation
%    r|f(x) ~ Poiss(f(x))     % conditional spike count distribution
%
% INPUTS
%      xx [n x m] - each row is a stimulus at which to evaluate tf
%    dat [struct] - training data:  
%                   .x = stimuli (each row is a stim), 
%                   .r = responses (integer spike counts)
%       gfun [@f] - function handle for nonlinearity
%   theta [struct] - hyperparameters: 
%                   .mu = mean of GP
%                   .rho = marginal variance of kernel function
%                   .d = length scale of kernel function
%     phi0 [n x 1] - initial guess at fmap (OPTIONAL)
%
% OUTPUTS
%      fmap [n x 1] - MAP estimate of function value
%      fstd [n x 1] - f posterior marginal stdev at each point (delta method)
%    phimap [n x 1] - MAP esimate (posterior mode) of phi(x)
%    phicov [n x n] - full covariance of phi(x)
%    gpstat [struct] - stats from GP just where we have observed data
%                     .xid = stimuli 
%                     .mu = mean of phi(xid)
%                     .cov = covariance over phi(xid)

% set threshold for determining if K is ill-conditioned
condthresh = 1e8; 

% Unpack inputs
[n,ndim] = size(xx); % number of stimuli at which to evaluate fmap

% Unpack & pre-process training data 
xtr = dat.x;
rtr = dat.r;
ntr = length(rtr);
[xid,~,jj] = unique(xtr,'rows');  % find unique stimuli
Mid = sparse(1:ntr,jj,1); % matrix mapping unique stimuli to original stimuli
rtrid = Mid'*rtr;  % combined spike counts for each unique stimulus
xct = full(sum(Mid)'); % number of times each stimulus presented
nxuniq = length(xid);

if nargin < 5
    phi_init = randn(nxuniq,1)*.01;  % initial value of function
else
    [~,iiphi] = intersect(xx,xid,'rows');
    phi_init = phi0(iiphi);
end

% Make prior covariance kernel
K = mkKernelMatrix_RBF(theta,xid); 

% Test if K is ill-conditioned & project to reduced space if so
if rcond(K)< (1/condthresh);
    isILLCOND = true; 
    [U,S] = svd(K); % compute SVD
    S = diag(S); % eigenvalues
    iikp = S>S(1)/condthresh; % keep only these eigenvalues
    nkp = sum(iikp); % number of eigenvalues to keep
    Uproj = U(:,iikp); % projection matrix for GP params
    Kinv = spdiags(1./S(iikp),0,nkp,nkp); % inverse cov matrix in reduced space
    mu = Uproj'*ones(nxuniq,1)*theta.mu; % mean of GP in reduced space 
    Bproj = Uproj; % projection from optimization params to phi values * count
    phi_init = Uproj'*phi_init;
else
    isILLCOND = false; % not ill-conditioned
    Kinv = inv(K); % inverse prior covariance matrix
    mu = theta.mu; % mean of GP
    Bproj = speye(nxuniq); % count how many times each stimulus presented
end

% Set up optimization of log-posterior
lossfun = @(prs)(neglogpost_GPtf(prs,Bproj,rtrid,xct,gfun,mu,Kinv)); % loss function 
opts = optimset('display', 'off','gradobj','on','Hessian','on','algorithm','trust-region');

% Run optimization (only need stimuli where we have observed data)
[phimap_xid,~,flag,~,~,H_xid] = fminunc(lossfun,phi_init,opts);
if flag < 0
    warning('GPtf: posterior optimization did not converge to optimum');
end

% Compute posterior over phi at all points in xx
Kstr = mkKernelMatrix_RBF(theta,xx,xid);
Kstrstr = mkKernelMatrix_RBF(theta,xx);
if isILLCOND
    K = spdiags(S(iikp),0,nkp,nkp); % kernel matrix in reduced space
    L = H_xid-Kinv; % Hessian just from log-likelihood
    phimap = theta.mu + Kstr*Uproj*Kinv*(phimap_xid-mu); % find GP mode in phi
    phicov = Kstrstr - Kstr*Uproj*((L*K+speye(nkp))\L)*Uproj'*Kstr';
else
    L = H_xid-Kinv; % Hessian just from log-likelihood
    phimap = theta.mu + Kstr*Kinv*(phimap_xid-mu); % find GP mode in phi
    phicov = Kstrstr - Kstr*((L*K+speye(nxuniq))\L)*Kstr';
end
[fmap,df] = gfun(phimap);
fstd = sqrt(diag(phicov)).*df;

% Compute parameters for GP posterior in phi space at unique points xid
if nargout > 4
    gpstat.xid = xid;
    if isILLCOND
        gpstat.mu = Uproj*phimap_xid;
        gpstat.cov = Uproj*((H_xid)\Uproj');
    else
        gpstat.mu = phimap_xid;
        gpstat.cov = inv(H_xid);
    end
end