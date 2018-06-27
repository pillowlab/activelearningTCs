% demo2_gpTC_1Dminvar.m
%
% Demo script illustrating active learning for a non-parametric 1D
% tuning curve parametrized by a transformed Gaussian Process (GP) prior
% and Poisson spiking, with stimulus selected to minimize posterior
% variance of the TC

setpaths; % set paths to include needed subdirectory

% Set parameters for true TF by sampling from Gaussian process (GP)
nx = 500; % # grid points in input space
xx = (1:nx)'; % grid of input points

% set params for kernel function
rho = 25^2; % marginal variance
d = 25; % length scale
K = mkKernelMatrix_RBF([rho,d],xx); % kernel matrix (prior covariance over function)

% Generate true tuning function
mu = 2; % mean 
gpsamp = mvnrnd(mu*ones(1,nx),K)'; % sample from GP

% Transform by nonlinearity so TC is positive (2 standard choices here)
%ftrue = exp(gpsamp); % transform by exponential
ftrue = log(1+exp(gpsamp)); % transform by soft-rectification or "soft-plus"

% plot true TC
clf; subplot(221);
imagesc(K); title('prior covariance over \phi(x)');
subplot(223);
plot(1:nx,ftrue','k', 'linewidth', 2); 
title('true tuning curve f(x)');
xlabel('stimulus x'); ylabel('sp/s');

%% generate some initial stimulus-response observations

ninit = 10; % number of initial points to generate
iinit = randsample((1:nx)',ninit,true); % indices of (random) initial stimuli 
xinit = xx(iinit); % initial stimuli
xinit = sort(xinit);  % sort them (purely for convenience)
rinit = poissrnd(ftrue(xinit)); % spike counts 

% Add these initial datapoints to plot
subplot(223); hold on;
stem(xinit,rinit,'r','linewidth',1);
hold off;

%% Find MAP estimate for f and given the initial data

% set hyperparameters to true hyperparameters
theta.mu = mu; % mean
theta.rho = rho; % marginal variance
theta.d = d; % length scale

% Set nonlinearity 'g' for transforming GP to positive firing rates
%gfun = @expfun;  % exponential
gfun = @logexp1;  % soft-rectification

% Make struct for data
dat.x = xinit;
dat.r = rinit;

% compute MAP estimate and posterior covariance given data & prior
[fmap,fstd,phimu,phicov] = runMAPinference_GPtf(xx,dat,gfun,theta); 

% make plot
subplot(223); % --- tuning curve -------
errorbar(1:nx,fmap,fstd,'color', .9*[1 1 1]); hold on;
h = plot(1:nx,ftrue,'k',1:nx,fmap,xinit,rinit,'ro'); 
legend(h,'f_{true}', 'f_{MAP}','data');
hold off; axis tight;
title('tuning curve f(x)');
ylabel('sp/s'); xlabel('x');
subplot(222); % --- phi covariance ------
imagesc(phicov);
title('posterior cov over \phi(x)');
subplot(224); % --- phi mean  ----
plot(1:nx,phimu);
title('pre-transformed TC \phi(x)'); 
xlabel('x');


%% Select stimuli using uncertainty sampling (maximal variance of the posterior over TC)

Ntrials = 50;
for jj = 1:Ntrials
    
    % Select stimulus for which (approximate) posterior variance over f is maximal
    [~,idxnext] = max(fstd);  % index of stimulus with  maximal posterior firing rate std
    xnext = xx(idxnext); % stimulus value to show next
        
    % Present the selected stimulus to (simulated) neuron
    rnext = poissrnd(ftrue(xnext)); % sample spike count to this stimulus

    % Add to dataset
    dat.x(end+1,:) = xnext;
    dat.r(end+1) = rnext;

    % compute MAP estimate and posterior covariance given data & prior
    [fmap,fstd,phimu,phicov] = runMAPinference_GPtf(xx,dat,gfun,theta,phimu);

    %  ===== make plots ===================================
        subplot(221); % ----- posterior stdev -------
    lw = 2; % line width for plots
    plot(1:nx,fstd,'linewidth',lw);
    title('posterior std over TC');
    subplot(223); % --- tuning curve -------
    errorbar(1:nx,fmap,fstd,'color', .9*[1 1 1]); hold on;
    h = plot(1:nx,ftrue,'k',1:nx,fmap,dat.x,dat.r,'ro','linewidth',lw);
    legend(h,'f_{true}', 'f_{MAP}','data');
    box off; hold off; axis tight;
    title(sprintf('MAP estimate of TC (%d trials)', length(dat.r)));
    ylabel('sp/s'); xlabel('stimulus');
    subplot(222); % --- phi (pre-transformed tuning curve) covariance ------
    imagesc(phicov);
    title('posterior cov of pre-transformed TC \phi(x)');
    subplot(224); % --- phi mean  ----
    plot(1:nx,phimu,'linewidth', lw);
    title('MAP estimate of pre-transformed TC \phi(x)');
    xlabel('stimulus');drawnow;
    
    fprintf('Trial %d: presented stimulus x=%6.1f\n',length(dat.r)+1,xnext);
    %  =========================================================

end

