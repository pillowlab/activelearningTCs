% demo2_gpTC_1Dminvar.m
%
% Demo script illustrating active learning for a non-parametric 1D
% tuning curve parametrized by a transformed GP prior and Poisson spiking,
% with stimulus selected to minimize posterior variance of the TC.

setpaths; % set paths to include needed subdirectory

% Set parameters for true TF by sampling from GP
nx = 500; % # grid points in input space
xx = (1:nx)';

% set params for kernel function
rho = 25^2; % marginal variance
d = 25; % length scale
K = mkKernelMatrix_RBF([rho,d],xx); % kernel matrix

% Generate true tuning function
mu = 2; % mean 
gpsamp = mvnrnd(mu*ones(1,nx),K)'; % sample from GP

%ftrue = exp(gpsamp); % transform by exponential
ftrue = log(1+exp(gpsamp)); % transform by soft-rectification

% plot
subplot(221);
imagesc(K); title('prior covariance over \phi(x)');
subplot(223);
plot(1:nx,ftrue','k'); 
title('true tuning curve f(x)');
xlabel('x'); ylabel('sp/s');

%% generate some initial stimulus-response observations

ninit = 10; % number of initial points to generate
iinit = randsample((1:nx)',ninit,true); % indices of (random) initial stimuli 
xinit = xx(iinit); % initial stimuli
xinit = sort(xinit); 
rinit = poissrnd(ftrue(xinit)); % spike counts 

subplot(223); hold on;
plot([xinit,xinit]',[zeros(ninit,1),rinit]', '-r', ...
    xinit, rinit, 'ro');
hold off;

%% Find MAP estimate for f and given the initial data

% use true hyperparameters
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
title('\phi(x)'); 
xlabel('x');


%% Add datapoints 1 at a time using "uncertainty sampling"

Ntrials = 50;
for jj = 1:Ntrials
    
    % Select stimulus for which (approximate) posterior variance over f is maximal
    [~,idxnext] = max(fstd);  % index of stimulus with  maximal posterior firing rate std
    xnext = xx(idxnext); % stimulus value to show next
    fprintf('Trial %d: stimulus=%d\n',length(dat.r)+1,xnext);
    
    % Present the selected stimulus
    rnext = poissrnd(ftrue(xnext)); % spike counts 

    % Add to dataset
    dat.x(end+1,:) = xnext;
    dat.r(end+1) = rnext;

    % compute MAP estimate and posterior covariance given data & prior
    [fmap,fstd,phimu,phicov] = runMAPinference_GPtf(xx,dat,gfun,theta,phimu);

    %  ===== make plots ===================================
    subplot(221); % ----- posterior stdev -------
    lw = 2; % line width for plots
    plot(1:nx,fstd,'linewidth',lw);
    title('posterior std over f');
    subplot(223); % --- tuning curve -------
    errorbar(1:nx,fmap,fstd,'color', .9*[1 1 1]); hold on;
    h = plot(1:nx,ftrue,'k',1:nx,fmap,dat.x,dat.r,'ro','linewidth',lw);
    legend(h,'f_{true}', 'f_{MAP}','data');
    box off; hold off; axis tight;
    title(sprintf('f_{map}(x) after %d trials', length(dat.r)));
    ylabel('sp/s'); xlabel('x');
    subplot(222); % --- phi (pre-transformed tuning curve) covariance ------
    imagesc(phicov);
    title('posterior cov of pre-transformed TC \phi(x)');
    subplot(224); % --- phi mean  ----
    plot(1:nx,phimu,'linewidth', lw);
    title('log-TC \phi(x)');
    xlabel('x');
    drawnow;
    %  =========================================================

end

