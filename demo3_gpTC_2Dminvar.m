% demo2_gpTCinference_2D.m
%
% Demo script illustrating active learning for a non-parametric 2D
% tuning curve parametrized by a transformed GP prior and Poisson spiking,
% with stimulus selected to minimize posterior variance of the TC.

% Set parameters for true FR map by sampling from GP
nx1 = 40; % # grid points in first dimension of stimulus space
nx2 = 40; % # grid pionts in 2nd dim of stimulus space
[x1,x2] = meshgrid(1:nx1,1:nx2); % grid of x1 and x2 points
xx = [x1(:),x2(:)];
nx = nx1*nx2; % total number of grid points

% set params for kernel function
rho = 1.25^2; % marginal variance
d = 5; % length scale
K = mkKernelMatrix_RBF([rho,d],xx); % prior covariance matrix over TC
mu = 1; % mean of GP

% Sample the GP
gpsamp = mvnrnd(mu*ones(1,nx),K)'; % sample from GP to get pre-transformed TC

% Transform nonlinearity to get non-negative spike rates
ftruevec = exp(gpsamp); % vectorized tuning curve
ftrue = reshape(ftruevec,nx1,nx2); % reshaped as 2D image

% Plot true tuning curve
clf; 
subplot(221);
imagesc(1:nx1,1:nx2,ftrue); 
title('true tuning curve f(x)'); 
xlabel('stim axis 1'); ylabel('stim axis 2');
subplot(223); 
plot(1:nx1,ftrue'); 
title('horizontal slices of true f(x)');  xlabel('stim axis 1'); 
ylabel('firing rate (sp/s)'); 

%% observe some initial data points

ninit = 15; % number of random initial stimuli to show
iinit = randsample((1:nx)',ninit,true); % indices of (random) initial stimuli 
xinit = xx(iinit,:); % initial stimuli
rinit = poissrnd(ftruevec(iinit));  % spike responses

subplot(221); hold on;
plot(xinit(:,1), xinit(:,2), 'r.','linewidth',10);
hold off;


%% Find MAP estimate for f and given the initial data

% use true hyperparameters
theta.mu = mu; % mean
theta.rho = rho; % marginal variance
theta.d = d; % length scale

% Set nonlinearity 'g' for transforming GP to positive firing rates
gfun = @expfun;  % exponential
%gfun = @logexp1;  % soft-rectification

% Make struct for data
dat.x = xinit;
dat.r = rinit;

% compute MAP estimate and posterior covariance given data & prior
[fmap,fstd,phimu,phicov] = runMAPinference_GPtf(xx,dat,gfun,theta); 

% Make plot
fmapim = reshape(fmap,nx1,nx2);
subplot(222);
imagesc(1:nx1,1:nx2,fmapim);
hold on;
plot(dat.x(:,1),dat.x(:,2),'r.', 'linewidth', 10);
hold off;
title('f_{map}(x)')
subplot(223);
plot(1:nx1,ftrue','b',1:nx1,fmapim','r--'); 
title('true f(x) vs. f_{map}(x)')
xlabel('stim axis 1');
subplot(224);
imagesc(reshape(fstd,nx1,nx2)); 
title('posterior std over f(x) given initial data');


%% Add datapoints 1 at a time using "uncertainty sampling"

Ntrials = 200;
for jj = 1:Ntrials
    
    % stimulus for which (approximate) posterior variance over f is maximal
    [~,idxnext] = max(fstd); % index of stimulus with maximal posterior firing rate std
    xnext = xx(idxnext,:); % stimulus to show next
    fprintf('Trial %d: stimulus index selected =%d\n',length(dat.r)+1,idxnext);
    
    % Present the stimulus
    rnext = poissrnd(ftruevec(idxnext)); % spike counts 

    % Add to dataset
    dat.x(end+1,:) = xnext;
    dat.r(end+1) = rnext;

    % compute MAP estimate and posterior covariance given data & prior
    [fmap,fstd,phimu,phicov] = runMAPinference_GPtf(xx,dat,gfun,theta,phimu);
    
    %  ===== make plots ===================================
    fmapim = reshape(fmap,nx1,nx2);
    subplot(222);
    imagesc(1:nx1,1:nx2,fmapim);
    xlabel('stim axis 1'); ylabel('stim axis 2');
    hold on;
    plot(dat.x(:,1),dat.x(:,2),'r.', 'linewidth', 10);
    hold off;
    title(sprintf('f_{map}(x) after %d trials', length(dat.r)));
    subplot(223);
    plot(1:nx1,ftrue','b',1:nx1,fmapim','r--');
    title('slices of true f(x) (blue) vs. f_{map}(x) (red)')
    xlabel('stim axis 1'); ylabel('firing rate (sp/s)');
    subplot(224);
    imagesc(reshape(fstd,nx1,nx2));
    title('posterior std over f');
    xlabel('stim axis 1'); ylabel('stim axis 2');
    drawnow;
    %  =========================================================

end

