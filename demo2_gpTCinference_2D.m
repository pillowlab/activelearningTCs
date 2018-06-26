% demo2_gpTCinference_2D.m
%
% Test script illustrating posterior inference for a nonparametric 2D
% tuning curve parametrized by a transformed GP prior and Poisson spiking 

% Set parameters for true FR map by sampling from GP
nx1 = 40; % # grid points in first dimension of input space
nx2 = 40; % # grid pionts in 2nd dim of input space
[x1,x2] = meshgrid(1:nx1,1:nx2); % x and t coordinates
xx = [x1(:),x2(:)];
nx = nx1*nx2; % total number of grid points

% set params for kernel function
rho = 1.25^2; % marginal variance
d = 5; % length scale
K = mkKernelMatrix_RBF([rho,d],xx); % kernel matrix
mu = 1; % mean of GP

% Sample true FRmap
%ftruevec = log(1+exp(mvnrnd(mu*ones(1,nx),K)))';
ftruevec = exp(mvnrnd(mu*ones(1,nx),K))';
ftrue = reshape(ftruevec,nx1,nx2);

% Plot
clf; subplot(221);
imagesc(1:nx1,1:nx2,ftrue);
title('true f(x)'); xlabel('x_1'); ylabel('x_2');
subplot(223); plot(1:nx1,ftrue'); 
title('horizontal slices of true f(x)');  xlabel('x_1'); 

%% observe some initial data points

ninit = 15;
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
title('f_map(x)')
subplot(223);
plot(1:nx1,ftrue','b',1:nx1,fmapim','r--'); 
title('f(x) vs. f_map(x)')
subplot(224);
imagesc(reshape(fstd,nx1,nx2)); 
title('posterior std over f');


%% Add datapoints 1 at a time using "uncertainty sampling"

Ntrials = 200;
for jj = 1:Ntrials
    
    % stimulus for which (approximate) posterior variance over f is maximal
    [~,idxnext] = max(fstd); % index of stimulus with maximal posterior firing rate std
    xnext = xx(idxnext,:); % stimulus to show next
    fprintf('Trial %d: stimulus=%d\n',length(dat.r)+1,idxnext);
    
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
    hold on;
    plot(dat.x(:,1),dat.x(:,2),'r.', 'linewidth', 10);
    hold off;
    title(sprintf('f_{map}(x) after %d trials', length(dat.r)));
    subplot(223);
    plot(1:nx1,ftrue','b',1:nx1,fmapim','r--');
    title('f(x) vs. f_map(x)')
    subplot(224);
    imagesc(reshape(fstd,nx1,nx2));
    title('posterior std over f');
    drawnow;
    %  =========================================================

end

