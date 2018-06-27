% demo1_parametricTC_1Dinfomax.m
%
% Demo script illustrating a short run of infomax stimulus selection while using MCMC
% (slice sampling) to sample the posterior over parameters and evaluate
% utility function (which here is mutual information) after each stimulus.
% 
% Makes plots found in FIG 2 of [Pillow & Park 2016].

setpaths; % initialize path to include correct directories

%%  Set true tuning curve for simulated neuron

% grid over stimulus values "x"
xlims = [-10, 10]; % range of the stimulus space
dx = .05; % spacing of stimulus grid
xx = (xlims(1)+dx/2:dx:xlims(2))'; % grid of x points
xrnge = diff(xlims); % stimulus range
nx = length(xx); % number of stimulus grid points

% parameters of parametric TC (here an unnormalized Gaussian plus a
% baseline, but you can write your own parametric function instead!).
mutc = 3.5;  % tuning curve mean
sigtc = 1;  % tuning curve width
Amax = log(50);  % tuning curve amplitude
bl = 2;  % baseline firing rate
theta_true = [mutc,sigtc,Amax,bl];  % the true TC parameters

% make tuning curve
ftc0 = @(x,prs)(exp(prs(3)-(x-prs(1)).^2/(2*prs(2).^2))+prs(4)); % generic TC function
ftctrue = @(x)(ftc0(x,theta_true));  % true TC function
tctrue = ftctrue(xx);  % true TC evaluated on grid
clf;subplot(221); plot(xx,tctrue);
xlabel('stimulus'); ylabel('spike rate (sp/s)'); 

% run short random experiment for initialization
ninit = 3; % number of randomly selected stimuli to use for initialization
ntrials = 10; % number of adaptive trials to run for (after the initial ones)
xtr = rand(ninit,1)*xrnge+xlims(1); % some initial stimuli, randomly selected
ytr = poissrnd(ftctrue(xtr)); % response of neuron to these stimuli

% If desired, plot initial TC and responses
clf; plot(xx,ftctrue(xx), xtr,ytr,'ko', 'linewidth', 2);
xlabel('stimulus value'); ylabel('spike count'); 
title('true tuning curve and initial data');
legend('true tuning curve', 'data');


%% Set up slice sampling (MCMC) of posterior over parameters

% Set parameter ranges
murnge = xlims*.98; % range for mu parameter (mean of Gaussian bump)
sigrnge = [.1 20]; % range for sigma parameter (width of Gaussian bump)
Arnge = log([1 200]); % range for log amplitude of TC
brnge = [.01 50]; % range for baseline of TC

% Set bounds and options
LB = [murnge(1); sigrnge(1); Arnge(1); brnge(1)]; % lower bounds
UB = [murnge(2); sigrnge(2); Arnge(2); brnge(2)]; % upper bounds
prs0 = (LB+UB)/2; % initial parameters

nslice = 100;  % # samples of slice sampling to use for each trial
nburn = 100;  % # samples to use for "burn in" on each trial

% sample posterior to get a good initialization point for MCMC 
flogpdf = @(prs)logliPoissonTCbd(prs,xtr,ytr,LB,UB); % log-likelihood function
thetasamps = mean(slicesample(prs0',50,'logpdf',flogpdf,'burnin', 1000)); % mean of posterior samples


%% Run adaptive stimulus selection algorithm

% initialize plotting and variables to store
trialNumsToPlot = [1,2,3,8]; % generate plots of TC estimate for each of these #s
npl = 4; % total number of plots
jplot=1; % index variable for plots
colr = [.8 .8 .8]; % color for plotting posterior samples of TC
Errs = zeros(ntrials,2); % stores errors in posterior mean over parameters TC and posterior mean over TC

% main FOR loop
for jj = 1:max(trialNumsToPlot)

    % set log pdf using all available data collected so far
    flogpdf = @(prs)logliPoissonTCbd(prs,xtr,ytr,LB,UB);  % log-likelihood function
    thetasamps = slicesample(thetasamps(end,:),nslice,'logpdf',flogpdf,'burnin',nburn); % sample posterior & compute mean
    thetamu = mean(thetasamps)'; % mean of posterior over parameters (from samples)

    % compute posterior mean over TC
    % (note we could do this more efficiently WITHOUT a for loop!)
    TCmu = zeros(nx,1); 
    for j = 1:nslice 
        tc = ftc0(xx,thetasamps(j,:));  % TC for the j'th sample
        TCmu = TCmu+tc/nslice; % posterior mean of TC
    end

    %% Compute predictive distribution of response  P(r)
    rmax = ceil(max(TCmu)+sqrt(max(TCmu))*4); % max expected spike count given TC estimate
    rr = (0:rmax)'; % grid over integer spike counts
    prr = zeros(rmax+1,nx,nslice); % probability over spike counts, for each stim, for each sample
    for j = 1:nslice % loop over samples 
        tc = (ftc0(xx,thetasamps(j,:))); % TC for the j'th sample
        prr(:,:,j) = exp(bsxfun(@plus,bsxfun(@plus,rr*log(tc)',...
            -tc'), -gammaln(rr+1))); % Poisson probability over spike count for this sample, for each stimulus
    end
    
    Pr = mean(prr,3);  % marginal response distribution (aka "posterior predictive")
    Hr = -sum(Pr.*log2(Pr))'; % marginal entropy of response H(r)
    prr = max(prr,1e-6); % make sure probabilities >= 0
    Hrtheta = -mean(sum(prr.*log2(prr),1),3)'; % conditional entropy H(r|theta)
    MI = Hr-Hrtheta; % Mutual information between parameters and response, for each candidate stimulus

    % Select optimal (infomax) stimulus
    [xnext,idx] = argmax(MI,xx);  % select stimulus with maximal MI
    
    % Present selected stimulus to (simulated) neuron
    ynext = poissrnd(ftctrue(xnext)); % sample response to this stimulus under TC-Poisson model
    
    % add new (stimulus,response) pair to the dataset
    xtr(end+1) = xnext;
    ytr(end+1) = ynext;

    
    %% ==== Optional stuff for plotting and tracking error =========
    
    if jj==trialNumsToPlot(jplot) % Plot posterior over TC at beginning of this trial

        subplot(2,npl,jplot); % ========= top plot ======
        for j = 1:nslice
            tc = ftc0(xx,thetasamps(j,:)); % tuning curve for this sample
            plot(xx,tc,'color', colr,'linewidth', 2); hold on; % plot it
        end
        h = plot(xx,ftctrue(xx),'k',xx,TCmu,'r',xtr(1:end-1),ytr(1:end-1),'k.'); hold off;
        set(h(3:end),'markersize',16); set(h,'linewidth',2);
        set(gca,'ylim', [0 75],'xticklabel',[],'tickdir','out'); box off;
        title(sprintf('trial %d', jj+ninit-1)); 
        
        if jplot==1
            legend(h(1:2),{'true f', 'estimate'},'location','northwest');
            ylabel('spike rate (sp/s)');
        else
            set(gca,'yticklabel',[]);
        end
        
        subplot(2,npl,npl+jplot); % ====== bottom plot =============
        h=plot(xx,MI,xnext,MI(idx),'k*');
        set(h(1),'linewidth',2);
        xlabel('stimulus');box off;
        set(gca,'ylim',[0 2],'tickdir','out');

        if jplot==1
            ylabel('mutual information (bits)');
            legend('MI', 'selected stimulus', 'location', 'northwest');
        else 
            set(gca,'yticklabel',[]); 
        end
        
        drawnow;  
        jplot=jplot+1; % increment # of plots so far
    end
        
    % Compute errors
    Errs(jj,1) = norm(tctrue-ftc0(xx,thetamu)); % error in estimate from posterior mean of parameters
    Errs(jj,2) = norm(tctrue-TCmu);  % error in estimate from posterior mean of TC
    fprintf('Error (trial %2d):  mean-params=%7.2f  mean-TC=%7.2f\n', jj+ninit-1, Errs(jj,:));
            
    
end
