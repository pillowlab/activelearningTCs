% demo1_parametricTC_1Dinfomax.m
%
% Demo script illustrating a short run of infomax stimulus selection while using MCMC
% (slice sampling) to sample the posterior over parameters and evaluate
% utility function (which here is mutual information) after each stimulus.
% 
% Makes plots found in FIG 2 of [Pillow & Park 2016].


%%  Set true tuning curve

% grid over stimulus values "x"
xlims = [-10, 10]; % limits of left and right edge
dx = .05; % spacing x grid
xx = (xlims(1)+dx/2:dx:xlims(2))'; % grid of x points
xrnge = diff(xlims); % stimulus range
nx = length(xx); % number of stimulus grid points

% parameters of parametric TC (here an unnormalized Gaussian plus a
% baseline, but you can write your own parametric function instead).
mutc = 3.5;  % tuning curve mean
sigtc = 1;  % tuning curve width
Amax = log(50);  % tuning curve amplitude
bl = 2;  % baseline firing rate

% make tuning curve
ftc0 = @(x,prs)(exp(prs(3)-(x-prs(1)).^2/(2*prs(2).^2))+prs(4)); % generic TC function
ftc = @(x)(ftc0(x,[mutc,sigtc,Amax,bl]));  % true TC function
tctrue = ftc(xx);  % true TC evaluated on grid
clf;subplot(221); plot(xx,tctrue);
xlabel('stimulus value'); ylabel('spike rate (sp/s)'); 

% run short random experiment for initialization
ntrain = 3;
xtr = (rand(ntrain,1)*xrnge+xlims(1)); % some initial stimuli, randomly selected
ytr = poissrnd(ftc(xtr)); % response of neuron to these stimuli
clf; subplot(221); plot(xx,ftc(xx), xtr,ytr,'ro');
xlabel('stimulus value'); ylabel('spike count'); 
title('true tuning curve and initial data');

% for plotting
colr = [.8 .8 .8];


%% Set params for slice sampling (MCMC) of posterior over parameters

% Set parameter ranges
murnge = xlims*.98; % range for mu parameter
sigrnge = [.1 20]; % range for sigma parameter
Arnge = log([1 200]); % range for log amplitude of TC
brnge = [.01 50]; % range for baseline of TC

% Set bounds and options
LB = [murnge(1); sigrnge(1); Arnge(1); brnge(1)];
UB = [murnge(2); sigrnge(2); Arnge(2); brnge(2)];
prs0 = (LB+UB)/2;

ns = 100;  % # samples to use
nburn = 100;  % # samples to burn in

% Function pointer for log-likelihood function
flogpdf = @(prs)logliPoissonTCbd(prs,xtr,ytr,LB,UB);

% run initial sample to get good initialization point
psamps = mean(slicesample(prs0',50,'logpdf',flogpdf,'burnin', 1000));


%% Run algorithm
nTrials = 10; % number of total trials to run for (after the 3 initial ones above)
trialNumsToPlot = [1,2,3,8]; % generate plots of TC estimate for each of these #s
npl = 4; % number of plots
jplot=1; % index variable for plots
Errs = zeros(nTrials,2); % stores errors in posterior mean over parameters TC and posterior mean over TC
for jj = 1:max(trialNumsToPlot)

    % set log pdf using all available data collected so far
    flogpdf = @(prs)logliPoissonTCbd(prs,xtr,ytr,LB,UB); 
    psamps = slicesample(psamps(end,:),ns,'logpdf',flogpdf,'burnin',nburn); % sample posterior over params
    prsmu = mean(psamps)'; % mean of posterior over parameters (from samples)

    % compute posterior mean over TC
    % (note we could do this more efficiently WITHOUT a for loop!)
    muTC = zeros(nx,1); 
    for j = 1:ns 
        tc = ftc0(xx,psamps(j,:));  % TC for the j'th sample
        muTC = muTC+tc/ns; % posterior mean of TC
    end

    %% Compute predictive distributions
    rmax = ceil(max(muTC)+sqrt(max(muTC))*4); % compute max expected spike count
    rr = (0:rmax)'; % grid over spike counts
    prr = zeros(rmax+1,nx,ns); % probability over spike counts, for each stim, for each sample
    for j = 1:ns % loop over samples 
        tc = (ftc0(xx,psamps(j,:))); % TC for the j'th sample
        prr(:,:,j) = exp(bsxfun(@plus,bsxfun(@plus,rr*log(tc)',...
            -tc'), -gammaln(rr+1))); % Poisson probability over spike count for this sample, for each stimulus
    end
    
    
    Pr = mean(prr,3);  % marginal response distribution (aka "posterior predictive")
    Hr = -sum(Pr.*log(Pr))'; % marginal entropy of response H(r)
    prr = max(prr,.00001); 
    Hrtheta = -mean(sum(prr.*log(prr),1),3)'; % conditional entropy H(r|theta)
    MI = Hr-Hrtheta; % Mutual information between stimulus and parameters
    if any(isnan(MI)) | any(isinf(MI))
        keyboard; % for debugging purposes (please report errors to pillow@princeton.edu)
    end

    % Select and present new stimulus
    [xnext,idx] = argmax(MI,xx);  % select stimulus with maximal MI
    ynext = poissrnd(ftc(xnext)); % sample response to this stimulus under TC-Poisson model

    %% ==== Make plots % ==============================
    
    % Plot posterior over TC
    if jj==trialNumsToPlot(jplot)
        
        % Plot TC samples 
        subplot(2,npl,jplot);
        for j = 1:ns
            tc = ftc0(xx,psamps(j,:));
            plot(xx,tc,'color', colr,'linewidth', 2); hold on;
        end
        h = plot(xx,ftc(xx),'k',xx,muTC,'r',xtr,ytr,'k.');
        hold off;
        set(h(3:end),'markersize',14); set(h,'linewidth',2);
        set(gca,'ylim', [0 75]);
        if jplot==1
            legend(h(1:2),{'true f', 'estimate'},'location','northwest');
            ylabel('spike rate (sp/s)');
        else
            set(gca,'yticklabel',[]);
        end
        set(gca,'xticklabel',[]);set(gca,'tickdir','out');
        title(sprintf('trial %d', jj+ntrain-1)); box off;
        
        subplot(2,npl,npl+jplot); % =====================
        h=plot(xx,MI,xnext,MI(idx),'k*');
        set(h(1),'linewidth',2);
        xlabel('x');box off;

        % Compute errors
        Errs(jj,1) = norm(tctrue-ftc0(xx,prsmu)); % error in estimate from posterior mean of parameters
        Errs(jj,2) = norm(tctrue-muTC);  % error in estimate from posterior mean of TC
        fprintf('(trial %d) Errs:  mean-parameters:%.2f  mean-TC:%.2f\n', jj, Errs(jj,:));
        set(gca,'ylim',[0 2]);
        set(gca,'tickdir','out');
        if jplot==1
            ylabel('info gain (bits)');
        else
            set(gca,'yticklabel',[]);
        end
        jplot=jplot+1; % increment # of plots so far
        drawnow;  
    end
        
    % ======== end plotting code =======================
    
    % add new (stimulus,response) pair to the dataset
    xtr(end+1) = xnext;
    ytr(end+1) = ynext;
    
end
