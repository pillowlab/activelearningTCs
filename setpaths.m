% setpaths.m
%
% Adds paths for Gaussian Process tuning function estimateion (GPtf)

if ~exist('expfun','file')   % check if 'expfun.m' is in path
    addpath nlfuns
end

if ~exist('runMAPinference_GPtf','file')   % check if 'expfun.m' is in path
    addpath code_gaussianprocessTCs
end

if ~exist('logliPoissonTCbd','file')   % check if 'expfun.m' is in path
    addpath code_parametricTCs
end
