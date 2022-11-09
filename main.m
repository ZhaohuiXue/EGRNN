%% Setup
clear;clc;close all;

%% Load test data for regression from Matlab R2022a
load accidents % 
X = hwydata; 
X(:,4) = []; %Use all other 16 variables as the input feature (more details can be found in hwyheaders)
Y = hwydata(:,4); %Use traffic fatalities as the output response

%   X: Input data of size n x d
%   Y: Output/target/observation of size n x do
%   n: number of samples/examples/patterns (in rows)
%   d: input data dimensionality/features (in columns)
%   do: output data dimensionality (variables, observations).
%% Main loop
%METHODS = {'TREE','NN','RF','EGRNN'}
METHODS = {'EGRNN'} % add more other methods for comparison
ALL_independent_runs = 10
Ratio_for_train = 0.5
%% TRAIN ALL MODELS
numModels = numel(METHODS);

for m=1:numModels
    %fprintf(['Training ' METHODS{m} '... \n'])
    
    RMSE_ALL = [];
    R_ALL = [];
    TIME_ALL = [];
    for i_run = 1:ALL_independent_runs
        t=cputime;
        %% Split training-testing data
        rate = Ratio_for_train; %[0.05 0.1 0.2 0.3 0.4 0.5 0.6]
        % Fix seed random generator (important: disable when doing the 100 realizations loop!)
        % rand('seed',12345);
        %randn('seed',12345);
        % rng(0);
        [n d] = size(X);                 % samples x bands
        r = randperm(n);                 % random index
        ntrain = round(rate*n);          % #training samples
        Xtrain = X(r(1:ntrain),:);       % training set
        Ytrain = Y(r(1:ntrain),:);       % observed training variable
        Xtest  = X(r(ntrain+1:end),:);   % test set
        Ytest  = Y(r(ntrain+1:end),:);   % observed test variable
        [ntest do] = size(Ytest);

        if strcmp(METHODS{m},'EGRNN')
            Xtrain = Xtrain';
            Ytrain = Ytrain';
            Xtest = Xtest';
            
            % runEGRNN(input_train, output_train, input_test, knn_size,flag_norm)
            % Train the model, knn_size=7 needs to be optimized for your data set
            eval(['Yp = run' METHODS{m} '(Xtrain,Ytrain,Xtest,7);']);
        else
            eval(['model = train' METHODS{m} '(Xtrain,Ytrain);']); % Train other model
            eval(['Yp = test' METHODS{m} '(model,Xtest);']);       % Test other model
        end

        RESULTS.RMSE = sqrt(mean((Ytest-Yp).^2));
        [rr, pp]   = corr(Ytest, Yp);
        RESULTS.R  = rr;

        CPUTIMES = cputime - t;
        %MODELS = model;
        %YPREDS = Yp;
        % show the results
        fprintf('%s%s%d%s%0.3f%s%0.3f%s%0.3f\n',METHODS{m},' run ',i_run,' RMSE = ',RESULTS.RMSE,' R = ',RESULTS.R,' Time = ',CPUTIMES);
        %fprintf('%s%s%d%s%0.3f\n',METHODS{m},' run ',i_run,' R = ',RESULTS.R);

        RMSE_ALL = [RMSE_ALL RESULTS.RMSE];
        R_ALL = [R_ALL RESULTS.R];
        TIME_ALL = [TIME_ALL CPUTIMES];
    end
    fprintf('%s%s%0.3f%s%0.3f%s%0.3f\n',METHODS{m},' Mean RMSE = ',mean(RMSE_ALL),' Mean R = ',mean(R_ALL),' Mean Time = ',mean(TIME_ALL));
    fprintf('%s%s%0.3f%s%0.3f%s%0.3f\n',METHODS{m},' STD RMSE = ',std(RMSE_ALL),' STD R = ',std(R_ALL),' Mean Time = ',mean(TIME_ALL));
    %fprintf('%s%s%0.3f\n',METHODS{m},' Mean R = ',mean(R_ALL));
end





