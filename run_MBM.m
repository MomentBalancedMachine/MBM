%% Moment Balanced Machine (MBM): A New Supervised Inference Engine for Regression problem
clc;
clear all;

% Define grid search hyperparameter
C = [0.1, 1, 10, 100];
sig = [0.1, 0.5, 1, 5];

% Number of folds
kfold = 1;

% Initialize a cell array to hold best C and sigma values for each fold
bestParams = cell(kfold, 3); % Column 1 for Fold, Column 2 for best C, Column 3 for best sigma

% Loop through each fold
for i = 1:kfold
    tic;
    t1=cputime();

    %input training and testing data
    train = readmatrix(['training_productivity', num2str(i), '.xlsx']);
    test = readmatrix(['testing_productivity', num2str(i), '.xlsx']);

   %define input and output variable
    xtr=train(:,1:end-1);
    ytr=train(:,end);
    xte=test(:,1:end-1);
    yte=test(:,end);
    
    titlename=['Fold',num2str(i)];
    disp(titlename)
    
    %running BPNN algortihm 
    [ytrp, ytep, ytr, yte, options]=simBPNN(train,test);
        
    errors=postLSIM(ytrp,ytr,ytep,yte);
    timeElapsed = toc

     % Compute weights based on BPNN algortihm 
     w_bpnn = (1./(abs(ytrp-ytr)./ytr));

    bestModel = [];
    bestError = inf;
    bestC = NaN;
    bestSig = NaN;

    % Initialize a matrix to store C, Sigma, and RMSE for each combination
    resultsMatrix = [];

    % MBM training model using grid search
    for j = 1:length(C)
        for k = 1:length(sig)
            model = svmtrain(w_bpnn, ytr, xtr, ['-s 3 -c ', num2str(C(j)), ' -g ', num2str(sig(k))]);
            ytep = svmpredict(yte, xte, model);
            ytrp = svmpredict(ytr, xtr, model);
            errors = postLSIM(ytrp, ytr, ytep, yte);

            % Store C, Sigma, and RMSE in the results matrix
            resultsMatrix = [resultsMatrix; C(j), sig(k), errors.RMSEte];

            if errors.RMSEte < bestError
                bestError = errors.RMSEte;
                bestModel = model;
                bestC = C(j);
                bestSig = sig(k);
            end
        end
    end

    % Record best parameters for this fold
    bestParams{i, 1} = ['Fold', num2str(i)];
    bestParams{i, 2} = bestC;
    bestParams{i, 3} = bestSig;

    % Train the best model with identified best parameters
    bestModel = svmtrain(w_bpnn, ytr, xtr, ['-s 3 -c ', num2str(bestC), ' -g ', num2str(bestSig)]);
    ytep = svmpredict(yte, xte, bestModel);
    ytrp = svmpredict(ytr, xtr, bestModel);
    finalErrors = postLSIM(ytrp, ytr, ytep, yte);

    % Record and display the elapsed time for this fold
    timeElapsed = cputime() - t1;
    fprintf('Fold %d completed.\n', i);
    fprintf('Best C: %.2f, Best Sigma: %.2f\n', bestC, bestSig);
    fprintf('Best RMSE: %.4f\n', finalErrors.RMSEte);
    fprintf('Best MAPE: %.4f\n', finalErrors.MAPEte);
    fprintf('Best MAE: %.4f\n', finalErrors.MAEte);
    fprintf('Best R: %.4f\n', finalErrors.Rte);
    fprintf('Best R2: %.4f\n', finalErrors.R2te);
    fprintf('Computational time: %.3f seconds.\n', timeElapsed);
    disp('---*---');
end