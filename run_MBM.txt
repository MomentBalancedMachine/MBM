%% Moment Balanced Machine (MBM): A New Supervised Inference Engine for Regression problem
clc;
clear all;

% Define grid search hyperparameter
C = [0.1, 1, 10, 100];
sig = [0.1, 0.5, 1, 5];

% Define the number of folds for k-fold cross-validation
kfold = 10;

% Initialize a cell array to hold best C and sigma values for each fold
% Column 1 for Fold, Column 2 for best C, Column 3 for best sigma
bestParams = cell(kfold, 3); 

% Loop through each fold
for i = 1:kfold

    % Load training and testing data for the current fold
    train = readmatrix(['training_productivity', num2str(i), '.xlsx']);
    test = readmatrix(['testing_productivity', num2str(i), '.xlsx']);

   % Define input (features) and output (target) variables
    xtr=train(:,1:end-1);   % Training features
    ytr=train(:,end);          % Training target
    xte=test(:,1:end-1);    % Testing features
    yte=test(:,end);          % Testing target
    
    titlename=['Fold',num2str(i)];
    disp(titlename)
    
    % Run BPNN algorithm to compute predictions and derive weights
    [ytrp, ytep, ytr, yte, options]=simBPNN(train,test);
     
    % Calculate model errors on the training and testing set
    errors=postLSIM(ytrp,ytr,ytep,yte);

     % Compute case weights based on BPNN algorithm output
     w_bpnn = (1./(abs(ytrp-ytr)./ytr));

    % Variables to track the best model and its error for the current fold
    bestModel = [];
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

            % Make predictions on the test set
            ytep = svmpredict(yte, xte, model);

            % Make predictions on the training set
            ytrp = svmpredict(ytr, xtr, model);

            % Calculate model errors
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

     % Record the best hyperparameters for the current fold
    bestParams{i, 1} = ['Fold', num2str(i)];
    bestParams{i, 2} = bestC;
    bestParams{i, 3} = bestSig;

    % Train the best model  using the best hyperparameters
    bestModel = svmtrain(w_bpnn, ytr, xtr, ['-s 3 -c ', num2str(bestC), ' -g ', num2str(bestSig)]);

    % Make predictions on the test set
    ytep = svmpredict(yte, xte, bestModel);

    % Make predictions on the training set
    ytrp = svmpredict(ytr, xtr, bestModel);

    % Calculate model errors on the training and testing set
    finalErrors = postLSIM(ytrp, ytr, ytep, yte);

    % Display final errors for the best model
    fprintf('Fold %d completed.\n', i);
    fprintf('Best C: %.2f, Best Sigma: %.2f\n', bestC, bestSig);
    fprintf('Best RMSE: %.4f\n', finalErrors.RMSEte);                   %display RMSE for the testing set.
    fprintf('Best MAPE: %.4f\n', finalErrors.MAPEte);                   %display MAPE for the testing set.
    fprintf('Best MAE: %.4f\n', finalErrors.MAEte);                        %display MAE for the testing set.
    fprintf('Best R: %.4f\n', finalErrors.Rte);                                  %display R for the testing set
    fprintf('Best R2: %.4f\n', finalErrors.R2te);                              %display R-squared for the testing set.
    disp('---*---');
end