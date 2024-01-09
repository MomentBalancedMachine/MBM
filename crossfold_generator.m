% Crossfold_generator - generates k-fold training and testing datasets
% This script reads data from an Excel file, divides it into k-folds, and 
% saves training and testing sets for each fold. It optionally normalizes the data.

clc;
clear all;

% Read data from the Excel file
data=readmatrix('productivity.xlsx');

% Specify the number of folds for cross-validation
kfold=10;

% Flag for applying normalization (1 for yes, 0 for no)
normalization=1;

% Generate indices for dividing the data into k-folds
foldnum=crossvalind('Kfold',size(data,1),kfold);

% Divide data into k-folds
loop=[1:kfold];

for i=loop
    datafold{i,:}=data(foldnum==i,:);
end

% Create training and testing datasets for each fold
for i=loop
    temp2=[];
    temp=loop(loop~=i);
    for j=1:length(temp)
        temp2=[temp2; datafold{temp(j),:}];
    end
    train{i,:}=temp2;

     % Use the remaining fold for testing
    test{i,:}=datafold{i,:};

    % Apply normalization if enabled
    if normalization~=0;
        TRAIN=train{i,:};
        TEST=test{i,:};

        % Extract min and max values for normalization
        DATA=[TRAIN;TEST];
        minb=min(DATA);
        maxb=max(DATA);
        xmin=minb(1,1:end-1);    xmax=maxb(1,1:end-1);

        for j=1:size(TRAIN,1)
            TRAIN(j,1:end-1) = normalizer(TRAIN(j,1:end-1),xmin, xmax, normalization); 
        end
        for j=1:size(TEST,1)
            TEST(j,1:end-1) = normalizer(TEST(j,1:end-1),xmin, xmax, normalization); 
        end
        train{i,:}=TRAIN;
        test{i,:}=TEST;
        % end of normalization   
    end
    
    % Save training and testing datasets to Excel files
    filename1=['training_productivity',num2str(i),'.xlsx'];
    filename2=['testing_productivity',num2str(i),'.xlsx'];
    
    xlswrite(filename1,train{i,:})
    xlswrite(filename2,test{i,:})
    
end
