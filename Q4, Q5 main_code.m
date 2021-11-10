%% 5. Experiment with Caltech dataset for image categorisation (Coursework 1)

param.num = 10;
param.depth = 10;    % trees depth
param.splitNum = 3; % Number of trials in split function
param.split = 'IG'; % Currently support 'information gain' only

% Complete getData.m by writing your own lines of code to obtain the visual 
% vocabulary and the bag-of-words histograms for both training and testing data. 
% You can use any existing code for K-means (note different codes require different memory and computation time).

%[data_train, data_test] = getData('Caltech')a;


%filename = 'datatraintest.xlsx';
%writematrix(data_train, filename,'Sheet', 1);
%writematrix(data_test, filename, 'Sheet', 2);



init ;
%data_train = readmatrix("datatraintest.xlsx", 'Sheet','Sheet1');
%data_test = readmatridatatraintest.xlsx", 'Sheet','Sheet2');

%[data_train, data_test] = getData('Caltech', 512);
for N = [10] % Number of trees, try {1,3,5,10, or 20}
    param.num = 19;
    param.depth = 5;    % trees depth
    param.splitNum = 10; % Number of trials in split function
    param.split = 'IG'; % Currently support 'information gain' only
  

    % Select dataset
    [data_train, data_test] = getData('Caltech', N); % {'Toy_Gaussian', 'Toy_Spiral', 'Toy_Circle', 'Caltech'}
    
    % Train Random Forest

    trees = growTrees(data_train, param);
    
    % Test Random Forest
    testTrees_script;
    
   
    % Visualise
    %visualise(data_train,p_rf,[],0);
    %disp('Press any key to continue');
    %pause;
end