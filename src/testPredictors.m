
loadAllData;

%% Try naive Bayes with hard-coded alpha value

nbParams.alpha = .001;
nbParams.numberOfClasses = 20; % added the number of classes to nbParams

%nbModel = naiveBayesTrainTesting(trainData, trainLabels, nbParams);

[likelihood_model, priors] = naiveBayesTrain(trainData, trainLabels, nbParams);

% compute training accuracy

nbTrainMaxVals = zeros(length(trainLabels), 1);
nbTrainPredictions = zeros(length(trainLabels), 1);

for i = 1:length(trainLabels)
    [nbTrainMaxVals(i), nbTrainPredictions(i)] = naiveBayesPredict(trainData(:,i), likelihood_model, priors); %trainData(:,i) contains ONLY the words that belong to document i
end

nbTrainAccuracy = nnz(nbTrainPredictions == trainLabels) ./ length(trainLabels)

% compute testing accuracy

nbMaxVals = zeros(length(testLabels), 1);
nbPredictions = zeros(length(testLabels), 1);

for i = 1:length(testLabels)
    [nbMaxVals(i), nbPredictions(i)] = naiveBayesPredict(testData(:,i), likelihood_model, priors);
end

nbAccuracy = nnz(nbPredictions == testLabels) ./ length(testLabels)


%% Filter by information gain

% d = 5000;
% 
% 
% 
% I don't think my implementation of IG is working perfectly
%gain = calculateInformationGain(trainData, trainLabels);
% [~, I] = sort(gain, 'descend');
% trainData = trainData(I(1:d), :);
% testData = testData(I(1:d), :);
 

% %% Try decision tree with hard-coded maximum depth
 
dtParams.maxDepth = 16;
dtModel = decisionTreeTrain(trainData, trainLabels, dtParams);

dtPredictions = zeros(length(testLabels), 1);

% compute training accuracy
dtTrainPredictions = decisionTreePredict(trainData, dtModel);

dtTrainAccuracy = nnz(dtTrainPredictions == trainLabels) ./ length(trainLabels)

% compute testing accuracy

for i = 1:length(testLabels)
    dtPredictions(i) = decisionTreePredict(testData(:,i), dtModel);
end

dtAccuracy = nnz(dtPredictions == testLabels) ./ length(testLabels)


