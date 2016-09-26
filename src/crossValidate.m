function score = crossValidate(trainer, predictor, trainData, trainLabels, folds, params)

% Performs cross validation with random splits
% trainer: function that trains a model from data with the template 
%          model = functionName(trainData, trainLabels, params)
% predictor: function that predicts a label from a single data point
%            label = functionName(data, model)
% trainData: d x n data matrix
% trainLabels: n x 1 label vector
% folds: number of folds to run of validation
% params: auxiliary variables for training algorithm (e.g., regularization
%         parameters)
% score: the accuracy averaged over all folds

trainingAlgName = func2str(trainer);
predictAlgName = func2str(predictor);
numDocuments = size(trainData,2);
numElements = ceil(numDocuments / folds);
runningAccuracy = 0;

       for i = 1:100000
           randomVector = randperm(11269,2);
           rand1 = randomVector(1);
           rand2 = randomVector(2);
           
           % swap data
           tempVector = trainData(:,rand1);
           trainData(:,rand1) = trainData(:,rand2);
           trainData(:,rand2) = tempVector;
           
           % swap labels
           tempValue = trainLabels(rand1,:);
           trainLabels(rand1,:) = trainLabels(rand2,:);
           trainLabels(rand2,:) = tempValue;
           
       end

% if(strcmp(trainingAlgName, 'decisionTreeTrain'))
%     % Need to randomize the trainData & trainLabels
%     % loop for 100,000 iterations
%        % pick a random number b/t 1 and 11269
%        % pick another random b/t 1 and 11269 but NOT the one already
%        % chosen
%        % swap the two columns, and swap the corresponding labels
%        
% 
% end

for fold = 1:folds
    
    % Create the training subset & testing subsets for each fold
    if fold == 1
       startIndex = 1;
       endIndex = startIndex + numElements -1;
       
       trainSet = (trainData(:, endIndex:numDocuments));
       trainLabelSet = trainLabels(endIndex:numDocuments,1);
    else
        startIndex = (numElements * (fold - 1)) + (fold -1);
        endIndex = startIndex + numElements;
        
        if fold == folds
            trainSet = trainData(:, 1:startIndex-1);
            trainLabelSet = trainLabels(1:startIndex-1,1);
        else
            leftTrainSet = trainData(:, 1:startIndex-1);
            rightTrainSet = trainData(:, endIndex+1:numDocuments);
            trainSet = horzcat(leftTrainSet, rightTrainSet);
            
            leftTrainLabelSet = trainLabels(1:startIndex-1,1);
            rightTrainLabelSet = trainLabels(endIndex+1:numDocuments,1);
            trainLabelSet = vertcat(leftTrainLabelSet, rightTrainLabelSet);
        end
    end
    
    if fold == folds
        testSet = trainData(:, startIndex:endIndex-folds);
        testLabelSet = trainLabels(startIndex:endIndex-folds, 1);
    else
        testSet = trainData(:, startIndex:endIndex);
        testLabelSet = trainLabels(startIndex:endIndex, 1);
    end
    
    % because naivebayes and decision tree have some slightly different
    % return values and parameters i used this if statement to make sure
    % the correct training/prediction algorithms are called. Not the best
    % way to do this but, it's effective for this assignment
    if strcmp(trainingAlgName, 'naiveBayesTrain')
        
        % train
        [likelihood_model, priors] = trainer(trainSet, trainLabelSet, params);
        
        % predict and get accuracy
        nbPredictions = zeros(length(testLabelSet), 1);

        for i = 1:length(testLabelSet)
            [~, nbPredictions(i)] = predictor(testSet(:,i), likelihood_model, priors);
        end

        nbAccuracy = nnz(nbPredictions == testLabelSet) ./ length(testLabelSet);
        runningAccuracy = runningAccuracy + nbAccuracy;
        
    else
        
        %train
        dtModel = trainer(trainSet, trainLabelSet, params);
        
        % predict and get accuracy
        %dtPredictions = zeros(length(testLabelSet), 1);
        dtPredictions = predictor(testSet, dtModel);
        %for i = 1:length(testLabelSet)
        %    dtPredictions(i) = decisionTreePredict(testSet(:,i), dtModel);
        %end
        
        dtaccuracy = nnz(dtPredictions == testLabelSet) ./ length(testLabelSet)
        runningAccuracy = runningAccuracy + dtaccuracy;
    end
end

score = runningAccuracy/folds;
