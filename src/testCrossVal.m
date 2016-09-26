folds = 10;

% Cross validate naive bayes
params.alpha = .0001;
params.numberOfClasses = 20;
nbscore = crossValidate(@naiveBayesTrain, @naiveBayesPredict, trainData, trainLabels, folds, params);

% Cross validate decision tree
params.maxDepth = 16;
%dtscore = crossValidate(@decisionTreeTrain, @decisionTreePredict, trainData, trainLabels, folds, params);
