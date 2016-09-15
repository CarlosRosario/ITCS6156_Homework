function [score, models] = crossValidate(trainer, predictor, trainData, trainLabels, folds, params)

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

% FILL IN YOUR CODE HERE

