function [likelihood_model, priors] = naiveBayesTrainTesting(trainData, trainLabels, params)

alpha = params.alpha;
numWords = size(trainData,1);
%numDocuments = size(trainData,2);
numClasses = params.numberOfClasses;

likelihood_model = zeros(numClasses, numWords, 'double');
priors = zeros(numClasses, 1, 'double');

for class = 1:numClasses
   % trainData(find(trainLabels == class),:) is a matrix that contains the
   % words for every document in class 'class'
   % words(1,:) for example, will give the documents that have word with id
   % 1 in them.
   words = trainData(:,find(trainLabels == class));
   
   % sum(words,1) counts the number of words found for each document in
   % this class
   % calculate likelihoods
   likelihoods = (sum(words,2) + alpha) ./ (size(words,2) + 2*alpha);
   likelihood_model(class, :) = likelihoods;
   
   % calculate priors
   priors(class) = (size(words,2) + alpha) / (size(trainData,1) + 2*alpha);
end