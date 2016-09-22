function [max_val label] = naiveBayesPredict(data, likelihood_model, priors)

% FILL IN YOUR CODE AND COMMENTS HERE

wordsInThisDocument = find(data == 1);
probabilitiesForEachWord = likelihood_model(:,wordsInThisDocument);
finalProbabilities = (prod(probabilitiesForEachWord,2)) .* priors ; % the numbers in prod(probabilitiesForEachWord,2) are ridiculously small

% This performs the "prediction" by choosing the highest probability
[max_val, class_index] = max(finalProbabilities);

label = class_index;
