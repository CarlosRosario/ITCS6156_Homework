function model = decisionTreeTrain(trainData, trainLabels, params)

% FILL IN YOUR CODE AND COMMENTS HERE
trainData = full(trainData); % Un-sparse the matrix
trainData = double(trainData); % fitctree needs trainData to have double vlaues
trainData = trainData'; % transpose trainData to match trainLabels

depth = params.maxDepth;
model = fitctree(trainData, trainLabels);
model = prune(model, 'Level', depth); % prunes tree to a depth of 'depth'






