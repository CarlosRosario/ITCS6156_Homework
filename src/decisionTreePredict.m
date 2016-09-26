function labels = decisionTreePredict(data, model)

% FILL IN YOUR CODE AND COMMENTS HERE
data = full(data); % Un-sparse the matrix
data = double(data); % fitctree needs trainData to have double vlaues
data = data'; % transpose trainData to match trainLabels

labels = predict(model, data);

