function model = naiveBayesTrain(trainData, trainLabels, params)

numDocuments = size(trainLabels,1);
numWords = size(trainData,1);
numClasses = params.numberOfClasses;

model = zeros(numDocuments, numClasses , 'double');
temp = zeros(numDocuments,2, 'double');
% for class = 1:numClasses
for class = 1:20
    
    % This vector contains the document id for every document that belongs 
    % to class 'class'
    documentsForThisClass = find(trainLabels == class); 
    
    % This matrix is a sparse matrix that contains the words that are in
    % each document for class 'class'
    %wordsForEachDocumentInThisClass = trainData(:,documentsForThisClass);
    
    % This is P(y=c)
    numberOfDocumentsForThisClass = size(documentsForThisClass,1);
    p_y_c = numberOfDocumentsForThisClass / size(trainLabels,1);
    
    % Get the first & last document id's for all documents in this class
    %minDocumentInThisClass = min(documentsForThisClass);
    %maxDocumentInThisClass = max(documentsForThisClass);
    
    % loop through each word and compute running Product - Pr(X)
    for document = 1:numDocuments
        runningProduct_Pr_X = 1;
        for word = 1:numWords
        
            % Tally up the occurrences of word 'word' for all documents in this
            % class (numerator of Pr(x_i))
            wordTally = nnz(trainData(word, document));
        
            % Compute Pr(x_i)
            probablityOfWord = (wordTally + 1) / 3;
            
            % Produce Pr(X) for each word/Pr(x_i)
            runningProduct_Pr_X = runningProduct_Pr_X * probablityOfWord;
        end
        
        % Finally, compute Pr(y_i = c | x_i) and store in
        % model(document,class)
        temp(document,1) = runningProduct_Pr_X;
        temp(document,2) = p_y_c;
        model(document,class) = p_y_c * runningProduct_Pr_X;
    end
end








