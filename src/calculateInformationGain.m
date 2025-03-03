function gain = calculateInformationGain(data, labels)

% Computes the information gain on label probability for each feature in data
% data: d x n matrix of d features and n examples
% labels: n x 1 vector of class labels for n examples
% gain: d x 1 vector of information gain for each feature (H(y) - H(y|x_d))

numDocuments = size(data,2);
gain = zeros(size(data, 1),1);
allPossibleClasses = unique(labels);

%Compute H(Y) term / Entropy(Y)
HY = 0;
for class = 1:size(allPossibleClasses,1)
   totalNumberOfClassInstances = find(allPossibleClasses==class, 1, 'Last'); 
   probabilityOfClass = totalNumberOfClassInstances / size(labels,1);
   HY = HY + (probabilityOfClass * log2(probabilityOfClass));
end
HY = HY * -1; % multiply by negative 1 due to negative sign in the formula for H(Y)


%Compute H(Y|X) term
for word = 1 : size(data,1)
    
    %HY_X = 0;
    entropyForClass = 0;
    
    % probability of finding word 'word' in any given document.This is 
    % basically the (S_v)/(S) in the formula described in mitchell 8.3 for. 
    % information gain. 
    % sum(data(word,:)) gives the total # of ocurrences of word 'word' for 
    % all documents
    overallWordProbability = sum(data(word,:))/numDocuments; 
    
    % vector containing the document ID's for each document that contains word 'word'
    documentsWhereWordExists = find(data(word,:)); 
    
    % Corresponding class labels (id's) for documents located in the above line of code. 
    % Don't need to consider classes where word does not exist because it
    % will simply add zero to the running entropy value
    classesThatContainWord = labels(documentsWhereWordExists); 
    
    % Set of classes that contain documents that contain word 'word'
    classes = unique(classesThatContainWord);
     
    for class = 1:size(classes,1)
       % get a count of the total number of documents with class 'class'
       % that contain word 'word'. This will be the numerator for the value
       % that is plugged into the entropy formula.
       totalNumberOfClassInstances = find(classesThatContainWord==class, 1, 'Last'); 
       
       % prevent sparse-matrix indexing issues where a 0 would exist in a 
       % dense matrix. This is an implementation detail
       if ~isempty(totalNumberOfClassInstances) 
           probabilityOfClass = totalNumberOfClassInstances / size(classesThatContainWord,1);
           entropyForClass = entropyForClass + probabilityOfClass * log2(probabilityOfClass);  
       end
    end
    
    entropyForClass = entropyForClass * -1; % multiply by negative 1 due to formula
    HY_X = overallWordProbability * entropyForClass;
    
    % implementation detail. Cannot substract dense and sparse matrices in matlab.
    HY_X = full(HY_X);  
    
    % final computation to get information gain for each word
    gain(word) = HY - HY_X;
end

