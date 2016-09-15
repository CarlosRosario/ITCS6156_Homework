function gain = calculateInformationGain(data, labels)

% Computes the information gain on label probability for each feature in data
% data: d x n matrix of d features and n examples
% labels: n x 1 vector of class labels for n examples
% gain: d x 1 vector of information gain for each feature (H(y) - H(y|x_d))

gain = zeros(size(data, 1),1);

allPossibleClasses = unique(labels);

%Compute H(Y) term
HY = 0;
for class = 1:size(allPossibleClasses,1)
   totalNumberOfClassInstances = find(allPossibleClasses==class, 1, 'Last'); 
   probabilityOfClass = totalNumberOfClassInstances / size(labels,1);
   HY = HY + probabilityOfClass * log2(probabilityOfClass);
end
HY = HY * -1; % multiply by negative 1 due to formula


%Compute H(Y|X) term
for word = 1 : size(data,1)
    
    % calculate entropy
    hyx = 0;
    hyf = 0;
    
    pf = sum(data(word,:))/size(data,2); % probability of finding a word in any given document
    documentsWhereWordExists = find(data(word,:)); % vector containing the document ID's for each document that contains word 'word'
    yf = labels(documentsWhereWordExists); % Corresponding class labels for documents located in the above line of code. 
    
    % calculate h for classes given feature f
    yclasses = unique(yf);
     
    for class = 1:size(yclasses,1)
       totalNumberOfClassInstances = find(yf==class, 1, 'Last'); 
       if isempty(totalNumberOfClassInstances)
           
       else
            probabilityOfClass = totalNumberOfClassInstances / size(yf,1);
            hyf = hyf + probabilityOfClass * log2(probabilityOfClass);
       end
       
      
    end
    hyf = hyf * -1; % multiply by negative 1 due to formula
    hyx = pf * hyf;
    hyx = full(hyx);
    gain(word) = HY - hyx;
end

