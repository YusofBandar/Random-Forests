data = readtable('iris.csv');

% format table for testing (in this example whole table is used for testing)
test_labels = categorical(data{:,'species'});
test_examples = data;
test_examples(:,'species') = [];

%=======================================BAGGING================================
% Fraction of input data to sample with replacement from the input data for growing each new tree
InBagFraction = 0.1;

% Data to be used to grow the tree
InBagData = randi(size(test_examples,1),1,size(test_examples,1) * InBagFraction);
InBagData = test_examples(InBagData, :)

%=======================================FEATURE SELECTION=======================
% Number of Features to select at random for each decision split.
numFeatures = 2;

InBagDataColumnSize = size(InBagData,2);

if(numFeatures > InBagDataColumnSize) numFeatures = InBagDataColumnSize; end

% Features to be used
selectedFeatures = randperm(InBagDataColumnSize,numFeatures);
selectedFeatures = InBagData(:,selectedFeatures)

%=======================================SUMMING VOTES===========================
votes = readtable('votes.csv')


% Finding the highest vote
[val,ind] = max(sum(votes{:,:}));
% Extract which classification
votes.Properties.VariableNames(ind)