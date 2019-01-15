data = readtable('iris.csv');

% format table for testing (in this example whole table is used for testing)
test_labels = categorical(data{:,'species'});
test_examples = data;
test_examples(:,'species') = [];

% Fraction of input data to sample with replacement from the input data for growing each new tree
InBagFraction = 0.1;

% Data to be used to grow the tree
InBagData = randi(size(test_examples,1),1,size(test_examples,1) * InBagFraction);
InBagData = test_examples(InBagData, :)