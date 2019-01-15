data = readtable('iris.csv');

% format table for testing (in this example whole table is used for testing)
test_labels = categorical(data{:,'species'});
test_examples = data;
test_examples(:,'species') = []