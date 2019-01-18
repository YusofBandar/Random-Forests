classdef myrandomforest
    methods(Static)
        
        function m = fit(num_trees,train_examples,train_labels,in_bag_fraction,num_features_to_sample)
            % Number of trees to grow
            m.num_tree = num_trees;
            % Fraction of input data to sample with replacement from the input data for growing each new tree
            m.in_bag_fraction = in_bag_fraction;
            % Number of Features to select at random for each decision split.
            m.num_features_to_sample = num_features_to_sample;
            
            % minimum number of training examples needed for a split
            m.min_parent_size = 10;
            
            % unique classes
            m.unique_classes = unique(train_labels);
            
            % feature names
            m.feature_names = train_examples.Properties.VariableNames;
            
            % training examples
            m.train_examples = train_examples;
            % training labels
            m.train_labels = train_labels;
            
            % number of training examples (rows)
            m.N = size(train_examples,1);
            
            m.trees = {};
            
            for i=1:num_trees
                [bag_examples,bag_labels] = myrandomforest.bagging(m);
                tree = mytree.fit(bag_examples,bag_labels,num_features_to_sample);
                m.trees{end + 1} = tree;
            end
            
        end
        
        function predictions = predict(m, test_examples)
            
            predictions = categorical;
            
            for i=1:size(test_examples,1)
                
                fprintf('classifying example %i/%i\n', i, size(test_examples,1));
                this_test_example = test_examples{i,:};
                
                tree = m.trees{1,1}
                
                this_prediction = mytree.predict_one(tree, this_test_example);
                predictions(end+1) = this_prediction;
                
            end
        end
        
         function prediction = predict_one(m, this_test_example)
            
            node = mytree.descend_tree(m.tree, this_test_example);
            prediction = node.prediction;
            
        end
        
        % Create a singe bag of testing examples.
        % The size of the bag is determined by the in bag fraction.
        function [bag_examples,bag_labels] = bagging(m)
            bag_size = 	int32(size(m.train_examples,1) * m.in_bag_fraction)
            
            bag_indices = randi(m.N,1,bag_size);
            
            bag_examples = m.train_examples(bag_indices, :);
            bag_labels = m.train_labels(bag_indices,:);
        end
        
    end
    
    
end