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
        end
        
        
        
    end
    
    
end