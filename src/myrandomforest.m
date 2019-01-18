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
            
            % forest
            m.trees = {};
           
        
            
          
            
            % bag the data and train a decision tree on the bagged data
            for i=1:num_trees
                [bag_examples,bag_labels,bag_indices] = myrandomforest.bagging(m);
                
                tree = mytree.fit(bag_examples,bag_labels,num_features_to_sample);
                tree.id = i;
                
                %store which examples are out-of-bag by index
                tree.out_bag_example_indices = [1:m.N];
                tree.out_bag_example_indices(bag_indices) =[];
               
                
                
                m.trees{end + 1} = tree;
            end
            
            
     
            
           
           
            
        end
        
        function out_of_bag_error = oobError(m)
            
            
            correctly_classified = 0;
            
            % all the examples that were not bagged
            collected_oob_indices = myrandomforest.collect_oob_indices(m);
            
            
            for a=1:size(collected_oob_indices)
                i = collected_oob_indices(a);
                % get the training example
                this_train_example = m.train_examples{i,:};
                
                tree_predictions = categorical;
                
                %iterate through trees
                for k=1:size(m.trees,2)
                    tree = m.trees{1,k};
                    
                    % did this tree use this example, if not try to predict
                    if ismember(i,tree.out_bag_example_indices)
                        this_prediction = mytree.predict_one(tree, this_train_example);
                        tree_predictions(end+1) = this_prediction;
                    end
                end
                % final prediction
                prediction = myrandomforest.modal_prediction(tree_predictions);
                
                % was the predicition classified correctly
                if prediction == m.train_labels(i,1)
                   correctly_classified = correctly_classified + 1; 
                end
            end
            
            % calculate the number of example incorrectly classified
            out_of_bag_error = (collected_oob_indices - correctly_classified)/collected_oob_indices;
            
            
        end
        
        function oob_indices = collect_oob_indices(m)
            oob_indices = []; 
            for k=1:size(m.trees,2)
                    tree = m.trees{1,k};
                    oob_indices = [oob_indices,tree.out_bag_example_indices];
            end
             
            oob_indices = unique(oob_indices);
        end
        
       
        
        function predictions = predict(m, test_examples)
            
            predictions = categorical;
            
            for i=1:size(test_examples,1)   
                this_test_example = test_examples{i,:};
                
                %holds the predicition of each tree in the forest for this
                %test example
                tree_predictions = categorical;
                for k=1:size(m.trees,2)
                    tree = m.trees{1,k};
                    this_prediction = mytree.predict_one(tree, this_test_example);
                    tree_predictions(end+1) = this_prediction;
                    
                end
               % caluclate the prediction by finding the modal predicition 
               predictions(end+1) = myrandomforest.modal_prediction(tree_predictions);
                
               
                
            end
        end
        
        % find the modal prediction 
        function prediction = modal_prediction(predictions)
            if(isempty(predictions))
                prediction = "";
                return 
            end
            
            [unique_predictions, ~,indicies] = unique(predictions);
            v = mode(indicies);
            prediction = unique_predictions(v);
        end
        
         function prediction = predict_one(m, this_test_example)
            
            node = mytree.descend_tree(m.tree, this_test_example);
            prediction = node.prediction;
            
        end
        
        % Create a singe bag of testing examples.
        % The size of the bag is determined by the in bag fraction.
        function [bag_examples,bag_labels,bag_indices] = bagging(m)
            bag_size = 	int32(size(m.train_examples,1) * m.in_bag_fraction);
            
            bag_indices = randi(m.N,1,bag_size);
         
            bag_examples = m.train_examples(bag_indices, :);
            bag_labels = m.train_labels(bag_indices,:);
        end
        
    end
    
    
end
