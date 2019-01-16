classdef mytree
    methods(Static)
        
        % Generates descision tree from the training examples
        % Pre:
        %   train_examples : table of training examples
        %   train_labels : table of labels mapped to the train_examples
        % Post: returns decision tree classifier
        function m = fit(train_examples, train_labels)
            
            % initialise empty node
            emptyNode.number = [];
            % traning examples the node holds
            emptyNode.examples = [];
            % labels associated with training examples
            emptyNode.labels = [];
            % prediction node makes
            emptyNode.prediction = [];
            % the impurity of the node (used when splitting)
            emptyNode.impurityMeasure = [];
            % children of the node
            emptyNode.children = {};
            % feature to split on
            emptyNode.splitFeature = [];
            % feature name
            emptyNode.splitFeatureName = [];
            % feature column index
            emptyNode.splitValue = [];
            
            m.emptyNode = emptyNode;
            
            % create root node
            r = emptyNode;
            % root unique number set to 1
            r.number = 1;
            % root holds all training labels
            r.labels = train_labels;
            % root holds all training examples
            r.examples = train_examples;
            r.prediction = mode(r.labels);
            
            % minimum number of training examples needed for a split
            m.min_parent_size = 10;
            % unique classes
            m.unique_classes = unique(r.labels);
            % feature names
            m.feature_names = train_examples.Properties.VariableNames;
            % number of nodes 1 as we only have a root node
         			m.nodes = 1;
            % number of training examples (rows)
            m.N = size(train_examples,1);
            % attempt to split root node
            m.tree = mytree.trySplit(m, r);
            
        end
        
        
        % Attempts to split the current node into two children
        % Pre:
        %   m : classifier
        %   node : node structure
        % Post: returns generated descision tree
        function node = trySplit(m, node)
            
            % BASE CASE
            % checks if current node can be split (can it become a
            % parent)
            if size(node.examples, 1) < m.min_parent_size
            				return
            end
            
            % calculate node impurity
            node.impurityMeasure = mytree.weightedImpurity(m, node.labels);
            
            % iterate through each feature (column)
            for i=1:size(node.examples,2)
                
            				fprintf('evaluating possible splits on feature %d/%d\n', i, size(node.examples,2));
                % re-order rows by the ith feature (ascending)
            				[ps,n] = sortrows(node.examples,i);
                % re-order the labels
                ls = node.labels(n);
                % initialise biggest reduction to infintly small
                biggest_reduction(i) = -Inf;
                biggest_reduction_index(i) = -1;
                biggest_reduction_value(i) = NaN;
                
                % iterate through each row of examples, every possiable
                % split
                for j=1:(size(ps,1)-1)
                    
                    % is the value of the ith feature for the jth row the
                    % same as the next if so don't split
                    if ps{j,i} == ps{j+1,i}
                        continue;
                    end
                    
                    % calculate the reduction of impurity for this split
                    % calculates wieghted impurity for row(1-jth) -
                    % row(jth+1-end)
                    this_reduction = node.impurityMeasure - (mytree.weightedImpurity(m, ls(1:j)) + mytree.weightedImpurity(m, ls((j+1):end)));
                    
                    % is this reduction the largest if so store the biggest reduction and row number
                    if this_reduction > biggest_reduction(i)
                        biggest_reduction(i) = this_reduction;
                        biggest_reduction_index(i) = j;
                    end
                end
                
            end
            
            % finds the largest reduction across each feature
            [winning_reduction,winning_feature] = max(biggest_reduction);
            % winning_index the row to split
            winning_index = biggest_reduction_index(winning_feature);
            
            % BASE CASE
            % was it possiable to acheive a reduction in impurity
            if winning_reduction <= 0
                return
            else
                % generate two children nodes
                
                % re-order rows by the winning feature (ascending)
                [ps,n] = sortrows(node.examples,winning_feature);
                % re-order the labels
                ls = node.labels(n);
                
                % set the winning feature column number
                node.splitFeature = winning_feature;
                % set the winning feature name
                node.splitFeatureName = m.feature_names{winning_feature};
                % CONFUSED
                % value to split on when testing
                node.splitValue = (ps{winning_index,winning_feature} + ps{winning_index+1,winning_feature}) / 2;
                
                % clear examples for this node
                node.examples = [];
                % clear labels
                node.labels = [];
                % clear prediction this node is not a leaf
                node.prediction = [];
                
                % FIRST CHILD
                % generate empty child
                node.children{1} = m.emptyNode;
                % increment the number of nodes
                m.nodes = m.nodes + 1;
                % child unique number
                node.children{1}.number = m.nodes;
                % child examples less than or equal to split value
                node.children{1}.examples = ps(1:winning_index,:);
                % labels associated with examples
                node.children{1}.labels = ls(1:winning_index);
                % prediction equal modal label
                node.children{1}.prediction = mode(node.children{1}.labels);
                
                % SECOND CHILD
                node.children{2} = m.emptyNode;
                m.nodes = m.nodes + 1;
                node.children{2}.number = m.nodes;
                % child examples greater than the split value
                node.children{2}.examples = ps((winning_index+1):end,:);
                node.children{2}.labels = ls((winning_index+1):end);
                node.children{2}.prediction = mode(node.children{2}.labels);
                
                % RECURSIVE CALL
                % attempt to split each child
                node.children{1} = mytree.trySplit(m, node.children{1});
                node.children{2} = mytree.trySplit(m, node.children{2});
            end
            
        end
        
        % Calculates the impurity of labels given by Ginis Diversity Index
        % (GDI) (0 for pure)
        %
        % Pre:
        %   m : classifier
        %   labels : categorical table of labels
        % Post: returns weighted impurity
        function e = weightedImpurity(m, labels)
            % CONFUSED
            weight = length(labels) / m.N;
            
            summ = 0;
            
            % numbers of labels observed for this node
            obsInThisNode = length(labels);
            
            % calculate GDI for each class abd add it to running total summ
            for i=1:length(m.unique_classes)
                % calculate the fraction of the class labels in the set
                % observed labels
            				pi = length(labels(labels==m.unique_classes(i))) / obsInThisNode;
                summ = summ + (pi*pi);
                
            end
            
            % minus 1 from summ to calculate GDI
            g = 1 - summ;
            
            e = weight * g;
            
        end
        
        % Predicts the labels for each test example
        % Pre:
        %   m : classifier
        %   test_examples : table of test examples
        % Post: returns table of labels
        function predictions = predict(m, test_examples)
            
            % initialise empty categorical table to store the predictions
            % for each test
            predictions = categorical;
            
            % iterate through test_examples (each row)
            for i=1:size(test_examples,1)
                
            				fprintf('classifying example %i/%i\n', i, size(test_examples,1));
                %get the ith row
                this_test_example = test_examples{i,:};
                % predict the label for the this text example (ith row)
                this_prediction = mytree.predict_one(m, this_test_example);
                % push the prediction to the predicitions array
                predictions(end+1) = this_prediction;
                
         			end
        end
        
        % Predicts the label for this specific text example by descending
        % the tree until a leaf node (node has no children) is reached
        %
        % Pre:
        %   m : classifier
        %   this_test_example : array of a specific test example
        % Post: returns prediction (label) for this specific test example
        function prediction = predict_one(m, this_test_example)
            
            %find the leaf node
         			node = mytree.descend_tree(m.tree, this_test_example);
            %extract predicition from leaf node
            prediction = node.prediction;
            
        end
        
        % Descends the tree until it reaches a leaf node (has no children)
        % Pre:
        %   node : current node
        %   this_test_example : array of a specific test example
        % Post: returns a leaf node
        function node = descend_tree(node, this_test_example)
            % BASE CASE
            % is node a leaf (has no children)
         			if isempty(node.children)
                return;
            else
                % is test example less than split value descebd the left child if
                % not descend the right child
                if this_test_example(node.splitFeature) < node.splitValue
                    node = mytree.descend_tree(node.children{1}, this_test_example);
                else
                    node = mytree.descend_tree(node.children{2}, this_test_example);
                end
            end
            
      		end
        
        % Prints representation of the tree
        % Pre :
        %   node : current node
        function describeNode(node)
            
            % If node is a child (no children) print node details
         			if isempty(node.children)
                fprintf('Node %d; %s\n', node.number, node.prediction);
            else
                % print current node details
                fprintf('Node %d; if %s <= %f then node %d else node %d\n', node.number, node.splitFeatureName, node.splitValue, node.children{1}.number, node.children{2}.number);
                
                %descend the tree
                mytree.describeNode(node.children{1});
                mytree.describeNode(node.children{2});
            end
            
      		end
        
    end
end
