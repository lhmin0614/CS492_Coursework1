% Test Random Forest
leaf_assign = testTrees_fast(data_test,trees);

for T = 1:length(trees)
    p_rf(:,:,uint8(T)) = trees(1).prob(leaf_assign(:,uint8(T)),:);
end

% average the results from all trees
p_rf = squeeze(sum(p_rf,3))/length(trees); % Regression
[~,c] = max(p_rf'); % Regression to Classification
accuracy_rf = sum(c==data_test(:,end)')/length(c);% Classification accuracy (for Caltech dataset)
fprintf("accuracy is: ")
disp(accuracy_rf)

idx = sub2ind([10, 10], data_test(:,end)', c) ;
conf = zeros(10) ;
conf = vl_binsum(conf, ones(size(idx)), idx) ;

imagesc(conf) ;
title(sprintf('Confusion matrix (%.2f %% accuracy)', 100 * accuracy_rf) ) ;
%pause;