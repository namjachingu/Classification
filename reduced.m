x1 = load('class_1');
x2 = load('class_2');
x3 = load('class_3');

%Producing histograms for each feature and class:  
 for i = 1:4
 figure(i)
 histogram(x1(:, i), 'FaceColor', 'g');
 hold on;
 histogram(x2(:, i),'FaceColor', 'b'); 
 histogram(x3(:, i),'FaceColor', 'm'); 
 hold off;
 
 if i ==1 
    title('Sepal length of the three different variants of the Iris flower');
 elseif i==2
     title('Sepal width of the three different variants of the Iris flower');
 elseif i==3
     title('Petal length of the three different variants of the Iris flower');
 else
     title('Petal width of the three different variants of the Iris flower');
 end
 end
 
x1tr1 = [x1(1:30, 1), x1(1:30, 3), x1(1:30, 4)]; %New training set with one feature removed
x1te1 = [x1(31:50, 1), x1(31:50, 3), x1(31:50, 4)]; %New test set with one feature removed
x2tr1 = [x2(1:30, 1), x2(1:30, 3), x2(1:30, 4)];
x2te1 = [x2(31:50, 1), x2(31:50, 3), x2(31:50, 4)];
x3tr1 = [x3(1:30, 1), x3(1:30, 3), x3(1:30, 4)];
x3te1 = [x3(31:50, 1), x3(31:50, 3), x3(31:50, 4)];

x_all1 = [x1tr1; x2tr1; x3tr1]; %90x3 dimension, new training vector with one feature removed. There are now only three features in this vector.
x_class1 = [x_all1, ones(90,1)]; %dimension 90x4, where the fourth column is the ones at the input of the classifier. 

x1tr2 = [x1(1:30, 1), x1(1:30, 3)]; %New training set with only two features
x1te2 = [x1(31:50, 1), x1(31:50, 3)]; %New test set with only two features
x2tr2 = [x2(1:30, 1), x2(1:30, 3)];
x2te2 = [x2(31:50, 1), x2(31:50, 3)];
x3tr2 = [x3(1:30, 1), x3(1:30, 3)];
x3te2 = [x3(31:50, 1), x3(31:50, 3)];

x_all2 = [x1tr2; x2tr2; x3tr2]; %90x2 dimension, new training vector consisting of only two features. 
x_class2 = [x_all2, ones(90,1)]; %dimension 90x3, where the third column is ones at the input of the classifier. 

x1tr3 = [x1(1:30, 1)]; %New training set with only one feature
x1te3 = [x1(31:50, 1)]; %New test set with only one feature
x2tr3 = [x2(1:30, 1)];
x2te3 = [x2(31:50, 1)];
x3tr3 = [x3(1:30, 1)];
x3te3 = [x3(31:50, 1)];

x_all3 = [x1tr3; x2tr3; x3tr3]; %90x1 dimension, new training vector consisting of only one feature. 
x_class3 = [x_all3, ones(90,1)]; %dimension 90x2, where the second column is ones at the input of the classifier. 

W_new = abs(randn(3,2)/10); %Finding the parameters W, MSE and g again for the training and test sets with fewer features. 
threshold_new = 100;

while threshold_new > 1/1000
z_new = x_class3 * W_new.' ;    
g_new = 1./(1+exp(-z_new));
MSE_new = 0;

for i = 1:90
   if i<31 
       t = [1,0,0];
   elseif i<61
       t=[0,1,0];
   else
       t=[0,0,1];
   end
   MSE_new = MSE_new + [(g_new(i,:)-t).*g_new(i,:).*(1-g_new(i,:))]'*(x_class3(i,:)); 
end
W_new = W_new - 0.01*MSE_new;
threshold_new =abs( sum(sum(MSE_new)))/15; %stops when the error is small
end


[mx_new, ind_new] = max(g_new');
class1_new = ind_new(1:30); class2_new = ind_new(31:60); class3_new = ind_new(61:90);

confmat_new = [sum(class1_new(:)==1), sum(class1_new(:)==2), sum(class1_new(:)==3);... 
    sum(class2_new(:)==1), sum(class2_new(:)==2), sum(class2_new(:)==3);... 
    sum(class3_new(:)==1), sum(class3_new(:)==2), sum(class3_new(:)==3)]; %Finding the confusion matrix of the new training sets where some of the features have been removed. 

antallfeiltraining_new = sum(confmat_new(1, 2:3)) + confmat_new(2,1) + confmat_new(2,3)+ sum(confmat_new(3, 1:2));
ERR_training_new = 100*(antallfeiltraining_new/90); %Error rate for the training set where some of the features have been removed. 
fprintf('Error rate for the training set: %.3f`\n', ERR_training_new);

%Testing
xtest_new = [x1te3; x2te3; x3te3]; %New vector for test set with some removed features.
x_classtest_new = [xtest_new, ones(60,1)]; %Input vector at the linear classifier with some removes features.
ztest_new = x_classtest_new*transpose(W_new);  
gtest_new = 1./(1+exp(-ztest_new));
[mxtest_new, indtest_new] = max(gtest_new');
class1test_new = indtest_new(1:20); class2test_new = indtest_new(21:40); class3test_new = indtest_new(41:60);

confmattest_new = [sum(class1test_new(:)==1), sum(class1test_new(:)==2), sum(class1test_new(:)==3);...
    sum(class2test_new(:)==1), sum(class2test_new(:)==2), sum(class2test_new(:)==3);... 
    sum(class3test_new(:)==1), sum(class3test_new(:)==2), sum(class3test_new(:)==3)]; %Finding confusion matrix for the test set with removed features.

antallfeiltest_new = sum(confmattest_new(1, 2:3)) + confmattest_new(2,1) + confmattest_new(2,3)+ sum(confmattest_new(3, 1:2));
ERR_test_new = 100*(antallfeiltest_new/60); %Finding the error rate for the new test set with some removed features.
fprintf('Error rate for the test set: %.3f\n', ERR_test_new);
