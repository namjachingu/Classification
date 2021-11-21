x1 = load('class_1'); 
x2 = load('class_2');
x3 = load('class_3');

x1tr = x1(1:30, :); %training set of samples 1 to 30 (total of 30 samples) for class 1, dimension of 30x4
x1te = x1(31:50, :); %test set of samples 31 to 50 (total of 20 samples) for class 1, dimension of 20x4 
x2tr = x2(1:30, :);
x2te = x2(31:50, :);
x3tr = x3(1:30, :);
x3te = x3(31:50, :);

x_all = [x1tr; x2tr; x3tr]; %Dim 90x4 

x_class = [x_all, ones(90,1)]; %Input to the linear classifier with dimension 90x5, where the fifth column are 1s to make the input linear. 

W = abs(randn(3,5)/10); %Starts the classifier as a Cx(D+1) (3x5) matrix.

threshold = 100; 

while threshold > 1/1000
z = x_class * W.' ; 
g = 1./(1+exp(-z)); 
MSE = 0;

for i = 1:90 
   if i<31 
       t = [1,0,0]; 
   elseif i<61
       t=[0,1,0]; 
   else
       t=[0,0,1]; 
   end
   MSE = MSE + [(g(i,:)-t).*g(i,:).*(1-g(i,:))]'*(x_class(i,:)); 
end

W = W - 0.01*MSE; %W is updated by the gradiant of MSE, and keep it in continuously training in a while loop.

threshold = abs( sum(sum(MSE)))/15; %Stops the while loop when the error(MSE) is small. 
end


[mx, ind] = max(g'); %Decision rule. The maximum of g is found in order to find the numbers that are closest to 1.
class1 = ind(1:30); class2 = ind(31:60); class3 = ind(61:90); %Finding labels from the decision rule. 

%Making a confusion matrix, dimension 3x3, for the training set
confmat = [sum(class1(:)==1), sum(class1(:)==2), sum(class1(:)==3);...
     sum(class2(:)==1), sum(class2(:)==2), sum(class2(:)==3);... 
     sum(class3(:)==1), sum(class3(:)==2), sum(class3(:)==3)];
disp(confmat);


%Error rate of the training set. How many of the input samples were classified
%correctly or not.
amountOfWrong = sum(confmat(1, 2:3)) + confmat(2,1) + confmat(2,3)+ sum(confmat(3, 1:2)); %Finding where in the confusion matrix there are errors
ERR_training = 100*(amountOfWrong/90);
fprintf('This is the error rate for the training set: %.3f`\n', ERR_training);

%Runs the test set through the classifier. 
x_test = [x1te; x2te; x3te]; 
x_classtest = [x_test, ones(60,1)]; %60x5 dimension (60 samples in the test set, from three classes)
ztest = x_classtest*transpose(W);  
gtest = 1./(1+exp(-ztest));

gtestmax = max(gtest'); %Decision rule for the test set
[mx_test, indtest] = max(gtest');
class1test = indtest(1:20); class2test = indtest(21:40); class3test = indtest(41:60);

%Making a confusion matrix, dimension 3x3, for the test set:
confmattest = [sum(class1test(:)==1), sum(class1test(:)==2), sum(class1test(:)==3);...
    sum(class2test(:)==1), sum(class2test(:)==2), sum(class2test(:)==3);... 
    sum(class3test(:)==1), sum(class3test(:)==2), sum(class3test(:)==3)];

%Error rate of the test set. 
n_errors = sum(confmattest(1, 2:3)) + confmattest(2,1) + confmattest(2,3)+ sum(confmattest(3, 1:2));
ERR_test = 100*(n_errors/60);
fprintf('This is the error rate for the test set: %.3f\n', ERR_test);
