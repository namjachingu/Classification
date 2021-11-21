%knn-classifier
K = 7; 
%knnsearch(....) is a build-in function that finds the K nearest neighbors
%between training set and the test set with the euclidean distance. 

[cIdx,cD] = knnsearch(trainv,testv,'K',K,'Distance','euclidean'); %cIdx: 10000x7 matrix, holds the indices where the k nearest neighbors is located in trainv.
estLab_knn = trainlab (mode(transpose(cIdx))); 

%Confusion matrix
Conf_Mat_knn = confusionmat(testlab,transpose(estLab_knn));
disp(Conf_Mat_knn)

%Error rate = 0.0510
knn_antallfeil = num_test;
for i = 1:10
    for j = 1:10
        if i == j
            knn_antallfeil = knn_antallfeil - Conf_Mat_knn (i,j);
        end
    end
end

knn_errorrate = knn_antallfeil/num_test;
