load('data_all.mat');

z_all=ones(60, 1000); 
index_all = zeros(60,1000);
estInd=zeros(1000,1);
estLab=zeros(10000,1);


for j = 1:10
    v=testv((1+(j-1)*1000):(j*1000), :); %1000*j test vectors.

    for i = 1:60 
        w=trainv((1+(i-1)*1000):(i*1000), :); %1000*i training vectors.
        z = pdist2(w, v, 'euclidean'); % v is 1000x784 and w is 60x784.
      [distanse_mindre,index] = min(z); 
      z_all (i, :) = transpose(distanse_mindre); %Works as a memory, saving the 60 mimimum distances from each group i. 
      index_all (i,:) = transpose(index+(i-1)*1000);%Same as z_all, but saves the indices(1-1000) where the minimum distances is located in z_all.
    end
     
[minAll, index_ny] = min(z_all); 
for k = 1:1000
estInd(k) = index_all(index_ny(k), k); %Saves the index corresponding to each test vector where the shortest distance for all the training vectors is located.  
end
estLab(1+(j-1)*1000:(j*1000))= trainlab(estInd); %Fetches all numbers in trainlab on the estInd rows, and saves it is estLab.  

end
estLab = transpose(estLab);

%Confusion matrix
Conf_Mat = confusionmat(testlab,estLab); 
disp(Conf_Mat)

%Error rate
nn_antallfeil = num_test; 
for i = 1:10 
    for j = 1:10
        if i == j
            nn_antallfeil = nn_antallfeil - Conf_Mat (i,j);
        end
    end
end

nn_errorrate = nn_antallfeil/num_test; 
