 %Plotting some of the misclassified pictures

x = zeros (28,28); 
n =10;

    for i = 1 : num_test 
        if testlab (i) ~= estLab (i) 
            x(:)  = testv(i,:);  
            n = n-1;
            text = sprintf ('Number %i misclassifed as %i', testlab(i), estLab(i));
        end  
        if n < 1 
            break
        end
        figure (n) 
        
        image(transpose(x)); title (text); hold on %Print.
    end 
    
n = 10;
m = 20;
     for i = 1 : num_test
        if testlab (i) == estLab (i)
            x(:)  = testv(i,:);
            m = m-1;
            text = sprintf ('Number %i classifed as %i', testlab(i), estLab(i));
        end  
        if m < 1
            break
        end
        figure (n+m) 
        
        image(transpose(x)); title (text); hold on 
    end 
