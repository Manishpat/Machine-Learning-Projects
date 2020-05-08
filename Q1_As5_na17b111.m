load('news_posts.mat');
Z=documents;
[m,n] = size(Z)
%converting a document matrix into full matrix
data=full(double(Z));

 %mean shifting
data=data';
means=mean(data); 
for i=1:100     
    data(i,:)=data(i,:)-means; 
end 
%applying svd to get pcs variences
 [U S V]=svds(data,100);
 %varience of first three pcs are
 pc1_var=S(1,1)*S(1,1);
 pc2_var=S(2,2)*S(2,2);
 pc3_var=S(3,3)*S(3,3);
 total_var=sum(diag(S).^2);
 var_perc=(pc1_var+pc2_var+pc3_var)/total_var;
 fprintf('Percentage Variance captured by  first 3 PCs %f\n',var_perc); 
 fprintf('Variance by 1st PC %f\n',pc1_var);
 
 
%-----------------------------------------------------
%PART B
for  i=1:100
    [F,adj_var,cum_var] = sparsePCA(data, i, 1);
    if adj_var>pc1_var*(.75)
        p=i;
        load_vec=F;
        break
    end
end

load_vec;
ind_nonzero=find(load_vec);
%word on that indeces
fprintf('No. of Non-zero elements in Sparse PCs for 0.75 of Variance in 1st PC is %f\n',p); 
fprintf('List of words in 1st Sparse PC are \n')
words_list=wordlist(ind_nonzero)




% %---------------------------------------------
%PART C
for  j=1:100
    [F1,adj_var1,cum_var1] = sparsePCA(data,j, 2);
    if sum(cum_var1(2))>=(pc1_var+pc2_var)*(.75)
        k=j;
        load_vec2=F1;
        break
    end
end

% load_vec2
 ind_nonzero1=find(load_vec2(:,2));
%word on that indexes
fprintf('No. of Non-zero elements in SPCs for 0.75 of cumalative variance is %f\n',k); 
fprintf('List of words in second Sparse PC are \n') 
words_list1=wordlist(ind_nonzero1)

