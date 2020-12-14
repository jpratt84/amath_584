% J. Pratt AMATH 584 HW 6
% The function mnist_parse.m was taken from Stack Exchange, and was
% authored by user rayryeng in September of 2016.

clear; close all; clc

[trainim, trainlabels] = mnist_parse('train-images-idx3-ubyte', ...
    'train-labels-idx1-ubyte');

[m,n,j] = size(trainim); %store size of trainim tensor

Xtr = []; %initialize training data matrix
for jj=1:j
    Xtr(:,jj) = reshape(trainim(:,:,jj), [m*n, 1]); %convert trainim tensor
    %into a matrix, where each column is one image
end

Btr = zeros(10,j);
% Build matrix of labels for the training data. A '1' in the kth row
% corresponds to that digit shown in the images; i.e., a '1' in the first
% row corresponds to the digit '1', a '1' in the fifth row corresponds to
% the digit '5', and a '1' in the 10th row corresponds to the digit '0',
% etc.
for jj=1:j
    for k=1:10
        if k==trainlabels(jj)
            Btr(k,jj) = 1;
        elseif trainlabels(jj)==0
            Btr(10,jj) = 1;
        end
    end
end

% Repeat the above process to build test data and label matrices
[testim, testlabels] = mnist_parse('t10k-images-idx3-ubyte', ...
    't10k-labels-idx1-ubyte');
[mm,nn,tt] = size(testim);

Xtest = [];

for jj=1:tt
    Xtest(:,jj) = reshape(testim(:,:,jj),[mm*nn, 1]);
end

Btest = zeros(10,tt);
for jj=1:tt
    for kk=1:10
        if kk==testlabels(jj)
            Btest(kk,jj) = 1;
        elseif testlabels(jj)==0
            Btest(10,jj) = 1;
        end
    end
end

%% Least-Square Fitting via pinv()
a1 = Btr*pinv(Xtr); %build matrix of weightings
tresult1 = a1*Xtest; %apply matrix to test data
digitpred1 = zeros(tt,1);
correct1 = 0;
for s=1:tt
    [val,ind] = max(tresult1(:,s)); %find index of max weighting in each 
    %column of the test results; the index of this max value corresponds to
    %the predicted digit value
    if ind==10 %if the index is 10, change it to zero, since we are only 
        %considering digits 0 through 9
        ind = 0;
    end
    digitpred1(s) = ind;
    if digitpred1(s) == testlabels(s)
        correct1 = correct1+1;
    end
end
% Correctly labeled digits = 8534

% Plot predicted and true digit values
mxd = 25; %max number of digits to plot out of 10000
figure(1), subplot(2,1,1), bar(digitpred1(1:mxd)),
xlabel('kth Digit'), ylabel('Digit Value'),
title('Predicted Digit Values, pinv');
subplot(2,1,2), bar(testlabels(1:mxd));
xlabel('kth Digit'), ylabel('Digit Value'),
title('True Digit Values, pinv');

%% Lasso Method
lambda = 0.1;
a2 = lasso(Xtr',trainlabels,'Lambda',lambda,'Intercept',false);
%%
tresult2 = a2'*Xtest;
tresult2 = round(tresult2); %round results to the nearest integer
figure(2), subplot(2,1,1), bar(tresult2(1:25)),
xlabel('kth Digit'), ylabel('Digit Value'),
title('Predicted Digit Values, lasso (\lambda = 0.1)');
subplot(2,1,2), bar(testlabels(1:25)),
xlabel('kth Digit'), ylabel('Digit Value'),
title('True Digit Values, lasso (\lambda = 0.1)');

%Count correct guesses out of 10,000
correct2 = 0;
for i=1:10000
    if tresult2(i) == testlabels(i)
        correct2 = correct2+1;
    end
end
% for lambda = 0.001, correct = 2455
% for lambda = 0.01, correct = 2442
% for lambda = 0.1, correct = 2494

%% Backslash operator
a3 = Xtr'\trainlabels;
%%
tresult3 = a3'*Xtest;
tresult3 = round(tresult3);

figure(3), subplot(2,1,1), bar(tresult3(1:25)),
xlabel('kth Digit'), ylabel('Digit Value'),
title('Predicted Digit Values, Backslash Operator');
subplot(2,1,2), bar(testlabels(1:25)),
xlabel('kth Digit'), ylabel('Digit Value'),
title('True Digit Values, Backslash Operator');

%Count correct guesses out of 10,000
correct3 = 0;
for i=1:10000
    if tresult3(i) == testlabels(i)
        correct3 = correct3+1;
    end
end
%correct = 2433

%% Using pinv, determine and rank pixels
% Determine the largest values in each row of a1 to determine the most
% important pixel values for each digit image. Set all other pixel values
% to zero to promote sparsity
[j,k] = size(a1);
a1sp = a1;
for i=1:j
    for l=1:k
        if (a1sp(i,l))<1e-7
            a1sp(i,l) = 0;
        end
    end
end

testsparse = a1sp*Xtest;

% Determine accuracy of sparse matrix applied to test data
dsparse = zeros(10000,1);
csparse = 0;
for s=1:10000
    [val,ind] = max(testsparse(:,s)); %find index of max weighting in each 
    %column of the test results; the index of this max value corresponds to
    %the predicted digit value
    if ind==10 %if the index is 10, change it to zero, since we are only 
        %considering digits 0 through 9
        ind = 0;
    end
    dsparse(s) = ind;
    if dsparse(s) == testlabels(s)
        csparse = csparse+1;
    end
end
% Correctly labeled digits = 7345. Cutting out values in a1 less than 1e-4
% resulted in much reduced accuracy. Best accuracy using 1e-7

% Plot predicted and true digit values
mxd = 25; %max number of digits to plot out of 10000
figure(4), subplot(2,1,1), bar(dsparse(1:mxd)),
xlabel('kth Digit'), ylabel('Digit Value'),
title('Predicted Digit Values, sparse');
subplot(2,1,2), bar(testlabels(1:mxd));
xlabel('kth Digit'), ylabel('Digit Value'),
title('True Digit Values, sparse');

%% Compare first 50 values of original and spare coefficient matrices
figure(5), subplot(2,1,1), plot(a1(1,1:50),'o','Linewidth',[2]), 
ylim([-0.015 0.015]), xlabel('Column'), ylabel('Value'),
title('First 50 Values of Coefficient Matrix, Row 1');
subplot(2,1,2), plot(a1sp(1,1:50),'o','Linewidth',[2]), 
ylim([-0.015 0.015]), xlabel('Column'), ylabel('Value'),
title('First 50 Values of Sparse Coeff. Matrix, Row 1');
%% Calculate size of sparse coeff. matrix
num0 = 0;
num0sp = 0;
for i=1:10
    for j=1:784
        if a1(i,j) == 0
            num0 = num0+1;
        end
    end
end

for i=1:10
    for j=1:784
        if a1sp(i,j)==0
            num0sp = num0sp+1;
        end
    end
end
%num0 in original a1 matrix = 150, a1 is 10 by 784 = 7840 total values
%num0 in sparse a1 matrix = 4301, 4301/7840 = 0.5486. a1sparse uses 45% 
%less data than the full a1, accuracy loss about 12%
%accuracy reduced from 85.34% to 73.45%

%% Determine Most Important Pixels for Each Digit
a1r1 = reshape(a1(1,:),m,n);
a1r2 = reshape(a1(2,:),m,n);
a1r3 = reshape(a1(3,:),m,n);
a1r4 = reshape(a1(4,:),m,n);
a1r5 = reshape(a1(5,:),m,n);
a1r6 = reshape(a1(6,:),m,n);
a1r7 = reshape(a1(7,:),m,n);
a1r8 = reshape(a1(8,:),m,n);
a1r9 = reshape(a1(9,:),m,n);
a1r10 = reshape(a1(10,:),m,n);

a1spr1 = reshape(a1sp(1,:),m,n);
a1spr2 = reshape(a1sp(2,:),m,n);
a1spr3 = reshape(a1sp(3,:),m,n);
a1spr4 = reshape(a1sp(4,:),m,n);
a1spr5 = reshape(a1sp(5,:),m,n);
a1spr6 = reshape(a1sp(6,:),m,n);
a1spr7 = reshape(a1sp(7,:),m,n);
a1spr8 = reshape(a1sp(8,:),m,n);
a1spr9 = reshape(a1sp(9,:),m,n);
a1spr10 = reshape(a1sp(10,:),m,n);

figure(6), subplot(3,2,1), pcolor(flipud(a1r1)), colormap gray, colorbar,
title('Digit 1 Full');
subplot(3,2,2), pcolor(flipud(a1spr1)), colormap gray, colorbar,
title('Digit 1 Sparse');
subplot(3,2,3), pcolor(flipud(a1r2)), colormap gray, colorbar,
title('Digit 2 Full');
subplot(3,2,4), pcolor(flipud(a1spr2)), colormap gray, colorbar,
title('Digit 2 Sparse');
subplot(3,2,5), pcolor(flipud(a1r3)), colormap gray, colorbar,
title('Digit 3 Full');
subplot(3,2,6), pcolor(flipud(a1spr3)), colormap gray, colorbar,
title('Digit 3 Sparse');

figure(7), subplot(3,2,1), pcolor(flipud(a1r4)), colormap gray, colorbar,
title('Digit 4 Full');
subplot(3,2,2), pcolor(flipud(a1spr4)), colormap gray, colorbar,
title('Digit 4 Sparse');
subplot(3,2,3), pcolor(flipud(a1r5)), colormap gray, colorbar,
title('Digit 5 Full');
subplot(3,2,4), pcolor(flipud(a1spr5)), colormap gray, colorbar,
title('Digit 5 Sparse');
subplot(3,2,5), pcolor(flipud(a1r6)), colormap gray, colorbar,
title('Digit 6 Full');
subplot(3,2,6), pcolor(flipud(a1spr6)), colormap gray, colorbar,
title('Digit 6 Sparse');

figure(8), subplot(4,2,1), pcolor(flipud(a1r7)), colormap gray, colorbar,
title('Digit 7 Full');
subplot(4,2,2), pcolor(flipud(a1spr7)), colormap gray, colorbar,
title('Digit 7 Sparse');
subplot(4,2,3), pcolor(flipud(a1r8)), colormap gray, colorbar,
title('Digit 8 Full');
subplot(4,2,4), pcolor(flipud(a1spr8)), colormap gray, colorbar,
title('Digit 8 Sparse');
subplot(4,2,5), pcolor(flipud(a1r9)), colormap gray, colorbar,
title('Digit 9 Full');
subplot(4,2,6), pcolor(flipud(a1spr9)), colormap gray, colorbar,
title('Digit 9 Sparse');
subplot(4,2,7), pcolor(flipud(a1r10)), colormap gray, colorbar,
title('Digit 0 Full');
subplot(4,2,8), pcolor(flipud(a1spr10)), colormap gray, colorbar,
title('Digit 0 Sparse');