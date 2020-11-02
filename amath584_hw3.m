% James Pratt AMATH 584 HW 3

clear; close all; clc;

%% Part I - Modified Gram-Schmidt Orthogonalization
% See Algorithm 8.1 in Trefethen and Bau for basis for this algorithm

m = 25;
n = 23;
A = randn(m,n);
A = [A A(:,1) A(:,10)]; %append first and fifth columns to A to increase 
% condition number to something >>1
cond(A) %display condition number of A
[m,n] = size(A); %reassign m,n to match A above

% Use modified Gram-Schmidt orthogonalization
Q = zeros(m,n); %create empty Q matrix to store orthogonal vectors
R = zeros(m,n); %create empty upper triangular matrix R
for i = 1:n
    Q(:,i) = A(:,i); %create vector q_i, where q_i is column in Q
    for j = 1:i-1
        R(j,i) = Q(:,j)'*Q(:,i); %calculate normalization elements of R 
        % matrix
        Q(:,i) = Q(:,i) - R(j,i)*Q(:,j); %calculate orthogonal vectors q_i
    end
    R(i,i) = norm(Q(:,i))'; %calculate normalization values
    Q(:,i) = Q(:,i)/R(i,i); %normalize each column in the matrix Q
end

recon = Q*R; %reproduce A using QR approximation
figure(1),
subplot(2,2,1), imagesc(A), colorbar; %plot original matrix A
title('Original Matrix A');
subplot(2,2,2), imagesc(recon), colorbar;
title('Modified Gram-Schmidt Reconstruction');


% Use qrfactor from class
[Qc,Rc] = qrfactor(A);
subplot(2,2,3), imagesc(Qc*Rc), colorbar;
title('qrfactor Reconstruction');

% Use MATLAB's built-in QR function
[Qm,Rm] = qr(A);
subplot(2,2,4), imagesc(Qm*Rm), colorbar;
title('MATLAB QR Function Reconstruction');

% Calculate max error between original matrix and QR reconstructions
mgserror = max(A-recon,[],'all');
qrferror = max(A-Qc*Rc,[],'all');
mlerror = max(A-Qm*Rm,[],'all');
toterror = [mgserror qrferror mlerror];

figure(2)
bar(toterror), title('Max Error in Each Reconstruction');

%% Part II - Plotting a Polynomial

dx = 0.001;
x = 1.920:dx:2.080;
p = x.^9-18*x.^8+144*x.^7-672*x.^6+2016*x.^5-4032*x.^4+5376*x.^3-4608*x.^2+2304*x-512;

figure(3)
plot(x,p,'b','Linewidth',2), hold on;
plot(x, (x-2).^9,'r','Linewidth',2)
xlabel('x'), ylabel('p(x)'),
legend('Expanded p(x)','(x-2)^9');
title('Numerical Error in Plotting a Polynomial')

for i=1:length(x)
    residual(i) = p(i)-(x(i)-2).^9;
end

figure(4)
bar(x,residual)
xlabel('x'),ylabel('Residual')
title('Numerical Error between Plotting the Same Polynomial Two Ways')
max(residual), min(residual)

%% Part III - Conditioning of a Matrix
%% part a
%Fix m, vary n
m = 100;
n = [];
cnums = [];
for j=1:m-1
    n = j;
    A = randn(m,n);
    cnums(j) = cond(A);
end

figure(5)
plot(cnums,'o','Linewidth',2), xlabel('k'),ylabel('kth Condition Number'),
title('Condition Numbers for Fixed m, Varied n')

% Fix n, vary m
n1 = 100;
m1 = [];
cunms1 = [];
for k=(n1+1):2*n1
    m1 = k;
    A1 = randn(m1,n1);
    cnums1(k) = cond(A1);
end

figure(6)
plot(cnums1,'o','Linewidth',2), xlabel('k'),ylabel('kth Condition Number'),
xlim([101 200]);
title('Condition Numbers for Fixed n, Varied m')

%% Fix m,n, append column to A (part b)
m = 100;
n = 99;
A = randn(m,n);
alpha = cond(A);
A = [A A(:,1)]; %append first column of A to the end
beta = cond(A);
gamma = det(A);

%% Add noise to appended column (part c)
m = 50;
n = 49;

epsilon = [10^-1 10^-5 10^-10 10^-12 10^-14 10^-16];
cnums2 = [];
dets = [];
for l = 1:length(epsilon)
    A = randn(m,n);
    A = [A A(:,1)*epsilon(l)];
    cnums2(l) = cond(A);
    dets(l) = det(A);
end

figure(7)
subplot(2,1,1), bar(cnums2);
xlabel('k'),ylabel('kth Condition Number');
subplot(2,1,2), bar(dets);
xlabel('k'),ylabel('kth Determinant Value');