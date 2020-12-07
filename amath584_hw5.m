% J. Pratt AMATH 584 HW 5
clear; close all; clc;

%% Part I Eigenvalues and Power Iterations
%% Part (a) Symmetric Matrix
m = 10; %size of matrix A
A = randn(m); %generate random matrix of size m by m
As = A*A'; %force A to be symmetric
issymmetric(As) %returns 1 if As is truly symmetric, 0 else. Verification

[V, E] = eig(As); %return ground-truth matrix of eigenvectors Vt and
% diagonal matrix of eigenvalues Et

%% Part (b) Power Iteration Method (Symmetric Matrix)
v0 = randn(m,1);
v0 = v0/norm(v0); %normalize v0
vs = [];
vs(:,1) = v0;
lambdas = [];
maxIter = [5 10 50 100]; %set max number of iterations
error = [];
tol = 1e-10;
%% Test out accuracy as a function of the number of iterations 
% (Symmetric Matrix)
for i=1:length(maxIter)
    for k=1:maxIter(i)
        w = As*vs(:,k);
        vs(:,k+1) = w/norm(w);
        lambdas(k) = vs(:,k+1)'*As*vs(:,k+1);
        error(k) = abs(lambdas(k)-max(E,[],'all'));
        figure(1), hold on,
        subplot(4,1,i), plot(k,error(k),'ko','Linewidth',[2]);
        xlabel('Iteration Number'), ylabel('Error');
        if error(k)<tol
            break
        end

    end
end

%% Part (c) Rayleigh Quotient Iteration (Symmetric Matrix)
maxIter = 5; %set max number of iterations
tol = 10^-10; %set convergence criterion
evalR = []; %create array to store final eigenvalue approximations
count = []; %create array to store final iteration numbers

%% Rayleigh Quotient Iteration Loop (Symmetric Matrix)
for j=1:m
    v0 = V(:,j)+10^(-2); %create eigenvector approximation
    v0 = v0/norm(v0);
    alpha0 = v0'*As*v0; %create eigenvalue approximation
    alphas = [];
    alphas(1) = alpha0;
    vr = []; %create array of eigenvector approximations
    vr(:,1) = v0; %first eigenvector approximation
    w = zeros(m,1);
    for t=2:maxIter
        w = (As-alphas(t-1)*eye(m,m))\vr(:,t-1); %solve for w
        vr(:,t) = w/norm(w); %create new eigenvector approximation
        alphas(t) = vr(:,t)'*As*vr(:,t); %calculate new eigenvalue 
        % approximation
        err = abs(alphas(t)-E(j,j)); %determine difference between 
        % eigenvalue approx and true eigenvalue
        if err<tol
            break; %break out of loop if convergence criterion reached
        end
    end
    count(j) = t;
    evalR(j) = alphas(end); %store final eigenvalue approximations
    figure(2), hold on;
    plot(j,alphas(end),'bo','Linewidth',[2]); %plot eigenvalue number and 
    % value
    xlabel('j'),ylabel('jth Eigenvalue'),
    title('Eigenvalues Found via Rayleigh Iteration');
end

figure(3), plot(count,'ko','Linewidth',[2]), xlabel('j'),
ylabel('Max Iterations'),
title('Max Iterations to find the jth Eigenvalue');
ylim([0 5]);

%% Part (d) Repeat of parts (a) through (c) using non-symmetric matrix
clear, clc;
m = 10; %size of matrix A
A = randn(m); %generate random matrix of size m by m
issymmetric(A) %returns 1 if As is truly symmetric, 0 else

[Va, Ea] = eig(A); %return ground-truth matrix of eigenvectors Vt and
% diagonal matrix of eigenvalues Et

%% Power Iteration Method (Asymmetric Matrix)
v0 = randn(m,1);
v0 = v0/norm(v0); %normalize v0
vs = [];
vs(:,1) = v0;
lambdas = [];
maxIter = [5 10 50 100]; %set max number of iterations
error = [];
tol = 10^(-10);
%% Test out accuracy as a function of the number of iterations
% (Asymmetric Matrix)
for i=1:length(maxIter)
    for k=1:maxIter(i)
        w = A*vs(:,k);
        vs(:,k+1) = w/norm(w);
        lambdas(k) = vs(:,k+1)'*A*vs(:,k+1);
        error(k) = abs(abs(lambdas(k))-abs(max(Ea,[],'all')));
        figure(4), hold on,
        subplot(4,1,i), plot(k,error(k),'ko','Linewidth',[2]);
        xlabel('Iteration Number'), ylabel('Error');
        if error(k)<tol
            break;
        end
    end
end

%% Rayleigh Quotient Iteration (Asymmetric Matrix)
maxIter = 10; %set max number of iterations
tol = 10^(-10); %set convergence criterion
evalR = []; %create array to store final eigenvalue approximations
count = []; %create array to store final iteration numbers

% Rayleigh Quotient Iteration Loop (Asymmetric Matrix)
for j=1:m
    v0 = Va(:,j)+10^(-2); %create eigenvector approximation
    v0 = v0/norm(v0);
    alpha0 = v0'*A*v0; %create eigenvalue approximation
    alphas = [];
    alphas(1) = alpha0;
    vr = []; %create array of eigenvector approximations
    vr(:,1) = v0; %first eigenvector approximation
    w = zeros(m,1);
    for t=2:maxIter
        w = (A-alphas(t-1)*eye(m,m))\vr(:,t-1); %solve for w
        vr(:,t) = w/norm(w); %create new eigenvector approximation
        alphas(t) = vr(:,t)'*A*vr(:,t); %calculate new eigenvalue 
        % approximation
        err = abs(abs(alphas(t))-abs(Ea(j,j))); %determine difference between 
        % eigenvalue approx and true eigenvalue
        if err<tol
            break; %break out of loop if convergence criterion reached
        end
    end
    count(j) = t;
    evalR(j) = alphas(end); %store final eigenvalue approximations
    figure(5), hold on;
    plot(real(alphas(end)),imag(alphas(end)),'bo','Linewidth',[2]); %plot 
    % eigenvalues in complex plane
    xlabel('Re(jth eigenvalue)'),ylabel('Im(jth eigenvalue)'),
    title('Eigenvalues Found via Rayleigh Iteration (Asymmetric Matrix)');
end
plot(real(diag(Ea)),imag(diag(Ea)),'kx','Linewidth',[2]);

figure(6), plot(count,'ko','Linewidth',[2]), xlabel('j'),
ylabel('Max Iterations'),
title('Max Iterations to find the jth Eigenvalue (Asymmetric Matrix)');
ylim([0 10]);

%% Part II Yale Faces
clear; close all; clc;
cropFolder = dir('CroppedYale');
cropFolder = cropFolder(3:40); %loads each subfolder - 38 total

cropImg = []; %create array to store each image as a column vector
for i=1:length(cropFolder)
    name = cropFolder(i).name; %first element of cropFolder is name field
    subfolder = ['CroppedYale','/',name]; %access images in each subfolder
    subdir = dir(subfolder); %create new subdirectory for each face image 
    %set
    subdir = subdir(3:end); %take only the relevant fields
    for j=1:length(subdir)
        data = imread(subdir(j).name); %read each image in each 
        %subdirectory
        [m,n] = size(data); %compute size of each image for reshaping
        data = reshape(data,[m*n,1]); %reshape each image into column vector
        cropImg = [cropImg data]; %concatenate each image into matrix, 
        %each column is a separate image
    end
end

cropImg = double(cropImg); %convert from uint8 to double for SVD
[U,S,V] = svd(cropImg,'econ');

%% (a) Power Iterate on Matrix of Cropped Images
% Convert cropImg matrix into square correlation matrix
C = cropImg'*cropImg; %square
[n,n] = size(C);

maxEc = S(1,1); %establish ground truth eigenvalue (max)
maxVc = V(:,1); %establish ground truth eigenvector (assoc w/ max e-value)
v0 = randn(n,1);
v0 = v0/norm(v0); %normalize v0
vs = [];
vs(:,1) = v0;
lambdas = [];
maxIter = 100; %set max number of iterations
error = [];
tol = 1e-9;
%% Power Iteration Loop over square correlation matrix C
for k=1:maxIter
    w = C*vs(:,k);
    vs(:,k+1) = w/norm(w);
    lambdas(k) = vs(:,k+1)'*C*vs(:,k+1);
    error(k) = abs(sqrt(lambdas(k))-maxEc);
    figure(7), hold on, plot(k, error(k),'ko','Linewidth',[2]);
    xlabel('k'), ylabel('Error at kth Iteration'),
    title('Error vs Power Iteration (Cropped Images)');
    if error(k)<tol
        break;
    end
end

alpha = sqrt(lambdas(end));
valpha = vs(:,end);
acc = alpha-maxEc;

% Compare dominant eigenvalue and eigenvector found via power iteration to
% leading order SVD mode. 
% Dominant eigenvalue alpha = S(1,1), acc = 0
% Plot eigenvectors
figure(8);
hold on; plot(1:25, valpha(1:25),'bo',1:25, V(1:25,1),'rx','Linewidth',[2]);
xlabel('k'), ylabel('kth Value of Dominant Eigenvector'),
title('Dominant Eigenvector for Cropped Images');
legend('From Power Iterations','From SVD','Location','southoutside');
% eigenvectors equal up to sign, plotted first 10 values

%% Part (b) Randomized Sampling
% Stage A
[M,N] = size(cropImg);
K = 1000; %select number of random samples
Omega = randn(N,K); %initialize matrix of random projections
Y = cropImg*Omega; %project data onto Omega
[Q,R] = qr(Y,0); %perform QR decomposition

% Stage B
B = Q'*cropImg;
[u,s,v] = svd(B,'econ');
uapprox = Q*u;

%% Part (c) comparing randomized modes to true modes, singular value decay
figure(9),
x = 10;
plot(1:x,uapprox(1:x,1),'b',1:x,U(1:x,1),'b:',1:x,uapprox(1:x,2),'r',...
    1:x,U(1:x,2),'r:',1:x,uapprox(1:x,3),'g',1:x,U(1:x,3),'g:',...
    'Linewidth',[2]);
xlabel('Mode'),ylabel('U and Uapprox');
title('Randomized v. True Modes for Cropped Images (2000 Samples)');

z = 25;
figure(10),
subplot(2,1,1), plot(diag(s(1:z,1:z)),'bo','Linewidth',[2]),
xlabel('k'),ylabel('kth singular value'),
title('Singular Value Decay (Randomized Modes, 2000 Samples)');
subplot(2,1,2), plot(diag(S(1:z,1:z)),'ro','Linewidth',[2]);
xlabel('k'),ylabel('kth singular value'),
title('Singular Value Decay (True Modes)');