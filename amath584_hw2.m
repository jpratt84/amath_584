%AMATH 584 HW 2
close all; clear; clc;
%% Part 1 - Eigenfaces
% load files
cropFolder = dir('CroppedYale');
cropFolder = cropFolder(3:40); %loads each subfolder - 38 total
uncropFolder = dir('yalefaces_uncropped');
uncropFolder = uncropFolder(4); %loads the only subfolder

%% Perform SVD on cropped faces

% The following loop converts each image in each subfolder into a column
%vector for SVD analysis.

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
%%
figure(1)
subplot(2,1,1), plot(diag(S),'ko','Linewidth',[2]),
xlabel('k'), ylabel('Singular Values \sigma_k'), title('Singular Values of Cropped Images')
set(gca,'Fontsize',12);
subplot(2,1,2), plot(cumsum(diag(S))/sum(diag(S)),'ko','Linewidth',[2])
xlabel('k'), ylabel('Percent of Total Singular Values'),title('Cumulative Percentage of Singular Value Spectrum')
set(gca,'Fontsize',12);
% First 600 modes comprise 80% of total sv spectrum. 600/2432 = 0.246711
% percent of original data
%% Reconstruct a cropped image using low-rank approximations

testImg = uint8(reshape(cropImg(:,1),m,n));
figure(2)
subplot(3,2,1),imshow(testImg),title('Cropped Test Image')
testImg = double(reshape(testImg,m*n,1)); %reshape test image into
%column vector
r = [1,10,100,300,600]; %rank approximations, smallest to largest

for p = 1:length(r)
    U_red = U(:,1:r(p)); %calculate reduced rank matrix U (feature space)
    reconImg = U_red*U_red'*testImg; %apply U_red to test image
    reconImg = reshape(reconImg,192,168); %reshape reconstructed image 
%     back to original dimensions
    subplot(3,2,p+1)
    imshow(uint8(reconImg))
    t = ['r = ',num2str(r(p))]; %adds rank number to title for each 
%     reconstructed image
    title(t)
end
%Need approximately rank 100 to reconstruct image with some level of detail
set(gca,'Fontsize',12);

%% Perform SVD on uncropped images

uncropImg = [];
for i=1:length(uncropFolder)
    name = uncropFolder(i).name;
    subfolder = ['yalefaces_uncropped','/',name];
    subdir =dir(subfolder);
    subdir = subdir(3:end);
    for j = 1:length(subdir)
        data = imread(subdir(j).name);
        [m,n] = size(data);
        data = reshape(data,m*n,1);
        uncropImg = [uncropImg data]; 
    end
end

uncropImg = double(uncropImg); %convert from uint8 to double for SVD
[U1,S1,V1] = svd(uncropImg,'econ');
%%
figure(3)
subplot(2,1,1),plot(diag(S1),'ko','Linewidth',[2]);
xlabel('k'), ylabel('Singular Values \sigma_k'),
title('Singular Values for Uncropped Faces');
set(gca,'Fontsize',12);
subplot(2,1,2),plot(cumsum(diag(S1))/sum(diag(S1)),'ko','Linewidth',2);
xlabel('k'), ylabel('Percentage of Total Singular Values'),
title('Cumulative Percentage of Singular Value Spectrum'),
set(gca,'Fontsize',12);

% Need about 56 modes for 80% of spectrum; 56/165 = 0.3394 of original data
%% Reconstruct an uncropped image using low-rank approximation
r = 60;
rApprox = U1(:,1:r)*S1(1:r,1:r)*V1(:,1:r)'; %compute low-rank approx of image
img = [1,75,125];
figure(4)
for i = 1:length(img)
    subplot(3,2,2*i-1)
    imshow(uint8(reshape(uncropImg(:,img(i)),m,n)));
    title('Original Uncropped Image'),set(gca,'Fontsize',12);
    subplot(3,2,2*i)
    imshow(uint8(reshape(rApprox(:,img(i)),m,n)));
    title('Reconstructed Uncropped Image, r=60'),set(gca,'Fontsize',12);
end