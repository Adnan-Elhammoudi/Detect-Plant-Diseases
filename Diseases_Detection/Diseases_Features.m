clc
close all 
clear all



myFolder = 'TrainingSet'; 
if ~isfolder(myFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  uiwait(warndlg(errorMessage));
  return;
end
filePattern = fullfile(myFolder);
matFiles = dir(filePattern);
M=length(matFiles)-2;
xx=1;
Train_Feat=[];

cd  TrainingSet;
 for k = 1:M
            k;
            itr=int2str(k)
            itr2=strcat(itr,'.jpg');
            I=imread(itr2);
           

%RGB=imresize(I,[256,256];

%convert to Lab color space
RGB=im2double(I);
C=makecform('srgb2lab','Adaptedwhitepoint',whitepoint('D65'));
I=applycform(RGB,C);



%Define Threshold for each Channel
channel1Min = 0.040;
channel1Max = 98.524;

% Define thresholds for channel 2 based on histogram settings
channel2Min = -13.466;
channel2Max = 19.740;

% Define thresholds for channel 3 based on histogram settings
channel3Min = 4.988;
channel3Max = 80;

%Create mask based on choosen threshold values
BW=(I(:,:,1)>=channel1Min)&(I(:,:,1)<= channel1Max)&(I(:,:,2)>=channel2Min)...
    &(I(:,:,2)<= channel2Max)&(I(:,:,3)>=channel3Min)&(RGB(:,:,3)<= channel3Max);


%initial output based on input
maskedRGBImage=RGB;

%set background where BW is false to zero
maskedRGBImage(repmat(~BW,[ 1 1 3 ]))=0;


seg_img=maskedRGBImage;
%figure;
%imshow(SegmentedImage),title('Segmented Image');

img = rgb2gray(seg_img);
glcms = graycomatrix(img);
% Derive Statistics from GLCM
stats = graycoprops(glcms,'Contrast Correlation Energy Homogeneity');
Contrast = stats.Contrast;
Correlation = stats.Correlation;
Energy = stats.Energy;
Homogeneity = stats.Homogeneity;
Mean = mean2(seg_img);
Standard_Deviation = std2(seg_img);
Entropy = entropy(seg_img);
RMS = mean2(rms(seg_img));
Variance = mean2(var(double(seg_img)));
a = sum(double(seg_img(:)));
Smoothness = 1-(1/(1+a));
Kurtosis = kurtosis(double(seg_img(:)));
Skewness = skewness(double(seg_img(:)));
% Inverse Difference Movement
m = size(seg_img,1);
n = size(seg_img,2);
in_diff = 0;
for i = 1:m
    for j = 1:n
        temp = seg_img(i,j)./(1+(i-j).^2);
        in_diff = in_diff+temp;
    end
end
IDM = double(in_diff);
   
ff = horzcat([Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, Kurtosis, Skewness, IDM]);


  Train_Feat = [Train_Feat;ff];

 end
   disp('Extracting Features Completed!');
   cd  ..
    save('Train_Feat.mat','Train_Feat');
      
     
