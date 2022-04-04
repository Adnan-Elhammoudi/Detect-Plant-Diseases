
                                                  

                                     %%Enhancement and Segmentation
                                     
[filename, pathname] = uigetfile({'*.*';'*.bmp';'*.jpg';'*.gif'}, 'Pick a Leaf Image File');
I = imread([pathname, filename]);
figure;
subplot(1,2,1);
imshow(I),title('Original Image');

                
%RGB=imresize(I,[256,256];

%convert to Lab color space
RGB=im2double(I);
C=makecform('srgb2lab','Adaptedwhitepoint',whitepoint('D65'));
Image=applycform(RGB,C);

% Image = rgb2lab (RGB);
channel1Min = 0.040;
channel1Max = 98.524;

% Define thresholds for channel 2 based on histogram settings
channel2Min = -13.466;
channel2Max = 19.740;

% Define thresholds for channel 3 based on histogram settings
channel3Min = 4.988;
channel3Max = 80;
%Create mask based on choosen threshold values
BW=(Image(:,:,1)>=channel1Min)&(Image(:,:,1)<= channel1Max)&(Image(:,:,2)>=channel2Min)&...
(Image(:,:,2)<= channel2Max)&(Image(:,:,3)>=channel3Min)&(Image(:,:,3)<= channel3Max);

%initial output based on input
maskedRGBImage=RGB;

%set background where BW is false to zero
maskedRGBImage(repmat(~BW,[ 1 1 3 ]))=0;


seg_img=maskedRGBImage;

subplot(1,2,2);
imshow(seg_img),title('Effected Area Image');

                                                  %% Feature Extraction
% Convert to grayscale
 img = rgb2gray(seg_img);
% Create the Gray Level Cooccurance Matrices (GLCMs)
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

hsv = rgb2hsv(I);
h = hsv(:, :, 1);
s = hsv(:, :, 2);
v = hsv(:, :, 3);

% Find the black pixles   It's Intensity will be less than .1
blackPixels = v < 0.1;

% Find only the black that is outside the leaf, not inside the leaf
background = xor(blackPixels, imclearborder(blackPixels));

% Mask the H, S, and V images.
h(background) = 0;
v(background) = 0;
v(background) = 0;

% Call anything with a hue of between 0.15 and 0.5 "healthy".
healthyImage = (h > 0.15) & (h < 0.5);
% Call anything else (that is not background) "diseased."
diseasedImage = ~healthyImage & ~background;

% Compute the diseased area fraction
entireLeafPixels = sum(~background(:));
Affected_Area = sum(diseasedImage(:)) / entireLeafPixels;
   
tf = horzcat([Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, Kurtosis, Skewness, IDM]);


                                                      %% SVM Model
 %Loading the Data features .mat file                                                     
 load('Train_Feat.mat') 
 
% Labling the DataSet
% 'Pepper bell Bacterial spot'
  Class1=repmat({'1'},970,1);
%  'Potato Early blight'
Class2=repmat({'2'},970,1);  
%   'Potato Late blight'
 Class3=repmat({'3'},970,1);
% 'Tomato Bacterial spot'
 Class4=repmat({'4'},2100,1);
%   'Tomato Late blight'
 Class5=repmat({'5'},1870,1);
 

 GroupTrain = [Class1; Class2; Class3;Class4 ;Class5];


SVMModels=cell(5,1);
Y=GroupTrain;
classes=unique(Y);
rng(1);

for j= 1:numel(classes)
    index=strcmp(Y',classes(j));
    SVMModels{j}=fitcsvm(Train_Feat,index,'ClassNames',[false true],'Standardize',true,...
        'KernelFunction','rbf','BoxConstraint',1);
end

TestSet=tf;
for j=1:numel(classes)
    [~,score]=predict(SVMModels{j},TestSet);
    scores(:,j)=score(:,2);

end


[~,maxScore]=max(scores,[],2);

result=maxScore;
       

                                                             %% Visualize Results
  msg = cell(4,1);

switch result
    case 1
        msg{1} = sprintf('Leaf name :Pepper bell');
        msg{2} = sprintf('Leaf Status : Effected');
        msg{3} = sprintf('disease name : Pepper bell Bacterial spot');
        msg{4} = sprintf('Affected Area is: %g%%',(Affected_Area*100));
        msgbox(msg)
   
    case 2
        msg{1} = sprintf('Leaf name :Potato');
        msg{2} = sprintf('Leaf Status : Effected');
        msg{3} = sprintf('disease name : Potato Early blight');
        msg{4} = sprintf('Affected Area is: %g%%',(Affected_Area*100));
        msgbox(msg)
   
    case 3
       msg{1} = sprintf('Leaf name :Potato');
        msg{2} = sprintf('Leaf Status : Effected');
        msg{3} = sprintf('disease name :Potato Late blight');
        msg{4} = sprintf('Affected Area is: %g%%',(Affected_Area*100));
        msgbox(msg)
    
    case 4
        msg{1} = sprintf('Leaf name :Tomato');
        msg{2} = sprintf('Leaf Status : Effected');
        msg{3} = sprintf('disease name : Tomato Bacterial spot ');
        msg{4} = sprintf('Affected Area is: %g%%',(Affected_Area*100));
        msgbox(msg)
        
     case 5
        msg{1} = sprintf('Leaf name : Tomato');
        msg{2} = sprintf('Leaf Status : Effected');
        msg{3} = sprintf('disease name : Tomato Late blight');
        msg{4} = sprintf('Affected Area is: %g%%',(Affected_Area*100));
        msgbox(msg)
     

end


