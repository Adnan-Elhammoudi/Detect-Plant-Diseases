# Detect-Plant-Diseases
Applying Image Processing Techniques and Artificial Neural Network to Detect Plant Diseases.

The block diagram of the proposed system is shown in below Fig. The step by step proposed approach consists of leaf image database collection, it is started with capturing the images, but in our project, we will use a Dataset obtained from “Mendeley Data” website , since we do not have these materials, in this work we will be focusing on five different diseases for different vegetables, then segmented the affected Area using automatic thresholding value after converted image to L*a*b* Color Space. Now we have the interest area texture to extract the features using GLCM method and make some statistical calculations After All, the feature values are fed as input to the SVM classifier to classify the given image.

![image](https://user-images.githubusercontent.com/97694540/161538292-0d72a9a4-7cbf-4370-a323-edb73a119b8b.png)
[28] M.-L. Huang and Y.-H. Chang, “Dataset of Tomato Leaves,” vol. 1, May 2020, doi: 10.17632/ngdgg79rzb.1.
