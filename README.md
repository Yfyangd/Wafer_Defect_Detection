# A Deep Learning Model for Identification of Defect Patterns in Semiconductor Wafer Map
The semiconductors are used as various precision components in many electronic products. Each layer must be inspected of defect after drawing and baking the mask pattern in wafer fabrication. Unfortunately, the defects come from various variations during the semiconductor manufacturing and cause massive losses to the companies' yield. If the defects could be identified and classified correctly, then the root of the fabrication problem can be recognized and eventually resolved. 

Machine learning (ML) techniques have been widely accepted and are well suited for such classification and identification problems. In this study, we employ convolutional neural networks (CNN) and extreme gradient boosting (XGBoost) for wafer map retrieval tasks and the defect pattern classification. CNN is the most famous deep learning architecture. The recent surge of interest in CNN is due to the immense popularity and effectiveness of convnets. XGBoost is the most popular machine learning framework among data science practitioners, especially on Kaggle, which is a platform for data prediction competitions where researchers post their data and statisticians and data miners compete to produce the best models. CNN and XGBoost are compared with a random decision forests (RF), support vector machine (SVM), adaptive boosting (Adaboost), and the final results indicate a superior classification performance of the proposed method.

Our experimental result demonstrates the success of CNN and extreme gradient boosting techniques for the identification of defect patterns in semiconductor wafers. The overall classification accuracy for the test dataset of CNN and extreme gradient boosting is 99.2%/98.1%. We demonstrate the success of this technique for the identification of defect patterns in semiconductor wafers. We believe this is the first time accurate computational classification in such task has been reported achieving accuracy above 99%.

# Data Preprocess
we use Min-Max scaling to normalize the data. A Min-Max scaling will scale the data between the 0 and 1. It is typically done via the following equation:

## Xmin-max = (X-Xmin)/(Xmax-Xmin)

In this approach, the data is scaled to a fixed range (typically 0 to 1). We end up with a smaller standard deviation, which can suppress the effects of outliers.

# Feature Extraction
Singular Value Decomposition (SVD) has been successfully used to feature extraction in various area, such as physiological signals, motor faults classification. By SVD, we can extract orthogonal matrix (U), diagonal matrix (s), unitary matrix (V) from original image:

# Model Training
The input wafer map image size is 500 X 600. We have three convolutional layers. In the first convolutional layer, the receptive field size is 3 X 3 and stride is 1. In the second and third convolutional layer, the receptive field size is 2 X 2 and stride is 1. The first convolutional layer have 32 channels, the second and third convolutional layer has 64 channels. The rectified linear unit (ReLU) activation is used for each convolutional layer. The max pooling size is 3 X 3. One dense layer with 128 channels is added after convolution/pooling layers. The fully connected (FC) layer with channels 128 is added with sigmoid activation. After dropout, another fully connected layer with channels 6 (number of defect classes) is added. The last layer is the softmax layer for the class probability calculation.

# Result
We train our models as follows. First, we split the images randomly into 9384 (80%) training data set and 2346 (20%) test data set. The training data set is used for training our models, and the testing data set is used for the validation. In CNN model, we split 25% from training data set for validation in each epoch. The training accuracy after the 20 epoch is 98.8% and the validation accuracy is 97.9%. 

Table as below is the confusion matrix and it shows the per-class classification accuracy in percentage. Due to the confidentiality reason, we can only provide per-class accuracy, not the absolute number of wafers. Most of the class accuracy is greater than 98%. The overall accuracy is 99.2%.

| Type | PAR | RE | OM | DC | Ru | PC |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
|PAR | 99.7 | 0.2 | 0.9 | 0.0 | 0.0 | 0.3 |
|RE | 0.0 | 99.5 | 0.0 | 0.2 | 0.5 | 0.0 |
|OM | 0.1 | 0.0 | 98.3 | 0.0 | 0.0 | 0.0 |
|DC | 0.0 | 0.0 | 0.4 | 98.3 | 1.0 | 0.3 |
|Ru | 0.1 | 0.3 | 0.0 | 1.5 | 98.5 | 0.0 |
|PC | 0.0 | 0.3 | 0.4 | 0.0 | 0.0 | 99.4 |
