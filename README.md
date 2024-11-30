# Pneumonia-Detection

#### Objective
The goal of this project is to develop a reliable and efficient system for classifying X-ray images of lungs to determine whether they exhibit signs of pneumonia. This involves utilizing advanced image processing and machine learning techniques to analyze the features within the images and make accurate predictions. The [data](https://www.kaggle.com/datasets/pcbreviglieri/pneumonia-xray-images) set used in this project is an Adapted version of Paul Mooney's 'Chest X-Ray Images (Pneumonia)'. 

#### Pre-processing

The data preprocessing step utilized the ImageDataGenerator class to prepare the training and testing datasets. For the training set, we applied data augmentation techniques such as rescaling pixel values to [0,1] random shear and zoom transformations (range of 0.2), and horizontal flipping to enhance data variability and reduce overfitting. For the test set, only rescaling was applied to normalize pixel values while keeping the data unaltered for unbiased evaluation. This approach ensured robust training and accurate model performance assessment.

#### Model

In this project, we utilized ResNet50, a deep convolutional neural network pre-trained on ImageNet, to extract high-level features from the X-ray images. ResNet50 is well-known for its ability to capture rich and meaningful representations from visual data, making it a suitable choice for this medical imaging task.

Once the features were extracted, we applied t-Distributed Stochastic Neighbor Embedding (t-SNE), a dimensionality reduction technique, to visualize the extracted features in a two-dimensional space. This visualization revealed that the data points corresponding to the two classes (pneumonic and non-pneumonic) were largely linearly separable, indicating that the extracted features effectively distinguished between the two classes.

Based on this observation, we employed a Support Vector Machine (SVM) classifier with a linear kernel to perform the final classification. SVM was chosen due to its strength in handling linearly separable data with high accuracy. Upon evaluating the model on the testing dataset, we achieved an impressive accuracy of 93%, demonstrating the effectiveness of our approach in distinguishing between pneumonic and non-pneumonic X-ray images.





