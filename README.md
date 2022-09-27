# Letter Recognition

## Overview

One of the most widespread and classical applications of machine learning is to do optical character/letter recognition, which is used in applications from physical sorting of mail at post offices, to reading license plates at toll booths, to processing check deposits at ATMs. Optical character recognition by now involves very well-tuned and sophisticated models. I built a model that uses attributes of images of four letters in the Roman alphabet – A, B, P, and R – to predict which letter a particular image corresponds to.

## Understanding the Dataset

The file Letters train.csv and Letters test.csv contains 2181 and 935 observations, each of which corresponds to a certain image of one of the four letters A, B, P and R. The images came from 20 different fonts, which were then randomly distorted to produce the final images; each such distorted image is represented as a collection of pixels, and each pixel is either “on” or “off”. For each such distorted image, we have available certain attributes of the image in terms of these pixels, as well as which of the four letters the image is. These features are described in Table 1.

## Analytic Models

### Start by predicting whether or not the letter is “B”.

#### Baseline

Before building any models, first consider a baseline method that always predicts the most frequent outcome, which is “not B”, and accuracy of this baseline method on the test set was 0.7465. 

#### Logistic Regression Model

Construct a logistic regression model to Predict whether or not the letter is a B, using the training set to build the model. And accuracy of logistic regression model on the test set, using a threshold of p = 0.5 was 0.9412, and Area under the ROC Curve(AUC) was 0.9775.

#### CART

Built a CART tree to predict whether or not a letter is a B, using the training set to build the model. I choose the ccp_alpha: 0.001 (from graph of code) picking the one that maximized actually the largest value of CP that achieves the maximum accuracy, and use 5-fold cross validation. The accuracy of this CART model on the test set was 0.9348.

#### Random Forest

Construct a Random Forest model to predict whether or not the letter is a B with the Random Forest parameters at their default values, and accuracy of this Random Forest model on the test set was 0.9840

#### Compare the accuracy of your logistic regression, CART, and Random Forest models.

Random Forest Regressor has best performance(accuracy: 0.9786) on the test set. Accuracy is more important in this application since this application is not about finding the context or importance of the text, but about predicting each letter accurately.

### Predict whether or not a letter is one of the four letters A, B, P or R.

#### Baseline

Baseline method predicts the most frequency letter class in the training set which is P, and the baseline accuracy on the test set was 0.2406.

#### LDA

Construct an LDA model to predict letter, using the training set to build the model. The accuracy of this LDA model on the test set was 0.90053.

#### CART

Built a CART model to predict whether or not letter, using the training set to build the model. I choose ccp_alpha: 0.0 (from graph of code) to set the cp parameter, I tried different values of Cp between 0 and 0.10, and used 5-fold cross-validation. And picking the one that maximized actually the largest value of CP that achieves the maximum accuracy. The accuracy of this CART model on the test set was 0.904812


#### Bagging of CART

Construct a bagging of CART models to predict, using the training set to build the model. This achieved by setting `max_features` equal to the total number of features in the `RandomFOrestCLassifier` package in Python. I used max_features 16 and achieved accruacy of this model on 0.94973.

#### Random Forest

I set the max features parameters using cross-validation which is 5-fold cross validation, trying different values, and I ended up picking the smallest value of max features that maximizes the accuracy which is 3. The accuracy of Random Forest on the test set was 0.9614.

#### Boosting

Apply boosting using the `GradientBoostingClassifier` function set `n_estimators` to 3300, `max_leaf_nodes` to 10, and leave all other parameters at their default values, and test set accuracy og gradient boosting model was 0.9754.

#### Compare the test set accuracy of your LDA, CART, bagging, Random Forest, and boosting models

I used the bootstrap to carefully test which model performs best, and Based on bootstrap, I would recommend Boosting model for Letter Recognition. Confidence interval of Boosting was [-0.002139037433155022, 0.9850267379679144]



Table 1
![Screen Shot 2022-09-27 at 2 33 37 PM](https://user-images.githubusercontent.com/25239743/192639814-276f76ab-4e7c-486d-b3dc-9735343260f8.png)

