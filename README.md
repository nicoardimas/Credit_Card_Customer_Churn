# Credit Card Customer Churn Prediction
***

In this repository, I do machine learning modeling that can later be used to predict customer churn on credit card services. Here I use the bankchurn.csv dataset obtained from Kaggle. The case in this dataset is a binary classification with an unbalanced proportion of target variables, namely 16.1% for the negative class and the rest for the positive class. In this dataset there are several features that are used as parameters to make predictions, including:
* ***CLIENTNUM***: customer account number.
* ***Attrition_Flag***: customer status (Existing and Attrited).
* ***Customer_Age***: age of the customer.
* ***Gender***: gender of customer (M for male and F for female).
* ***Dependent_count***: number of dependents of customers.
* ***Education_Level***: customer education level (Uneducated, High School, Graduate, College, Post-Graduate, Doctorate, and Unknown).
* ***Marital_Status***: customer's marital status (Single, Married, Divorced, and Unknown).
* ***Income_Category***: customer income interval category (Less than $40K, $40K-$60k, $60K-$80K, $80K-$120K, $120K +, and Unknown).
* ***Card_Category***: type of card used (Blue, Silver, Gold, and Platinum).
* ***Months_on_Book***: period of being a customer (in months).
* ***Total_Relationship_Count***: the number of products used by customers in the bank.
* ***Months_Inactive_12_mon***: period of inactivity for the last 12 months.
* ***Contacts_Count_12_mon***: the number of interactions between the bank and the customer in the last 12 months.
* ***Credit_Limit***: credit card transaction nominal limit in one period.
* ***Total_Revolving_Bal***: total funds used in one period.
* ***Avg_Open_To_Buy***: the difference between the credit limit set for the cardholder's account and the current balance.
* ***Total_Amt_Chng_Q4_Q1***: increase in customer transaction nominal between quarter 4 and quarter 1.
* ***Total_Trans_Amt***: total nominal transaction in the last 12 months.
* ***Total_Trans_Ct***: the number of transactions in the last 12 months.
* ***Total_Ct_Chng_Q4_Q1***: the number of customer transactions increased between quarter 4 and quarter 1.
* ***Avg_Utilization_Ratio***: percentage of credit card usage.
***

### STEP 1
The first step, I will do Exploratory Data Analysis by using several functions and graphic visualization. At this stage I did not find any invalid data and duplicate data. Missing value in this dataset has been imputing with the status "Unknown". I found a data scale inequality between one feature and another features, where the range between the minimum and maximum values between one feature and another is not the same. In addition, in this dataset there are several features that have outlier values that can be seen through the boxplot. Then I saw that some features had abnormal data distribution, this I also confirmed by doing a normality test on numeric variables. In this dataset there are features that have a very strong correlation, namely Credit_Limit and Average_Open_to_Buy.

### STEP 2
The next step I took was Data Preprocessing with reference to the findings that I got in the Exploratory Data Analysis stage, the steps I took were as follows:

* Deleting the Credit Limit feature
  
  Based on the results of the heatmap, there are two features that have a very strong correlation, namely the Credit_Limit and Average_Open_to_Buy features. Here I will delete one of the features from the two features and the feature I choose to remove is Credit_Limit. Why did I remove the Credit_Limit? Why not Average_Open_to_Buy? My assumption is that Credit_Limit is obtained based on the provisions of the bank, while Average_Open_to_Buy is obtained based on customer behavior. In this project, I want to create a machine learning model to predict customer churn based on customer behavior, therefore I decided to remove the Credit_Limit and include Average_Open_to_Buy in the modeling process.

* Data Scaling
  
  Based on the results of the EDA, there are several features that have an abnormal data distribution. Here I standardize features that are normally distributed, namely Customer_Age, Months_on_Book, Total_Trans_Ct, and Total_Ct_Chng_Q4_Q1. Then the features that are not normally distributed are normalized. Scaling data can also overcome features that have outliers.

* Categorical Encoding
  
  At this stage, I change the categorical features to numeric, this is done because machine learning cannot read categorical data. There are two techniques that I use, namely Label Encoding and One Hot Encoding. I apply Label Encoding to features with an ordinal measurement scale and One Hot Encoding I apply to features with a nominal measurement scale.

* Split Data into Training and Testing sets
  
  Here I divide the dataset into training data and testing data with a proportion of 80% for training data and 20% for testing data. I use this training data to train the model and I will use the testing data to test the performance of the model.

* Dataset Balancing
  
  In the case of this dataset, there is a condition where the proportion between classes on the dependent variable (Existing Customers and Attributed Customers) is not balanced. This will affect the model's ability to predict customer status on new data, where the possibility of the model when making predictions for the minority class is not as good as the majority class. Therefore, I balance the data classes by using the SMOTE technique. Why the SMOTE technique? If I compare it with the oversampling technique, where the workings of the technique is to copy the data in the minority class up to a number of data in the majority class, so this technique allows for biased and less varied data. Meanwhile, when compared with undersampling, where the workings of this technique is to delete the data in the majority class up to a number of data in the minority class, so that this technique allows the reduction of information in the dataset. Therefore, I use the SMOTE technique, where this technique will add data in the minority class up to a number of data in the majority class based on the closest neighbors of the data points (this technique is similar to the KNN concept), so that by using this technique the resulting dataset is more varied and does not reduce the information from the dataset.


### STEP 3
The last stage, I did the modeling using 3 algorithms, namely Logistic Regression, Random Forest Classifier, and Gradient Boosting Classifier. The parameters that I use to see the performance of the model are the Accuracy, Precision, Recall, and Cross Entropy Loss scores. I use the Accuracy Score to see the model's ability to make predictions correctly for the entire data being tested. I use the Precision score to see the model's ability to correctly predict the positive class when compared to the overall results predicted as a positive class. I use the Recall score to see the model's ability to correctly predict the positive class when compared to the overall data that actually has a positive class. I use Cross Entropy Loss to calculate a score that summarizes the mean difference between the actual and predicted probability distributions for the positive prediction class. The ideal model has a Cross Entropy Loss value below the logK base e value (for the case of binary classification it has a value of K=2), where the value is 0.6931. If a model in the case of binary classification has a Cross Entropy Loss value below 0.6931, it means that our machine learning model has a lot of information, so that the model has a greater chance of accurately predicting new data.

