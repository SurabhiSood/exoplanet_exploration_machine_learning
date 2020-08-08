### Goal

Use various machine learning classification models to predict candidate exoplanet classifications. Use Grid Search to increase the accuracy of the model.

### Process

Data Cleaning and Pre-Processing

Data was first read in from a csv file, and null columns and null rolls were dropped. After this, there were still several columns available to select as features to train the model on. Wanting to use the most relevant features, I found the top ten features of the data set ranked by feature importances by using ExtraTreesClassifier() and stored those top ten features as a series to be used as my X values. The koi_disposition column contained the classification values of each exoplanet candidate and would be used as my y values.

With my X and y values set, I split the data into training and testing sets using train_test_split with stratify=y to ensure that there was an even distribution of classification values in both data sets. Then, I used MinMaxScaler to scale both sets of X data.

This method was used for all four models.

**Logistic Regression**

I initialized the model using LogisticRegression() and fit the model using the training data. I then scored the model using both the training and testing data. Both sets scored fairly well, with the training data at 85.14% and the testing data at 84.38%.

I again used GridSearchCV to further tune the parameters to create a better scoring model. The parameters were set to explore different C values using both L1 and L2 penalties as regularization methods. I then fit a new model using this grid and found the best parameters, before predicting on the test data. This new model's score was 86.99%.

**K-Nearest Neighbors**

To find the best k value to use in this model, I created a loop to run through a set of possible k values. Because there are three possible classifications, I started the range of k values at 5 with a step of 3 to avoid any even split of classifications. Comparing the training and testing scores of each model, it looked like k=20 was the best value from the graph. Accuracy was 82.7%

To further tune the model’s parameters, I used GridSearchCV and expanded the possible values of k. I then retrained the model using the best parameter found and scored the model using the test set of data.

Grid Search also found k=22 to be the best k value, with an accuracy of 82.68%, so this model was not improved by the use of Grid Search.

**Trees**

I initialised the 'clf' using DecisionTreeClassifier() and checked accuracy of the model to be at 85.06% with the scaled X-axis.

I initialized the 'rf' using RandomForestClassifier() and set the number of trees to 200 (n_estimators=200). I then fit and scored the model, with the testing data scoring at 89.35%.

Using Grid Search, I explored different parameters including n_estimators, max_depth, min_samples_split, and min_samples_leaf. Again, I used this grid to train a new model, before predicting and scoring. After training the new model, accuracy as 88.23%

**Support Vector Machine**

I initialized the model with SVC() and set the kernel to linear before training and scoring the model, with the testing data scoring at 84.15%.

With Grid Search, I explored various C values, gamma values, and linear and rbf kernels. After training the new model, accuracy increased to 89.41% with grid parameters as {'max_depth': 25, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 400}

### Summary

Random forest model had the highest accuracy at 89.1%.