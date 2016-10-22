# Machine Learning with Python

In this workshop we'll create a Supervised Learning example, predict if a person
is likely to develop diabetes.

#Asking the right question

After all, don't we already know that we want to predict? If a person will develop
diabetes?

We need a question that show us the direction of the data we gather, how we mold the data, how we interpret the solution,
and the criteria we use to validate the solution. What we need is more than one line question.

Our statement should to include:

1. Define scope (including data sources)
2. Define target performance
3. Determine context for usage

## Scope and data sources

Our initial statement is:

> Predict if a person will develop diabetes

Our statement should include the scope of our exercise, so here is very important to define the data sources
that we'll use to predict if a person will develop diabetes. The Pima Indian Diabetes study is a good source, this study conducted in the 90's about diabetes is in the UCI Machine Learning Repository, one of the best sources to get data for ML exercises.
So now, our statement will look like this:

> Using Pima Indian Diabetes data, predict which people will develop diabetes

## Performance targets

Our current statement is:

> Using Pima Indian Diabetes data, predict which people will develop diabetes

Our experiment will have a binary result (True or false) so, we have a 50% accuracy of predict the
correct result. Of course, we will want to have a better performance.
70% of accuracy is a common target, so we could take it.
Our statement will be:

> Using Pima Indian Diabetes data, predict with 70% or great accuracy, which people will develop diabetes

## Context

Our current statement is:

> Using Pima Indian Diabetes data, predict with 70% or great accuracy, which people will develop diabetes

We need to consider the context of our exercise, we are predicting disease. What does it mean? Does it
mean we are absolutely sure about the prediction? If we consider the practices of medical research, what is predicted
is the likelihood of developing the disease.

Now, we have our final solution statement:

> Using Pima Indian Diabetes data, predict with 70% or great accuracy, which people are likely to develop diabetes

# Preparing data

This is the most important step at Machine Learning workflow. 50%-80% of a ML project time
is spent getting, cleaning and organizing data.

The steps we will follow up are:

1. Find the data we need
2. Inspect and clean the data
3. Explore the data
4. Mold the data to Tidy data

## Getting data

We can get data from:

1. Google
2. Government databases
3. UCI Machine Learning Repository

We're going to get our data from UCI Machine Learning Repository. Pima Indian Diabetes data doesn't have many
of the issues you'll find with the data you can find in real world situations.

This data has 768 patient observation rows, each row has 10 columns, nine of these columns are feature columns such as
number of pregnancies, blood pressure, etc. and one class column (diabetes - true or false)

### Importing our data to Jupyter Notebook

We need to start our Notebook server. Type the following in terminal:

```bash
 jupyter notebook
 ```

we need to create a new notebook and import the following:

```python
#Import libraries
import pandas as pd                 # pandas is a dataframe library
import matplotlib.pyplot as plt     # matplotlib.pyplot plots data
import numpy as np                  # numpy provides N-dim object support

# do ploting inline instead of in a separate window
%matplotlib inline
```

### Load data

```python
df = pd.read_csv("./data/pima-data.csv")      # load Pima data.  Adjust path as necessary
```

### Cleaning the data

We need to remove:

1. Columns with no values
2. Correlated columns

Check if there are null values

```python
df.isnull().values.any()
```
Check for correlated columns

Let's define a function to draw a correlation table for our data set:

```python
def plot_corr(df, size=10):
    """
    Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot

    Displays:
        matrix of correlation between columns.  Blue-cyan-yellow-red-darkred => less to more correlated
                                                0 ------------------>  1
                                                Expect a darkred line running from top left to bottom right
    """

    corr = df.corr()    # data frame correlation function
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)   # color code the rectangles by correlation value
    plt.xticks(range(len(corr.columns)), corr.columns)  # draw x tick marks
    plt.yticks(range(len(corr.columns)), corr.columns)  # draw y tick marks
```

Now let's use this function:

```python
plot_corr(df)
```
We can see how `skin` and `thickness` are highly correlated. In this case we need to drop
this column to get rid of the correlation:

```python
del df['skin']
```

For a better performance, we need all values to be numerical.

```python
diabetes_map = {True : 1, False : 0}

df['diabetes'] = df['diabetes'].map(diabetes_map)
```
### Splitting data

```python
from sklearn.cross_validation import train_test_split

feature_col_names = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin', 'bmi', 'diab_pred', 'age']
predicted_class_names = ['diabetes']

X = df[feature_col_names].values
y = df[predicted_class_names].values
split_test_size = 0.30

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=split_test_size, random_state=42)
```

***Random state: setting a constant ensures that if we run the function again, the split will be identical.***


### Check if we have missing data

```python
print("# rows missing insulin: {0}".format(len(df.loc[df['insulin'] == 0])))
```

We have a lot of missing data. One way to fix this is to calculate the mean of the column and add
the result to the missing data columns:

```python
from sklearn.preprocessing import Imputer

fill_0 = Imputer(missing_values=0, strategy="mean", axis=0)

X_train = fill_0.fit_transform(X_train)
X_test = fill_0.fit_transform(X_test)
```

### Selecting our algorithm, training and testing our model

For this example let's use Random Forest algorithm:

```python
# Import Random Forest
from sklearn.ensemble import RandomForestClassifier
# Create Random Forest model
rf_model = RandomForestClassifier(random_state=42)
# Train model
rf_model.fit(X_train, y_train.ravel())
```

Let's use our model to predict with training data

```python
# Predict training data

rf_predict_train = rf_model.predict(X_train)
# Training metrics
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_train, rf_predict_train)))
```

Let's use our model to predict with testing data

```python
# Predict testing data

rf_predict_test = rf_model.predict(X_test)
# Training metrics
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, rf_predict_test)))
```

Weird? Let's see more details using `Confusion Matrix`

```python
print("Confusion Matrix")
# Note the use of labels for set 1=True to upper left and 0=False to lower right

print("{0}".format(metrics.confusion_matrix(y_test, rf_predict_test, labels=[1,0])))
print("")

print("Classification Report")
print(metrics.classification_report(y_test, rf_predict_test, labels=[1,0]))
```

We are having `Overfitting` issues with Random Forest algorithm.
We have some options to try to fix this issue:

1. Adjust current algorithm
2. Get more data
3. Switch algorithm

### Switch to Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(C=0.7, random_state=42)
lr_model.fit(X_train, y_train.ravel())
lr_predict_test = lr_model.predict(X_test)

# Print the results
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, lr_predict_test)))

print("Confusion Matrix")
# Note the use of labels for set 1=True to upper left and 0=False to lower right

print("{0}".format(metrics.confusion_matrix(y_test, lr_predict_test, labels=[1,0])))
print("")

print("Classification Report")
print(metrics.classification_report(y_test, lr_predict_test, labels=[1,0]))
```

Hmm it's seems we need to improve our algorithm, let's create a loop to try to
differents values of C, to see which will work better:

```python
C_start = 0.1
C_end = 5
C_inc = 0.1

C_values, recall_scores = [], []

C_val = C_start
best_recall_score = 0
while (C_val < C_end):
  C_values.append(C_val)
  lr_model_loop = LogisticRegression(C=C_val, random_state=42)
  lr_model_loop.fit(X_train, y_train.ravel())
  lr_predict_loop_test = lr_model_loop.predict(X_test)
  recall_score = metrics.recall_score(y_test, lr_predict_loop_test)
  recall_scores.append(recall_score)
  if(recall_score > best_recall_score):
    best_recall_score = recall_score
    best_lr_predict_test = lr_predict_loop_test

  C_val = C_val + C_inc

best_score_C_val = C_values[recall_scores.index(best_recall_score)]
print("1st max value of {0:.3f} occured at C={1:.3f}".format(best_recall_score, best_score_C_val))
```

It seems we are not achieving our 70% or more accuracy goal, what else we could try?

If we check our data, we can see that we have more non-diabetes results than diabetes results

```python
num_true = len(df.loc[df['diabetes'] == True])
num_false = len(df.loc[df['diabetes'] == False])
print("Number of True cases:  {0} ({1:2.2f}%)".format(num_true, (num_true/ (num_true + num_false)) * 100))
print("Number of False cases: {0} ({1:2.2f}%)".format(num_false, (num_false/ (num_true + num_false)) * 100))
```

Perhaps this imbalance is causing an issue. Luckily, algorithms like LogisticRegression includes a
hyper-parameter to compensate the imbalance.

```python
C_start = 0.1
C_end = 5
C_inc = 0.1

C_values, recall_scores = [], []

C_val = C_start
best_recall_score = 0
while (C_val < C_end):
  C_values.append(C_val)
  lr_model_loop = LogisticRegression(C=C_val, class_weight="balanced",random_state=42)
  lr_model_loop.fit(X_train, y_train.ravel())
  lr_predict_loop_test = lr_model_loop.predict(X_test)
  recall_score = metrics.recall_score(y_test, lr_predict_loop_test)
  recall_scores.append(recall_score)
  if(recall_score > best_recall_score):
    best_recall_score = recall_score
    best_lr_predict_test = lr_predict_loop_test

  C_val = C_val + C_inc

best_score_C_val = C_values[recall_scores.index(best_recall_score)]
print("1st max value of {0:.3f} occured at C={1:.3f}".format(best_recall_score, best_score_C_val))
```

Now let's train our model with those values:

```python
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(C=best_score_C_val, class_weight="balanced", random_state=42)
lr_model.fit(X_train, y_train.ravel())
lr_predict_test = lr_model.predict(X_test)

# Print the results
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, lr_predict_test)))

print("Confusion Matrix")
# Note the use of labels for set 1=True to upper left and 0=False to lower right

print("{0}".format(metrics.confusion_matrix(y_test, lr_predict_test, labels=[1,0])))
print("")

print("Classification Report")
print(metrics.classification_report(y_test, lr_predict_test, labels=[1,0]))
```

We finally achieve our goal, I guess we are done here..... NOT.
Now, we will persist our model so we could use it in a web application.

### Persisting our model

We are going to write this in Jupyter, and will create a new folder called `pkl_objects`
and into that folder, will create our model in the file `diabetes.pkl`

```python
import pickle
import os
dest = os.path.join('diabetesmodel', 'pkl_objects')
if not os.path.exists(dest):
     os.makedirs(dest)
pickle.dump(lr_model,
            open(os.path.join(dest, 'diabetes.pkl'), 'wb'),
            protocol=4)
```
Now we need to install some tools to build our web application.

### Installing our virtual environment

Let's run this in our terminal:

```bash
sudo pip3 install virtualenv
```
Now we will create a virtualenv in the webapp folder

```bash
virtualenv webapp
```

And we're going to activate the virtualenv

```bash
cd webapp
source bin/activate
```

### Install flask

```bash
pip install Flask
```

### Install WtForms

```bash
pip install wtforms
```

### Install NumPy, SciPy, Scikit-learn

```bash
pip install -t . numpy scipy scikit-learn
```

### Importing our model

Let's open `app.py` and replace in the line 10 the following:

```python
clf = pickle.load(open(os.path.join(cur_dir,'path/to/our/model'), 'rb'))
```

And let's test our app, we need to type in our terminal the following:

```bash
python3 app.py
```

And go into `http://localhost:5000`
