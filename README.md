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
