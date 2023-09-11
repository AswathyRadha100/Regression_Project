# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# Import numpy for numerical operations
import numpy as np

# Import Pandas for data manipulation 
import pandas as pd

# Import the 'train_test_split' function to split the dataset into training and testing sets
from sklearn.model_selection import train_test_split 


# +
############################## PREPARE ZILLOW FUNCTION ##############################

def prep_zillow(df):
    '''
    This function takes in a dataframe
    renames the columns, drops nulls values in specific columns,
    changes datatypes for appropriate columns, and renames fips to actual county names.
    Then returns a cleaned dataframe
    '''
    # Rename columns
    df = df.rename(columns={
        'bedroomcnt': 'bedrooms',
        'bathroomcnt': 'bathrooms',
        'calculatedfinishedsquarefeet': 'area',
        'taxvaluedollarcnt': 'taxvalue',
        'fips': 'county',
        'lotsizesquarefeet': 'lotsqft'
    })

    # Drop specific columns
    df = df.drop(['taxamount'], axis=1)

    # Drop rows with null values in specific columns
    columns_with_nulls = ['bedrooms', 'bathrooms', 'area', 'taxvalue', 'yearbuilt', 'lotsqft']
    df = df.dropna(subset=columns_with_nulls)

    # Change data types to integers for appropriate columns
    make_ints = ['bedrooms', 'area', 'taxvalue', 'yearbuilt', 'lotsqft']

    for col in make_ints:
        df[col] = df[col].astype(int)

    # Map county codes to county names 
    df['county'] = df['county'].map({6037: 'LA', 6059: 'Orange', 6111: 'Ventura'})
    # Create dummy variables for the 'county' column with integer data type
    dummies = pd.get_dummies(df['county'],dtype=int)
    # Concatenate the dummy variables with the original dataframe
    df = pd.concat([df, dummies], axis=1)
     
        
    # Convert Column Names to Lowercase and Replace Spaces with Underscores
    df.columns = map(str.lower,df.columns)
    df.columns = df.columns.str.replace(' ','_')    
    
    return df


# +
################Function for cleaned integer date column################

def convert_transactiondate_to_int(df):
    '''
    Convert the 'transactiondate' column in the DataFrame to an integer format 'YYYYMMDD'.

    Parameters:
    df (DataFrame): The input DataFrame.

    Returns:
    DataFrame: A copy of the input DataFrame with the cleaned integer date column.
    '''

    # Convert the 'transactiondate' column to a datetime object
    df['transactiondate'] = pd.to_datetime(df['transactiondate'])

    # Extract year, month, and day as integers
    df['year'] = df['transactiondate'].dt.year
    df['month'] = df['transactiondate'].dt.month
    df['day'] = df['transactiondate'].dt.day

    # Concatenate year, month, and day as an integer
    df['transactiondate_int'] = df['year'] * 10000 + df['month'] * 100 + df['day']

    # Optionally, you can drop the intermediate columns 'year', 'month', and 'day' if not needed
    df = df.drop(['year', 'month', 'day'], axis=1)

    # 'transactiondate_int' now contains the date as an integer in the format YYYYMMDD
    
    df = df.drop('transactiondate', axis=1)

    return df



# -

############################## MinMaxScaler Function##############################
def min_max_scaler(df, cols):
    """
    Scale specified columns in a DataFrame using MinMaxScaler.
    
    Parameters:
    df (DataFrame): The input DataFrame.
    cols (list): List of column names to scale.
    
    Returns:
    DataFrame: A new DataFrame with specified columns scaled.
    """
    from sklearn.preprocessing import MinMaxScaler

    # Create a MinMaxScaler object
    scaler = MinMaxScaler()

    # Fit the scaler to the specified columns and transform them
    df[cols] = scaler.fit_transform(df[cols])

    return df


# +
############################## StandardScaler Function##############################

from sklearn.preprocessing import StandardScaler

def standard_scaler(df, cols):
    """
    Scales the specified columns in a dataframe using StandardScaler.
    
    Args:
        df (pd.DataFrame): The dataframe containing the columns to be scaled.
        cols (list): A list of column names to be scaled.
        
    Returns:
        pd.DataFrame: The dataframe with the specified columns scaled using StandardScaler.
    """
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df



# +
# cols=None means- if a list of column names to scale is not given
# when calling the function, it will assume that we want to scale all numeric columns in the DataFrame

def robust_scaler(df, cols=None):
    """
    Apply RobustScaler to specified or all numeric columns in a pandas DataFrame.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    cols (list, optional): List of column names to scale. If None, scales all numeric columns.

    Returns:
    pandas.DataFrame: The input DataFrame with scaled numeric columns.
    
    How to call:
    robust_scale_columns = ['column1', 'column2', 'column3']
    scaled_df = robust_Scaler(df, cols=robust_scale_columns)

    """
    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler()

    if cols is None:
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df[cols] = scaler.fit_transform(df[cols])

    return df


# +
######################## SPLIT DATA #####################################################
# Calculate and returns Percentage,Shape of Train, Validate, and Test Datasets


def split_data(df):
    """
    Split the DataFrame into train, validate, and test sets without any specific proportions.
    Returns three separate DataFrames for train, validate, and test sets.

    Parameters:
    - df: DataFrame containing the original dataset.

    Returns:
    - Three separate DataFrames for train, validate, and test sets.
    """
    
    # Use default values for train_size and seed
    train_size = 0.7
    seed = 42

    # Initial Split (train_size Train, (1-train_size) Validate + Test)
    train, validate_test = train_test_split(df, train_size=train_size, random_state=seed)

    # Secondary Split (50% Validate, 50% Test)
    validate, test = train_test_split(validate_test, train_size=0.5, random_state=seed)
    print(f"train: {len(train)} ({round(len(train)/len(df)*100)}% of {len(df)})")
    print(
        f"validate: {len(validate)} ({round(len(validate)/len(df)*100)}% of {len(df)})"
    )
    print(f"test: {len(test)} ({round(len(test)/len(df)*100)}% of {len(df)})")

    return train, validate, test



# -

#########################creating a function to isolate the target variable########################
def X_y_split_tar(df, target):
    '''
    This function takes in a dataframe and a target variable
    Then it returns the X_train, y_train, X_validate, y_validate, X_test, y_test
    and a print statement with the shape of the new dataframes
    
    Parameters:
    - df: DataFrame containing the dataset
    - target: The target variable to predict
    
    Returns:
    - X_train, y_train, X_validate, y_validate, X_test, y_test: Split data for training, validation, and testing
    '''  
    # Split the data into train, validate, and test sets
    train, validate, test = split_data(df)

    # Split the features and target variable for each set
    X_train = train.drop(columns=target)
    y_train = train[target]

    X_validate = validate.drop(columns=target)
    y_validate = validate[target]

    X_test = test.drop(columns=target)
    y_test = test[target]
        
    # Print the shape of the resulting datasets
    print(f'X_train -> {X_train.shape}')
    print(f'X_validate -> {X_validate.shape}')
    print(f'X_test -> {X_test.shape}')  
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test


# +
#########################creating a function to isolate the target variable and string variable########################


def X_y_split_tar_str(df, target, str_obj):
    '''
    This function takes in a dataframe, a target variable, and a list of string/object variables.
    It isolates the target variable and the specified string/object variables from the data.
    Then it returns the X_train, y_train, X_validate, y_validate, X_test, y_test datasets.
    Finally, it prints the shape of each dataset.

    # Parameters:
    # - df: DataFrame containing the dataset
    # - target: The target variable to predict
    # - str_obj: A list of string/object variables to isolate from the data
    '''

    # Split the data into train, validate, and test sets
    train, validate, test = split_data(df)

    # Create a list of columns to drop
    columns_to_drop = [target] + str_obj

    # Drop the target variable and string/object variables from each dataset
    X_train = train.drop(columns=columns_to_drop)
    y_train = train[target]

    X_validate = validate.drop(columns=columns_to_drop)
    y_validate = validate[target]

    X_test = test.drop(columns=columns_to_drop)
    y_test = test[target]
        
    # Print datasets' shapes
    print(f'X_train -> {X_train.shape}')
    print(f'X_validate -> {X_validate.shape}')
    print(f'X_test -> {X_test.shape}')  
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test



# -

########Function for SelectKBest##############
def select_kbest(X, y, k=2):
    '''
    Selects the top k best features based on their correlation with the target variable.

    Parameters:
    X (pandas.DataFrame): A dataframe representing numerical independent features.
    y (pandas.Series): A pandas Series representing the target variable.
    k (int, optional): The number of ideal features to select (default is 2).

    Returns:
    list: A list of the selected feature names.
    '''
    # Initialize SelectKBest with the f_regression scoring method
    kbest = SelectKBest(f_regression, k=k)
    
    # Fit SelectKBest to the data and target
    kbest.fit(X, y)
    
    # Create a boolean mask of selected features
    mask = kbest.get_support()
    
    # Return the names of the selected features
    return X.columns[mask]


# +
########Function for RFE##############

def rfe(X, y, k=2):
    '''
    Selects the top k best features using Recursive Feature Elimination (RFE) with Linear Regression.

    Parameters:
    X (pandas.DataFrame): A dataframe representing numerical independent features.
    y (pandas.Series): A pandas Series representing the target variable.
    k (int, optional): The number of ideal features to select (default is 2).

    Returns:
    list: A list of the selected feature names.
    '''
    # Initialize RFE with Linear Regression estimator
    rf = RFE(LinearRegression(), n_features_to_select=k)
    
    # Fit RFE to the data and target
    rf.fit(X, y)
    
    # Create a boolean mask of selected features
    mask = rf.get_support()
    
    # Return the names of the selected features
    return X.columns[mask]
