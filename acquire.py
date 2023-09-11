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
#------------------- Tabular data imports--------------------- 
import pandas as pd
import numpy as np

#-------------------Import custom library --------------------- 
import env

#-------------------Import SQLAlchemy--------------------------
from sqlalchemy import create_engine, text

#-------------------ignore warnings----------------------------
import warnings
warnings.filterwarnings("ignore")

#----------------- Import the 'os' module to access operating system functionality---------
import os


# -

##############Function to fetch data from zillow database######################
def get_zillow_data():
    '''
    This function acquires zillow.csv if it is available,
    otherwise, it makes the SQL connection and uses the query provided
    to read in the dataframe from SQL.
    If the CSV is not present, it will write one.
    '''
    filename = "zillow.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col=0)
    else:
        # Create the URL for the database connection
        url = env.get_db_url('zillow')

        # Define the SQL query
        zillow_query = '''
            SELECT
                bedroomcnt,
                bathroomcnt,
                calculatedfinishedsquarefeet,
                taxvaluedollarcnt,
                yearbuilt,
                taxamount,
                fips,
                lotsizesquarefeet
                
                
            FROM
                properties_2017 
            
            WHERE
                propertylandusetypeid = 261;
        '''

        # Read the SQL query into a dataframe
        df = pd.read_sql(zillow_query, url)

        # Write the dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df



# +
##############Function to fetch data from 'predictions_2017' table(zillow database)######################

def get_zillow_pred_data():
    '''
    This function acquires zillow.csv if it is available,
    otherwise, it makes the SQL connection and uses the query provided
    to read in the dataframe from SQL.
    If the CSV is not present, it will write one.
    '''
    filename = "zillow_pred.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col=0)
    else:
        # Create the URL for the database connection
        url = env.get_db_url('zillow')

        # Define the SQL query
        zillow_query = '''
            SELECT
                transactiondate,logerror,parcelid
                
                
            FROM
                predictions_2017; 
            
            
        '''

        # Read the SQL query into a dataframe
        df = pd.read_sql(zillow_query, url)

        # Write the dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df

# -


