import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from scipy.stats import boxcox



'''
transformation.py contains functions that used to clean the data, calculating RFM features from
the transactional dataset, stadardizing RFM features, and handling the absent customers in current
given a custome set.
'''



def cleaning(clean_df):
    """
    Standard cleaning method used at the beginning of the data pipeline.

    Parameters
    ----------
    clean_df : DataFrame
        The DataFrame to be cleaned.

    Returns
    -------
    DataFrame
        The cleaned DataFrame with the following transformations applied:
        - Set the index to the 'Date' column and convert it to datetime.
        - Remove rows with 'All Customers' aggregation.
        - Remove data for September 2022.
    """
    # Set the index to the 'Date' column and convert it to datetime
    clean_df.set_index('Date', inplace=True)
    clean_df.index = pd.to_datetime(clean_df.index)

    # Eliminate 'All Customers' 
    all_cust_ids = clean_df[clean_df['Customer Group'] == 'All Customers']['Customer ID'].unique()
    clean_df = clean_df[~clean_df['Customer ID'].isin(all_cust_ids)]

    clean_df = clean_df[clean_df.index < '2022-09-01']

    return clean_df



def get_first_and_last_day(year, month):
    first_day = datetime(year, month, 1)
    if month == 12:
        last_day = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = datetime(year, month + 1, 1) - timedelta(days=1)
    return first_day, last_day


         
def rfm_calculation(transaction_data, start_date, end_date, customer_list):
    """
    Calculate Recency, Frequency, and Monetary (RFM) metrics for a list of customers 
    within a specified date range. For detailed definition, please see the paper.

    Parameters
    ----------
    transaction_data : DataFrame
        A DataFrame containing at least the following columns:
        - 'Customer ID'
        - 'date' (datetime or convertible to datetime)
        - 'Revenue' (numeric)
    start_date : str or datetime
        The start date of the analysis period (inclusive).
    end_date : str or datetime
        The end date of the analysis period (inclusive).
    customer_list : list
        A list of all sampled customer IDs.

    Returns
    -------
    DataFrame
        An RFM DataFrame indexed by 'Customer ID', containing columns (For detailed definition, please see the paper):
        - 'recency'
        - 'frequency'
        - 'monetary_value'
        - 'As of': The end date of the analysis period.
        Any customers in `customer_list` without transactions in the period 
        are assigned zeros for R, F, M.
    """

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    filtered_transactions = transaction_data[(transaction_data['date'] >= start_date) & (transaction_data['date'] <= end_date)].copy()
    
    recency = filtered_transactions.groupby('Customer ID').agg(
        t_n=('date', 'max')
    ).reset_index()
    recency['recency'] = (recency['t_n'] - start_date) / (end_date - start_date)
    
    purchase_days = filtered_transactions.groupby(['Customer ID', 'date']).size().reset_index(name='purchase_count')
    frequency = purchase_days.groupby('Customer ID').size().reset_index(name='purchase_days')
    
    frequency['frequency'] = frequency['purchase_days'] / (end_date - start_date).days
    
    monetary = filtered_transactions.groupby('Customer ID')['Revenue'].sum().reset_index(name='total_revenue')
    monetary['monetary_value'] = monetary['total_revenue'] / (end_date - start_date).days

    resulting_rfm = pd.merge(recency[['Customer ID', 'recency']], frequency[['Customer ID', 'frequency']], on='Customer ID')
    resulting_rfm = pd.merge(resulting_rfm, monetary[['Customer ID', 'monetary_value']], on='Customer ID')
    customer_df = pd.DataFrame(customer_list, columns=['Customer ID'])
    resulting_rfm = pd.merge(customer_df, resulting_rfm, on='Customer ID', how='left')

    # Fill missing values for customers with no purchases during the period
    resulting_rfm['recency'].fillna(0, inplace=True)
    resulting_rfm['frequency'].fillna(0, inplace=True)
    resulting_rfm['monetary_value'].fillna(0, inplace=True)  
    resulting_rfm['As of'] = end_date

    resulting_rfm.set_index('Customer ID', inplace=True)
    
    return resulting_rfm

def scaling(purchases_sorted_rfm):
    """
    Scale the 'Revenue', 'Recency', and 'Frequency' columns of each DataFrame in the list using MinMax scaling.

    Parameters
    ----------
    purchases_sorted_rfm : list of pd.DataFrame
        List of DataFrames containing RFM (Revenue, Recency, Frequency) data for different months.
        Each DataFrame should have columns: 'monetary_value', 'recency', 'frequency'.

    Returns
    -------
    list of pd.DataFrame
        List of DataFrames with scaled 'Revenue', 'Recency', and 'Frequency' columns.
    """
    scaler = MinMaxScaler()

    for month in purchases_sorted_rfm:
        month[['Revenue', 'Recency', 'Frequency']] = scaler.fit_transform(month[['monetary_value', 'recency', 'frequency']])

    purchases_rfm_stand = purchases_sorted_rfm.copy()

    return purchases_rfm_stand



def scaling_revenue(purchases_sorted_rfm_list):
    """
    Scale the revenue column of the calculated RFM dataframe.
    """

    scaler = MinMaxScaler()

    for purchases_sorted_rfm in purchases_sorted_rfm_list:
        purchases_sorted_rfm[['revenue']] = scaler.fit_transform(purchases_sorted_rfm[['monetary_value']])

    purchases_rfm_stand = purchases_sorted_rfm_list.copy()

    return purchases_rfm_stand
