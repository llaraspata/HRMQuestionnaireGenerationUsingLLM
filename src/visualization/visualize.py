"""
This module contains functions for visualizing data.
"""
import pandas as pd
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def get_head_using_essential_columns(dataframe, essential_columns):
    """
    Print the head of the dataframe using the essential columns.

    Parameters:
        - dataframe (DataFrame): The DataFrame to show.
        - essential_columns (list): The list of essential columns.
    """
    return dataframe[essential_columns].head()


def count_null_values_for_columns(df):
    """
    Counts the number of null values in each column of a DataFrame.

    Parameters:
        - df (DataFrame): The input DataFrame.

    Returns:
        - dict: A dictionary where keys are column names and values are the count of null values.
    """
    null_counts = {}

    for column in df.columns:
        null_count = df[column].isnull().sum()
        
        null_counts[column] = null_count
    
    null_counts_sorted = dict(sorted(null_counts.items(), key=lambda x: x[1], reverse=True))
    
    return null_counts_sorted


def plot_null_values(null_counts, df_name, color="#d2056d"):
    """
    Plot the number of null values for each column in a DataFrame.

    Parameters:
        - null_counts (dict): A dictionary where keys are column names and values are the count of null values.
        - df_name (str): The name of the DataFrame.
        - color (str, optional): The color for the histogram bars. Default is "#d2056d".
    """
    null_counts_filtered = {col: count for col, count in null_counts.items() if count != 0}

    if not null_counts_filtered:
        print("No columns with non-zero null counts.")
        return

    columns = list(null_counts_filtered.keys())
    counts = list(null_counts_filtered.values())

    fig = go.Figure(data=[go.Bar(x=columns, y=counts, marker_color=color)])

    fig.update_layout(title=f'{df_name} - Number of Null Values by Column',
                      xaxis_title='Column Name',
                      yaxis_title='Null Values Count',
                      )

    fig.show()


def average_number_per_column(df, column_name):
    """
    Calculate the average number of rows grouped by the specified column name.

    Parameters:
        - df (DataFrame): DataFrame containing data.
        - column_name (str): Name of the column to group by.

    Returns:
        - int: The average number of questions per questionnaire.
    """
    avg = df.groupby(column_name).size().mean()
    
    return round(avg)


def count_words(sentence):
  """
    Counts the number of words in a sentence.

    Parameters:
        - sentence (str): The input sentence.
  """
  splitted = sentence.split(" ")

  return len(splitted)


def plot_length_distribution(df, column_name, name, color="#d2056d"):
    """
        Plots the distribution of the length of the specified column.

        Parameters:
        - df (DataFrame): The DataFrame containing the data.
        - column_name (str): The name of the column whose length distribution is plotted.
        - name (str): The name of the entity whose length distribution is plotted.
        - color (str, optional): The color for the histogram bars. Default is "#d2056d".
    """
    lens = [count_words(str(sentence)) for sentence in df[column_name]]

    mean_train_len = np.mean(lens)
    max_train_len = max(lens)
    min_train_len = min(lens)

    var_name = column_name.replace("_", " ") + " LENGTH"

    fig = px.histogram(lens,
                       labels={'value': 'Length', 'count': 'Frequency', 'variable': var_name},
                       title=f'Length Distribution - {name} [Mean: {int(mean_train_len)} - Max: {max_train_len} - Min: {min_train_len}]',
                       color_discrete_sequence=[color])

    fig.update_layout(xaxis_title='Length', yaxis_title='Frequency', bargap=0.1)

    fig.show()