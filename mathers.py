import aiosqlite
import pandas as pd
import pandas_ta as ta
import asyncio
import logging
import numpy as np
import json
import os
import sqlite3
from skimage.restoration import denoise_tv_chambolle
from multiprocessing import Queue, Process, Pool, cpu_count
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy.optimize as opt
import numpy as np
import traceback
import concurrent.futures
import traceback
import multiprocessing
import cProfile
import io
import pstats

# Define the timeframes and corresponding database paths
timeframes = {
    '15m': 'fifteenminutebtc'        
}

DB_PATH_DATA = './data/btcdata1.db'

def setup_logging():
    # Create a formatter for the log messages
    formatter = logging.Formatter('%(processName)-10s %(levelname)-8s %(message)s')
    
    # Create a handler for writing log messages to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)    
   
    
    # Get the root logger and set its level to DEBUG
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)
    
    # Add the console and file handlers to the root logger
    logger.addHandler(console_handler)

def get_start_index(table_name):
    # Connect to the SQLite database
    conn = sqlite3.connect(DB_PATH_DATA)
    cursor = conn.cursor()

    # Fetch the last non-zero value from the 'actualClass' column
    query = f"SELECT time FROM {table_name} WHERE actualClass != 0.0 ORDER BY time DESC LIMIT 3;"
    
    try:
        cursor.execute(query)
        result = cursor.fetchone()
        
        # Close the database connection
        conn.close()

        # If a result is found, print and return the timestamp
        if result:
            print(f"Last timestamp for {table_name}: {result[0]}")
            return result[0]
        else:
            print(f"No pivot found for {table_name}")
            return 0
    except Exception as e:
        print(f"Error: {e}")
        # Close the database connection in case of an error
        conn.close()
        return 0


async def async_read_from_db(table_name):
    db_path = DB_PATH_DATA  # Fetch the correct database path using the function
    start_index = get_start_index(table_name)  # Fetch the start index
    
    async with aiosqlite.connect(db_path) as db:
        async with db.cursor() as cursor:
            await cursor.execute(f"SELECT * FROM {table_name} WHERE time > ?", (start_index,))           
            rows = await cursor.fetchall()
            print(f"Mathers Iteration, Fetched {len(rows)} rows from {table_name} starting from timestamp {start_index}.")
            columns = [column[0] for column in cursor.description]
            df = pd.DataFrame(rows, columns=columns)
            df = apply_exponential_smoothing_to_columns(df)
            df = parallel_transform(df)
    return df


def write_to_db(df, table_name, db_path=DB_PATH_DATA):
    if df.empty:
        print(f"Skipping write for {table_name} as the DataFrame is empty.")
        return

    # Round numeric columns to 2 decimal places
    numeric_cols = df.select_dtypes(include=['float64']).columns
    df[numeric_cols] = df[numeric_cols].round(2)

    # Connect to the SQLite database
    with sqlite3.connect(db_path) as conn:
        # Write the dataframe to the specified table in the SQLite database
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        
    print(f"Direct write completed for {table_name}.")

    
       
    

# Indicator settings
rsiLength = 10
rsiOverbought = 60
rsiOversold = 40
periodval = 7

# Calculate RSI Swing Indicator
async def calculate_rsi_swing_indicator(df1):
    DZbuy = 0.1
    DZsell = 0.1
    Period = periodval
    Lb = 20

    RSILine = ta.rsi(df1['F_close'], Period)
    df1['RSI_Highest'] = RSILine.rolling(Lb).max()
    df1['RSI_Lowest'] = RSILine.rolling(Lb).min()
    jh = RSILine.rolling(Lb).max()
    jl = RSILine.rolling(Lb).min()
    jc = (ta.ema((jh - jl) * 0.5, Period) + ta.ema(jl, Period))
    Hiline = jh - jc * DZbuy
    Loline = jl + jc * DZsell
    return (
        4 * RSILine
        + 3 * RSILine.shift(1)
        + 2 * RSILine.shift(2)
        + RSILine.shift(3)
    ) / rsiLength

async def main_processing_loop(table_name, rsiOverbought, rsiOversold):
    try:        
        df = await async_read_from_db(table_name) 
              
        rsi_swing_indicator = await calculate_rsi_swing_indicator(df)
        df['RSI_Swing_Indicator'] = rsi_swing_indicator
        
        df['isOverbought'] = df['RSI_Swing_Indicator'] >= rsiOverbought
        df['isOversold'] = df['RSI_Swing_Indicator'] <= rsiOversold             

        columns_to_exclude = ['RSI_Highest', 'RSI_Lowest', 'RSI_Swing_Indicator'] 
        df_selected = df[[col for col in df.columns if col not in columns_to_exclude]]
        
        return table_name, df_selected
    except Exception as e:
        print(f"mathers_iteration, Mr Anderson {table_name}: {e}")
        return None, None  # Return sentinel values
    





def compute_heikin_ashi(df, column):
    if column == 'F_HA_close':
        df[column] = (df['F_high'] + df['F_low'] + df['F_close']) / 3
    elif column == 'F_HA_open':
        df[column] = df['F_close'].shift(1)
        df[column].fillna((df['F_high'] + df['F_low']) / 2, inplace=True)
        df[column] = ((df[column] + df['F_close']).shift(1)) / 2
        df[column].fillna((df['F_high'] + df['F_low']) / 2, inplace=True)
    elif column == 'F_HA_high':
        df[column] = df[['F_high', 'F_open', 'F_close']].max(axis=1)
    elif column == 'F_HA_low':
        df[column] = df[['F_low', 'F_open', 'F_close']].min(axis=1)
    return df




def exponential_smoothing(series, alpha=0.3):
    """Apply Exponential Smoothing."""
    series = pd.Series(series)  # Ensure that it's a Series
    result = [series.iloc[0]]
    result.extend(
        alpha * series.iloc[n] + (1 - alpha) * result[n - 1]
        for n in range(1, len(series))
    )
    return result

def apply_exponential_smoothing_to_columns(df):
    """
    Apply exponential smoothing to specific columns of a dataframe.
    """
    columns_to_smooth = {
        'F_open': 'F_smoothedOpen',
        'F_high': 'F_smoothedHigh',
        'F_low': 'F_smoothedLow',
        'F_close': 'F_smoothedClose'
    }
    
    for original_col, smoothed_col in columns_to_smooth.items():
        df[smoothed_col] = exponential_smoothing(df[original_col].tolist())        
    return df

def total_variation_denoising_alpha(input_data, weight=10):
    """
    Apply Total Variation Denoising using scikit-image's denoise_tv_chambolle.
    
    Parameters:
    - input_data: Input data
    - weight: Denoising weight. Higher values produce a more denoised result, 
              but might also remove more details.
    
    Returns:
    - Denoised data
    """
    return denoise_tv_chambolle(input_data, weight=weight)

def total_variation_denoising_beta(input_data, weight=20):
    """
    Apply Total Variation Denoising using scikit-image's denoise_tv_chambolle.
    
    Parameters:
    - input_data: Input data
    - weight: Denoising weight. Higher values produce a more denoised result, 
              but might also remove more details.
    
    Returns:
    - Denoised data
    """
    return denoise_tv_chambolle(input_data, weight=weight)

def total_variation_denoising_charlie(input_data, weight=100):
    """
    Apply Total Variation Denoising using scikit-image's denoise_tv_chambolle.
    
    Parameters:
    - input_data: Input data
    - weight: Denoising weight. Higher values produce a more denoised result, 
              but might also remove more details.
    
    Returns:
    - Denoised data
    """
    return denoise_tv_chambolle(input_data, weight=weight)

def total_variation_denoising_delta(input_data, weight=200):
    """
    Apply Total Variation Denoising using scikit-image's denoise_tv_chambolle.
    
    Parameters:
    - input_data: Input data
    - weight: Denoising weight. Higher values produce a more denoised result, 
              but might also remove more details.
    
    Returns:
    - Denoised data
    """
    return denoise_tv_chambolle(input_data, weight=weight)


#######################################################################################################


def compute_Target(df, next_pivot_high_price, table_name, current_pivot):
    try:
        print(f"[{table_name}]compute_Target")

        # Determine the rows to be updated based on the actualClass values
        if current_pivot == 1:
            # Find the next pivot that's a '2' after the current index
            next_2_idx = df[(df['actualClass'] == 2) & (df.index > df[df['actualClass'] == current_pivot].index[0])].index.min()
            mask = (df.index < next_2_idx) & (df['actualClass'] == 1)
            column_name = 'sellTarget'
        elif current_pivot == 2:
            # Find the next pivot that's a '1' after the current index
            next_1_idx = df[(df['actualClass'] == 1) & (df.index > df[df['actualClass'] == current_pivot].index[0])].index.min()
            mask = (df.index < next_1_idx) & (df['actualClass'] == 2)
            column_name = 'buyTarget'
        else:
            return df  # If current_pivot is neither 1 nor 2

        # Update the target column
        df.loc[mask, column_name] = next_pivot_high_price - df['F_close']

    except Exception as e:
        logging.error(f"Error in compute_Target for table {table_name}. Error: {e}")

    return df


def compute_sellTarget_h(df_subset):
    actualClass, F_high = df_subset['actualClass'], df_subset['F_high']
    sellTarget_high = [0]*len(df_subset)  # Initialize an empty list for sellTarget_high

    # Identify where actualClass is non-zero
    change_points = [i for i, x in enumerate(actualClass) if x != 0]

    last_1_idx = None
    for idx in change_points:
        # If the current pivot is 2 and there was a 1 before
        if actualClass[idx] == 2 and last_1_idx is not None:
            # Get the sell target value
            sell_target_value = F_high[idx]

            # Now, for each row between last_1_idx and idx, subtract its F_high from sell_target_value
            for i in range(last_1_idx, idx+1):
                sellTarget_high[i] = sell_target_value - F_high[i]

            last_1_idx = None  # reset the last 1 index
        elif actualClass[idx] == 1:
            last_1_idx = idx

    df_subset['sellTarget_high'] = sellTarget_high
    return df_subset


def compute_buyTarget_l(df_subset):
    actualClass, F_low = df_subset['actualClass'], df_subset['F_low']
    buyTarget_low = [0]*len(df_subset)  # Initialize an empty list for sellTarget_high

    # Identify where actualClass is non-zero
    change_points = [i for i, x in enumerate(actualClass) if x != 0]

    last_1_idx = None
    for idx in change_points:
        # If the current pivot is 2 and there was a 1 before
        if actualClass[idx] == 1 and last_1_idx is not None:
            # Get the sell target value
            buy_target_value = F_low[idx]

            # Now, for each row between last_1_idx and idx, subtract its F_high from sell_target_value
            for i in range(last_1_idx, idx+1):
                buyTarget_low[i] = buy_target_value - F_low[i]

            last_1_idx = None  # reset the last 1 index
        elif actualClass[idx] == 2:
            last_1_idx = idx

    df_subset['buyTarget_low'] = buyTarget_low
    return df_subset

def compute_sellTarget_l(df_subset):
    actualClass, F_low = df_subset['actualClass'], df_subset['F_low']
    sellTarget_low = [0]*len(df_subset)  # Initialize an empty list for sellTarget_high

    # Identify where actualClass is non-zero
    change_points = [i for i, x in enumerate(actualClass) if x != 0]

    last_1_idx = None
    for idx in change_points:
        # If the current pivot is 2 and there was a 1 before
        if actualClass[idx] == 2 and last_1_idx is not None:
            # Get the sell target value
            sell_target_value = F_low[idx]

            # Now, for each row between last_1_idx and idx, subtract its F_high from sell_target_value
            for i in range(last_1_idx, idx+1):
                sellTarget_low[i] = sell_target_value - F_low[i]

            last_1_idx = None  # reset the last 1 index
        elif actualClass[idx] == 1:
            last_1_idx = idx

    df_subset['sellTarget_low'] = sellTarget_low
    return df_subset


def compute_buyTarget_h(df_subset):
    actualClass, F_high = df_subset['actualClass'], df_subset['F_high']
    buyTarget_high = [0]*len(df_subset)  # Initialize an empty list for sellTarget_high

    # Identify where actualClass is non-zero
    change_points = [i for i, x in enumerate(actualClass) if x != 0]

    last_1_idx = None
    for idx in change_points:
        # If the current pivot is 2 and there was a 1 before
        if actualClass[idx] == 1 and last_1_idx is not None:
            # Get the sell target value
            buy_target_value = F_high[idx]

            # Now, for each row between last_1_idx and idx, subtract its F_high from sell_target_value
            for i in range(last_1_idx, idx+1):
                buyTarget_high[i] = buy_target_value - F_high[i]

            last_1_idx = None  # reset the last 1 index
        elif actualClass[idx] == 2:
            last_1_idx = idx

    df_subset['buyTarget_high'] = buyTarget_high
    return df_subset




def compute_smoothedTarget_h(df_subset):
    actualClass, F_high = df_subset['actualClass'], df_subset['sellTarget_high']
    sellTarget_smoothed_high = [0]*len(df_subset)  # Initialize an empty list for smoothed_high

    # Identify where actualClass is non-zero
    change_points = [i for i, x in enumerate(actualClass) if x != 0]

    last_1_idx = None
    for idx in change_points:
        # If the current pivot is 2 and there was a 1 before
        if actualClass[idx] == 2 and last_1_idx is not None:
            # Apply exponential smoothing over the range
            smoothed_values = exponential_smoothing(pd.Series(F_high).iloc[last_1_idx:idx+1])

            # Store these smoothed values
            for i, val in enumerate(smoothed_values):
                sellTarget_smoothed_high[last_1_idx+i] = val

            last_1_idx = None  # reset the last 1 index
        elif actualClass[idx] == 1:
            last_1_idx = idx

    df_subset['sellTarget_smoothed_high'] = sellTarget_smoothed_high
    return df_subset

def compute_smoothedTarget_l(df_subset):
    actualClass, F_low = df_subset['actualClass'], df_subset['buyTarget_low']
    buyTarget_smoothed_low = [0]*len(df_subset)  # Initialize an empty list for smoothed_high

    # Identify where actualClass is non-zero
    change_points = [i for i, x in enumerate(actualClass) if x != 0]

    last_1_idx = None
    for idx in change_points:
        # If the current pivot is 2 and there was a 1 before
        if actualClass[idx] == 1 and last_1_idx is not None:
            # Apply exponential smoothing over the range
            smoothed_values = exponential_smoothing(pd.Series(F_low).iloc[last_1_idx:idx+1])

            # Store these smoothed values
            for i, val in enumerate(smoothed_values):
                buyTarget_smoothed_low[last_1_idx+i] = val

            last_1_idx = None  # reset the last 1 index
        elif actualClass[idx] == 2:
            last_1_idx = idx

    df_subset['buyTarget_smoothed_low'] = buyTarget_smoothed_low
    return df_subset


def compute_smoothedSellTarget_l(df_subset):
    actualClass, F_low = df_subset['actualClass'], df_subset['sellTarget_low']
    sellTarget_smoothed_low = [0]*len(df_subset)  # Initialize an empty list for smoothed_high

    # Identify where actualClass is non-zero
    change_points = [i for i, x in enumerate(actualClass) if x != 0]

    last_1_idx = None
    for idx in change_points:
        # If the current pivot is 2 and there was a 1 before
        if actualClass[idx] == 2 and last_1_idx is not None:
            # Apply exponential smoothing over the range
            smoothed_values = exponential_smoothing(pd.Series(F_low).iloc[last_1_idx:idx+1])

            # Store these smoothed values
            for i, val in enumerate(smoothed_values):
                sellTarget_smoothed_low[last_1_idx+i] = val

            last_1_idx = None  # reset the last 1 index
        elif actualClass[idx] == 1:
            last_1_idx = idx

    df_subset['sellTarget_smoothed_low'] = sellTarget_smoothed_low
    return df_subset

def compute_smoothedBuyTarget_h(df_subset):
    actualClass, F_high = df_subset['actualClass'], df_subset['buyTarget_high']
    buyTarget_smoothed_high = [0]*len(df_subset)  # Initialize an empty list for smoothed_high

    # Identify where actualClass is non-zero
    change_points = [i for i, x in enumerate(actualClass) if x != 0]

    last_1_idx = None
    for idx in change_points:
        # If the current pivot is 2 and there was a 1 before
        if actualClass[idx] == 1 and last_1_idx is not None:
            # Apply exponential smoothing over the range
            smoothed_values = exponential_smoothing(pd.Series(F_high).iloc[last_1_idx:idx+1])

            # Store these smoothed values
            for i, val in enumerate(smoothed_values):
                buyTarget_smoothed_high[last_1_idx+i] = val

            last_1_idx = None  # reset the last 1 index
        elif actualClass[idx] == 2:
            last_1_idx = idx

    df_subset['buyTarget_smoothed_high'] = buyTarget_smoothed_high
    return df_subset



def compute_sellTarget_smoothed(df_subset):
    actualClass, F_close = df_subset['actualClass'], df_subset['sellTarget']
    sellTarget_smoothed = [0]*len(df_subset)  # Initialize an empty list for smoothed_high

    # Identify where actualClass is non-zero
    change_points = [i for i, x in enumerate(actualClass) if x != 0]

    last_1_idx = None
    for idx in change_points:
        # If the current pivot is 2 and there was a 1 before
        if actualClass[idx] == 2 and last_1_idx is not None:
            # Apply exponential smoothing over the range
            smoothed_values = exponential_smoothing(pd.Series(F_close).iloc[last_1_idx:idx+1])

            # Store these smoothed values
            for i, val in enumerate(smoothed_values):
                sellTarget_smoothed[last_1_idx+i] = val

            last_1_idx = None  # reset the last 1 index
        elif actualClass[idx] == 1:
            last_1_idx = idx

    df_subset['sellTarget_smoothed'] = sellTarget_smoothed
    return df_subset

def compute_buyTarget_smoothed(df_subset):
    actualClass, F_close = df_subset['actualClass'], df_subset['buyTarget']
    buyTarget_smoothed = [0]*len(df_subset)  # Initialize an empty list for smoothed_high

    # Identify where actualClass is non-zero
    change_points = [i for i, x in enumerate(actualClass) if x != 0]

    last_1_idx = None
    for idx in change_points:
        # If the current pivot is 2 and there was a 1 before
        if actualClass[idx] == 1 and last_1_idx is not None:
            # Apply exponential smoothing over the range
            smoothed_values = exponential_smoothing(pd.Series(F_close).iloc[last_1_idx:idx+1])

            # Store these smoothed values
            for i, val in enumerate(smoothed_values):
                buyTarget_smoothed[last_1_idx+i] = val

            last_1_idx = None  # reset the last 1 index
        elif actualClass[idx] == 2:
            last_1_idx = idx

    df_subset['buyTarget_smoothed'] = buyTarget_smoothed
    return df_subset

def compute_sellTarget_denoised_alpha_h(df_subset):
    actualClass = df_subset['actualClass'].values
    F_high = df_subset['sellTarget_high'].values
    sellTarget_denoised_alpha_high = np.zeros(len(df_subset), dtype=float)

    change_points = np.where(actualClass != 0)[0]
    last_1_idx = None

    for idx in change_points:
        if actualClass[idx] == 2 and last_1_idx is not None:
            segment = F_high[last_1_idx:idx+1]
            denoised_values = total_variation_denoising_alpha(segment)
            sellTarget_denoised_alpha_high[last_1_idx:idx+1] = denoised_values
            last_1_idx = None
        elif actualClass[idx] == 1:
            last_1_idx = idx

    df_subset['sellTarget_denoised_alpha_high'] = sellTarget_denoised_alpha_high
    return df_subset
  
def compute_buyTarget_denoised_alpha_l(df_subset):
    actualClass = df_subset['actualClass'].values
    F_low = df_subset['buyTarget_low'].values
    buyTarget_denoised_alpha_low = np.zeros(len(df_subset), dtype=float)

    change_points = np.where(actualClass != 0)[0]
    last_2_idx = None

    for idx in change_points:
        if actualClass[idx] == 1 and last_2_idx is not None:
            segment = F_low[last_2_idx:idx+1]
            denoised_values = total_variation_denoising_alpha(segment)
            buyTarget_denoised_alpha_low[last_2_idx:idx+1] = denoised_values
            last_2_idx = None
        elif actualClass[idx] == 2:
            last_2_idx = idx

    df_subset['buyTarget_denoised_alpha_low'] = buyTarget_denoised_alpha_low
    return df_subset

def compute_sellTarget_denoised_alpha_l(df_subset):
    actualClass = df_subset['actualClass'].values
    F_low = df_subset['sellTarget_low'].values
    sellTarget_denoised_alpha_low = np.zeros(len(df_subset), dtype=float)

    change_points = np.where(actualClass != 0)[0]
    last_1_idx = None

    for idx in change_points:
        if actualClass[idx] == 2 and last_1_idx is not None:
            segment = F_low[last_1_idx:idx+1]
            denoised_values = total_variation_denoising_alpha(segment)
            sellTarget_denoised_alpha_low[last_1_idx:idx+1] = denoised_values
            last_1_idx = None
        elif actualClass[idx] == 1:
            last_1_idx = idx

    df_subset['sellTarget_denoised_alpha_low'] = sellTarget_denoised_alpha_low
    return df_subset
  
def compute_buyTarget_denoised_alpha_h(df_subset):
    actualClass = df_subset['actualClass'].values
    F_high = df_subset['buyTarget_high'].values
    buyTarget_denoised_alpha_high = np.zeros(len(df_subset), dtype=float)

    change_points = np.where(actualClass != 0)[0]
    last_2_idx = None

    for idx in change_points:
        if actualClass[idx] == 1 and last_2_idx is not None:
            segment = F_high[last_2_idx:idx+1]
            denoised_values = total_variation_denoising_alpha(segment)
            buyTarget_denoised_alpha_high[last_2_idx:idx+1] = denoised_values
            last_2_idx = None
        elif actualClass[idx] == 2:
            last_2_idx = idx

    df_subset['buyTarget_denoised_alpha_high'] = buyTarget_denoised_alpha_high
    return df_subset




def compute_buyTarget_denoised_alpha(df_subset):
    actualClass = df_subset['actualClass'].values
    F_close = df_subset['buyTarget'].values
    buyTarget_denoised_alpha = np.zeros(len(df_subset), dtype=float)

    change_points = np.where(actualClass != 0)[0]
    last_2_idx = None

    for idx in change_points:
        if actualClass[idx] == 1 and last_2_idx is not None:
            segment = F_close[last_2_idx:idx+1]
            denoised_values = total_variation_denoising_alpha(segment)
            buyTarget_denoised_alpha[last_2_idx:idx+1] = denoised_values
            last_2_idx = None
        elif actualClass[idx] == 2:
            last_2_idx = idx

    df_subset['buyTarget_denoised_alpha'] = buyTarget_denoised_alpha
    return df_subset

def compute_sellTarget_denoised_alpha(df_subset):
    actualClass = df_subset['actualClass'].values
    F_close = df_subset['sellTarget'].values
    sellTarget_denoised_alpha = np.zeros(len(df_subset), dtype=float)

    change_points = np.where(actualClass != 0)[0]
    last_1_idx = None

    for idx in change_points:
        if actualClass[idx] == 2 and last_1_idx is not None:
            segment = F_close[last_1_idx:idx+1]
            denoised_values = total_variation_denoising_alpha(segment)
            sellTarget_denoised_alpha[last_1_idx:idx+1] = denoised_values
            last_1_idx = None
        elif actualClass[idx] == 1:
            last_1_idx = idx

    df_subset['sellTarget_denoised_alpha'] = sellTarget_denoised_alpha
    return df_subset


def compute_sellTarget_denoised_beta_h(df_subset):
    actualClass = df_subset['actualClass'].values
    F_high = df_subset['sellTarget_high'].values
    sellTarget_denoised_beta_high = np.zeros(len(df_subset), dtype=float)

    change_points = np.where(actualClass != 0)[0]
    last_1_idx = None

    for idx in change_points:
        if actualClass[idx] == 2 and last_1_idx is not None:
            segment = F_high[last_1_idx:idx+1]
            denoised_values = total_variation_denoising_beta(segment)
            sellTarget_denoised_beta_high[last_1_idx:idx+1] = denoised_values
            last_1_idx = None
        elif actualClass[idx] == 1:
            last_1_idx = idx

    df_subset['sellTarget_denoised_beta_high'] = sellTarget_denoised_beta_high
    return df_subset

def compute_buyTarget_denoised_beta_l(df_subset):
    actualClass = df_subset['actualClass'].values
    F_low = df_subset['buyTarget_low'].values
    buyTarget_denoised_beta_low = np.zeros(len(df_subset), dtype=float)

    change_points = np.where(actualClass != 0)[0]
    last_2_idx = None

    for idx in change_points:
        if actualClass[idx] == 1 and last_2_idx is not None:
            segment = F_low[last_2_idx:idx+1]
            denoised_values = total_variation_denoising_beta(segment)
            buyTarget_denoised_beta_low[last_2_idx:idx+1] = denoised_values
            last_2_idx = None
        elif actualClass[idx] == 2:
            last_2_idx = idx

    df_subset['buyTarget_denoised_beta_low'] = buyTarget_denoised_beta_low
    return df_subset

def compute_sellTarget_denoised_beta_l(df_subset):
    actualClass = df_subset['actualClass'].values
    F_low = df_subset['sellTarget_low'].values
    sellTarget_denoised_beta_low = np.zeros(len(df_subset), dtype=float)

    change_points = np.where(actualClass != 0)[0]
    last_1_idx = None

    for idx in change_points:
        if actualClass[idx] == 2 and last_1_idx is not None:
            segment = F_low[last_1_idx:idx+1]
            denoised_values = total_variation_denoising_beta(segment)
            sellTarget_denoised_beta_low[last_1_idx:idx+1] = denoised_values
            last_1_idx = None
        elif actualClass[idx] == 1:
            last_1_idx = idx

    df_subset['sellTarget_denoised_beta_low'] = sellTarget_denoised_beta_low
    return df_subset

def compute_buyTarget_denoised_beta_h(df_subset):
    actualClass = df_subset['actualClass'].values
    F_high = df_subset['buyTarget_high'].values
    buyTarget_denoised_beta_high = np.zeros(len(df_subset), dtype=float)

    change_points = np.where(actualClass != 0)[0]
    last_2_idx = None

    for idx in change_points:
        if actualClass[idx] == 1 and last_2_idx is not None:
            segment = F_high[last_2_idx:idx+1]
            denoised_values = total_variation_denoising_beta(segment)
            buyTarget_denoised_beta_high[last_2_idx:idx+1] = denoised_values
            last_2_idx = None
        elif actualClass[idx] == 2:
            last_2_idx = idx

    df_subset['buyTarget_denoised_beta_high'] = buyTarget_denoised_beta_high
    return df_subset



def compute_sellTarget_denoised_beta(df_subset):
    actualClass = df_subset['actualClass'].values
    F_close = df_subset['sellTarget'].values
    sellTarget_denoised_beta = np.zeros(len(df_subset), dtype=float)

    change_points = np.where(actualClass != 0)[0]
    last_1_idx = None

    for idx in change_points:
        if actualClass[idx] == 2 and last_1_idx is not None:
            segment = F_close[last_1_idx:idx+1]
            denoised_values = total_variation_denoising_beta(segment)
            sellTarget_denoised_beta[last_1_idx:idx+1] = denoised_values
            last_1_idx = None
        elif actualClass[idx] == 1:
            last_1_idx = idx

    df_subset['sellTarget_denoised_beta'] = sellTarget_denoised_beta
    return df_subset

def compute_buyTarget_denoised_beta(df_subset):
    actualClass, F_close = df_subset['actualClass'], df_subset['buyTarget']
    buyTarget_denoised_beta = [0]*len(df_subset)  # Initialize an empty list for smoothed_high

    # Identify where actualClass is non-zero
    change_points = [i for i, x in enumerate(actualClass) if x != 0]

    last_1_idx = None
    for idx in change_points:
        # If the current pivot is 2 and there was a 1 before
        if actualClass[idx] == 1 and last_1_idx is not None:
            # Apply exponential smoothing over the range
            denoising_values = total_variation_denoising_beta(pd.Series(F_close).iloc[last_1_idx:idx+1])

            # Store these smoothed values
            for i, val in enumerate(denoising_values):
                buyTarget_denoised_beta[last_1_idx+i] = val
            last_1_idx = None  # reset the last 1 index
        elif actualClass[idx] == 2:
            last_1_idx = idx

    df_subset['buyTarget_denoised_beta'] = buyTarget_denoised_beta
    return df_subset



def normalise_rows(df):  # Replace with the actual function signature
    columns_to_exclude = [
        "ID", "time", "actualClass",
        "F_open", "F_high", "F_low", "F_close", "F_HA_open", "F_HA_high", "F_HA_low", "F_HA_close",
        "F_open_nrm", "F_high_nrm", "F_low_nrm", "F_close_nrm", "F_volume", "F_smoothedOpen", "F_smoothedHigh",
        "F_smoothedLow", "F_smoothedClose", "F_denoise_alphaOpen", "F_denoise_alphaHigh", "F_denoise_alphaLow",
        "F_denoise_alphaClose", "F_denoise_betaOpen", "F_denoise_betaHigh", "F_denoise_betaLow", "F_denoise_betaClose"
    ]

    # Identify the columns to be set to 0
    columns_to_modify = [col for col in df.columns if col not in columns_to_exclude]

    pivot_indices = df.index[df['pivot'].notna()].tolist()
    for i in range(len(pivot_indices) - 1):
        if pivot_indices[i + 1] - pivot_indices[i] < 10:
            df.loc[pivot_indices[i]:pivot_indices[i + 1], columns_to_modify] = 0

    return df





def process_pair(df, current_idx, next_pivot_idx, table_name):

    next_pivot_high_price = df.loc[next_pivot_idx, 'F_close']
    idx_range = df.index[df.index.isin(range(current_idx, next_pivot_idx + 1))]
   
    current_pivot = df.loc[current_idx, 'pivot']

    # Check if current_pivot is a string or an integer
    if isinstance(current_pivot, str):
        pivot_1 = '1'
        pivot_2 = '2'
    else:
        pivot_1 = 1
        pivot_2 = 2

    # Set the actualClass based on the current_pivot
    idx_range = df.index[df.index.isin(range(current_idx, next_pivot_idx + 1))]
    pd.set_option('display.float_format', '{:.2f}'.format)
    if current_pivot == pivot_1:
        df.loc[idx_range, 'sellTarget'] = (next_pivot_high_price - df.loc[idx_range, 'F_close'])
        df.at[current_idx, 'actualClass'] = 1   
    elif current_pivot == pivot_2:
        df.loc[idx_range, 'buyTarget'] = (next_pivot_high_price - df.loc[idx_range, 'F_close'])
        df.at[current_idx, 'actualClass'] = 2   
        print(f"Calculating pivot at index {current_idx}")

    return df



def process_targets(df):
        # Filter df based on actualClass and other columns you're interested in
    df_subset = df[['actualClass', 'F_high']].copy()
    df_subset = compute_sellTarget_h(df_subset)
    df['sellTarget_high'] = df_subset['sellTarget_high']
    print(f"Calculating targets")

    df_subset = df[['actualClass', 'F_low']].copy()
    df_subset = compute_buyTarget_l(df_subset)
    df['buyTarget_low'] = df_subset['buyTarget_low']

    df_subset = df[['actualClass', 'F_low']].copy()
    df_subset = compute_sellTarget_l(df_subset)
    df['sellTarget_low'] = df_subset['sellTarget_low']


    df_subset = df[['actualClass', 'F_high']].copy()
    df_subset = compute_buyTarget_h(df_subset)
    df['buyTarget_high'] = df_subset['buyTarget_high']

    
    df_subset = df[['actualClass', 'sellTarget_high']].copy()
    df_subset = compute_smoothedTarget_h(df_subset)
    df['sellTarget_smoothed_high'] = df_subset['sellTarget_smoothed_high']

    df_subset = df[['actualClass', 'buyTarget_low']].copy()
    df_subset = compute_smoothedTarget_l(df_subset)
    df['buyTarget_smoothed_low'] = df_subset['buyTarget_smoothed_low']

    df_subset = df[['actualClass', 'sellTarget_low']].copy()
    df_subset = compute_smoothedSellTarget_l(df_subset)
    df['sellTarget_smoothed_low'] = df_subset['sellTarget_smoothed_low']

    df_subset = df[['actualClass', 'buyTarget_high']].copy()
    df_subset = compute_smoothedBuyTarget_h(df_subset)
    df['buyTarget_smoothed_high'] = df_subset['buyTarget_smoothed_high']



    df_subset = df[['actualClass', 'sellTarget']].copy()
    df_subset = compute_sellTarget_smoothed(df_subset)
    df['sellTarget_smoothed'] = df_subset['sellTarget_smoothed']

    df_subset = df[['actualClass', 'buyTarget']].copy()
    df_subset = compute_buyTarget_smoothed(df_subset)
    df['buyTarget_smoothed'] = df_subset['buyTarget_smoothed']

    df_subset = df[['actualClass', 'sellTarget_high']].copy()
    df_subset = compute_sellTarget_denoised_alpha_h(df_subset)
    df['sellTarget_denoised_alpha_high'] = df_subset['sellTarget_denoised_alpha_high']

    df_subset = df[['actualClass', 'buyTarget_low']].copy()
    df_subset = compute_buyTarget_denoised_alpha_l(df_subset)
    df['buyTarget_denoised_alpha_low'] = df_subset['buyTarget_denoised_alpha_low']

    df_subset = df[['actualClass', 'buyTarget_high']].copy()
    df_subset = compute_buyTarget_denoised_alpha_h(df_subset)
    df['buyTarget_denoised_alpha_high'] = df_subset['buyTarget_denoised_alpha_high']

    df_subset = df[['actualClass', 'sellTarget_low']].copy()
    df_subset = compute_sellTarget_denoised_alpha_l(df_subset)
    df['sellTarget_denoised_alpha_low'] = df_subset['sellTarget_denoised_alpha_low']

    df_subset = df[['actualClass', 'sellTarget']].copy()
    df_subset = compute_sellTarget_denoised_alpha(df_subset)
    df['sellTarget_denoised_alpha'] = df_subset['sellTarget_denoised_alpha']

    df_subset = df[['actualClass', 'buyTarget']].copy()
    df_subset = compute_buyTarget_denoised_alpha(df_subset)
    df['buyTarget_denoised_alpha'] = df_subset['buyTarget_denoised_alpha']

    df_subset = df[['actualClass', 'sellTarget_high']].copy()
    df_subset = compute_sellTarget_denoised_beta_h(df_subset)
    df['sellTarget_denoised_beta_high'] = df_subset['sellTarget_denoised_beta_high']

    df_subset = df[['actualClass', 'buyTarget_low']].copy()
    df_subset = compute_buyTarget_denoised_beta_l(df_subset)
    df['buyTarget_denoised_beta_low'] = df_subset['buyTarget_denoised_beta_low']

    df_subset = df[['actualClass', 'buyTarget_high']].copy()
    df_subset = compute_buyTarget_denoised_beta_h(df_subset)
    df['buyTarget_denoised_beta_high'] = df_subset['buyTarget_denoised_beta_high']

    df_subset = df[['actualClass', 'sellTarget_low']].copy()
    df_subset = compute_sellTarget_denoised_beta_l(df_subset)
    df['sellTarget_denoised_beta_low'] = df_subset['sellTarget_denoised_beta_low']

    df_subset = df[['actualClass', 'sellTarget']].copy()
    df_subset = compute_sellTarget_denoised_beta(df_subset)
    df['sellTarget_denoised_beta'] = df_subset['sellTarget_denoised_beta']

    df_subset = df[['actualClass', 'buyTarget']].copy()
    df_subset = compute_buyTarget_denoised_beta(df_subset)
    df['buyTarget_denoised_beta'] = df_subset['buyTarget_denoised_beta']
    return df


def denoise_alphaOpen(df):
    df['F_denoise_alphaOpen'] = df['F_open'].tolist()
    return df

def denoise_alphaHigh(df):
    df['F_denoise_alphaHigh'] = df['F_high'].tolist()
    return df

def denoise_alphaLow(df):
    df['F_denoise_alphaLow'] = df['F_low'].tolist()
    return df

def denoise_alphaClose(df):
    df['F_denoise_alphaClose'] = df['F_close'].tolist()
    return df

def denoise_betaOpen(df):
    df['F_denoise_betaOpen'] = df['F_open'].tolist()
    return df

def denoise_betaHigh(df):
    df['F_denoise_betaHigh'] = df['F_high'].tolist()
    return df

def denoise_betaLow(df):
    df['F_denoise_betaLow'] = df['F_low'].tolist()
    return df

def denoise_betaClose(df):
    df['F_denoise_betaClose'] = df['F_close'].tolist()
    return df

def transform_F_HA_open(df):
    return compute_heikin_ashi(df, 'F_HA_open')

def transform_F_HA_high(df):
    return compute_heikin_ashi(df, 'F_HA_high')

def transform_F_HA_low(df):
    return compute_heikin_ashi(df, 'F_HA_low')

def transform_F_HA_close(df):
    return compute_heikin_ashi(df, 'F_HA_close')

def parallel_transform(df):
    # Process exponential smoothing transformations sequentially
    
    total_variation_denoising_charlie = [
        denoise_alphaOpen,
        denoise_alphaHigh,
        denoise_alphaLow,
        denoise_alphaClose
    ]
    total_variation_denoising_delta = [
        denoise_betaOpen,
        denoise_betaHigh,
        denoise_betaLow,
        denoise_betaClose
    ]

    transformations_F_HA = [
        transform_F_HA_open,
        transform_F_HA_high,
        transform_F_HA_low,
        transform_F_HA_close
    ]
    
    transformation_groups = [total_variation_denoising_charlie, total_variation_denoising_delta, transformations_F_HA]

    for group in transformation_groups:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for func in group:
                df = executor.submit(func, df).result()

    return df







#there's likely around 3 pivots that are incorrect before entering in here. fuck them
def adjust_buy_targets(df):
    buy_rows = df[(df['actualClass'] == 2) & (df['buyTarget'] > 0)].index.tolist()

    for idx in buy_rows:
        current_close = df.loc[idx, 'F_close']
        if correct_idx := next(
            (
                i
                for i in range(idx + 1, len(df))
                if df.loc[i, 'F_close'] < current_close
            ),
            None,
        ):
            # Adjust the actualClass for the incorrect and correct indices
            df.at[correct_idx, 'actualClass'] = 2
            df.at[idx, 'actualClass'] = 0

            # Calculate the correct buyTarget and buyTargetP values
            next_pivot_high_price = df.loc[correct_idx, 'F_close']
            df.loc[idx:correct_idx, 'buyTarget'] = next_pivot_high_price - df.loc[idx:correct_idx, 'F_close']
            

    return df




async def process_and_target(df, table_name, db_queue):
    threshold = 2.0
    df['pivot'] = None

    # Initial pivot setup
    last_pivot_price = df['F_close'].iloc[0]
    last_pivot_type = None
    last_pivot_index = None

    # Potential pivot setup
    potential_pivot_index = None
    potential_pivot_price = last_pivot_price
    potential_pivot_type = None
    sequential_pivot_count = 0

    # Sequence tracking
    sequential_pivots = []

    for i in range(1, len(df)):
        close_price = df['F_close'].iloc[i]

        # If there's no potential pivot being observed
        if not potential_pivot_type:
            if close_price >= last_pivot_price * (1 + threshold/100):
                potential_pivot_type = '2'
                potential_pivot_price = close_price
                potential_pivot_index = i
            elif close_price <= last_pivot_price * (1 - threshold/100):
                potential_pivot_type = '1'
                potential_pivot_price = close_price
                potential_pivot_index = i

        # If observing a potential bullish pivot
        elif potential_pivot_type == '2':
            # Update the potential pivot price and index if a new high is found
            if close_price > potential_pivot_price:
                potential_pivot_price = close_price
                potential_pivot_index = i
            # Confirm the pivot if a drop of 2% or more from the potential pivot price is observed
            elif close_price <= potential_pivot_price * (1 - threshold/100):
                percent_change = ((potential_pivot_price - last_pivot_price) / last_pivot_price) * 100
            
                # Warning check
                if last_pivot_type == '2':
                    sequential_pivot_count += 1
            
                df.at[potential_pivot_index, 'pivot'] = '2'
                last_pivot_price = potential_pivot_price
                last_pivot_type = '2'
                potential_pivot_index = None
                potential_pivot_type = None

        # If observing a potential bearish pivot
        elif potential_pivot_type == '1':
            # Update the potential pivot price and index if a new low is found
            if close_price < potential_pivot_price:
                potential_pivot_price = close_price
                potential_pivot_index = i
            # Confirm the pivot if an increase of 2% or more from the potential pivot price is observed
            elif close_price >= potential_pivot_price * (1 + threshold/100):
                percent_change = ((potential_pivot_price - last_pivot_price) / last_pivot_price) * 100
            
                # Warning check
                if last_pivot_type == '1':                   
                    sequential_pivot_count += 1
            
                df.at[potential_pivot_index, 'pivot'] = '1'
                last_pivot_price = potential_pivot_price
                last_pivot_type = '1'
                potential_pivot_index = None
                potential_pivot_type = None
    print(f"Total number of sequential pivots: {sequential_pivot_count}")
    df[['buyTarget', 'sellTarget', 'actualClass']] = [0, 0, 0]
    pivot_indices = df.index[df['pivot'].notna()].tolist()
    corrections_made = 0

    i = 0
    while i < len(pivot_indices) - 1:
        current_idx = pivot_indices[i]
        next_idx = pivot_indices[i + 1]

        current_pivot = df['pivot'].iloc[current_idx]
        next_pivot = df['pivot'].iloc[next_idx]

        # If two consecutive pivot points are of the same type
        if current_pivot == next_pivot:
            if current_pivot == '2':
                # Keep the higher of the two and delete the other
                if df['F_close'].iloc[current_idx] > df['F_close'].iloc[next_idx]:
                    df.at[next_idx, 'pivot'] = None
                    pivot_indices.pop(i + 1)
                else:
                    df.at[current_idx, 'pivot'] = None
                    pivot_indices.pop(i)
            elif current_pivot == '1':
                # Keep the lower of the two and delete the other
                if df['F_close'].iloc[current_idx] < df['F_close'].iloc[next_idx]:
                    df.at[next_idx, 'pivot'] = None
                    pivot_indices.pop(i + 1)
                else:
                    df.at[current_idx, 'pivot'] = None
                    pivot_indices.pop(i)
            corrections_made += 1
        else:
            i += 1

    print(f"Total number of corrections made: {corrections_made}")
    
    last_pivot_type = None
    last_pivot_index = None

    # Loop through the dataframe
    for i in range(len(df)):
        current_pivot = df['pivot'].iloc[i]
    
        # If the current pivot type is the same as the last pivot type found, log it
        if last_pivot_type and current_pivot == last_pivot_type:
            print(f"Sequential pivot still found at Row Index: {i}, Time: {df['time'].iloc[i]}, F_close: {df['F_close'].iloc[i]}, Pivot: {current_pivot}")
        elif current_pivot:  # If current pivot is not None but different from last pivot
            last_pivot_type = current_pivot
            last_pivot_index = i

    print(f"Mathers iteration, pivot points completed")

    # Pairing the pivots
    pivot_indices = df.index[df['pivot'].notna()].tolist()
    pairs = []

    pivot_indices = df.index[df['pivot'].notna()].tolist()

    # Check if the first pivot is of type 2, and if so, remove it.
    if df['pivot'].iloc[pivot_indices[0]] == 2:
        pivot_indices.pop(0)

    pairs = []
    for i in range(len(pivot_indices) - 1):
        if df['pivot'].iloc[pivot_indices[i]] != df['pivot'].iloc[pivot_indices[i + 1]]:
            pairs.append((pivot_indices[i], pivot_indices[i + 1]))
            

    # Process each pair
    for start_idx, end_idx in pairs:
        try:
            df = process_pair(df, start_idx, end_idx, table_name)
        except Exception as e:
            print(f"Error processing pair {start_idx}, {end_idx}: {e}")

    df =  process_targets(df)
    df =  normalise_rows(df)

    try:
        df.drop(columns=['pivot'], inplace=True)
        write_to_db(df, table_name)
    except Exception as e:
        logging.exception(f"mathers_iteration, Error occurred while processing: {e}")

        
    return table_name



async def main_mathers_iteration_async_logic(db_queue):
    setup_logging()
    try:
        print("mathers_iteration, Starting main function, init:mathers_iteration.py")

        tasks = []
        for timeframe, table_name in timeframes.items():
            print(f"mathers_iteration, Processing timeframe: {timeframe}, table_name: {table_name}")
            task = main_processing_loop(table_name, rsiOverbought, rsiOversold)
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        df_dict = {}
        successful_processing = True

        for table, df_selected in results:
            if table is None and df_selected is None:
                print("mathers_iteration, Skipping processing for table due to not enough data.")
                successful_processing = False  # Set the flag to False if there's an error or None value
                continue
            df_dict[table] = df_selected

        logging.debug("mathers_iteration, Gathered data. Now processing and targeting.")
        tasks = [process_and_target(df_dict.get(table_name), table_name, db_queue) for table_name in timeframes.values() if df_dict.get(table_name) is not None]

        dfs = await asyncio.gather(*tasks)

    except Exception as e:
        error_message = f"mathers_iteration, Exception occurred: {e}\n"
        error_message += "Traceback (most recent call last):\n"
        error_message += traceback.format_exc()
        print(error_message)

def main(db_queue):
    setup_logging()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main_mathers_iteration_async_logic(db_queue))
    print("Sentinel reached. Ending mathers_iteration.")

if __name__ == "__main__":
    db_queue = Queue()   
    main(db_queue)
    os._exit(0)

