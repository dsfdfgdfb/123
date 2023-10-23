import aiosqlite
import pandas as pd
import pandas_ta as ta
import asyncio
import logging
import numpy as np
import json
import os
import sqlite3
from lobotomiser.dbqueue import mr_writer, get_db_path
from multiprocessing import Queue, Process, Pool, cpu_count
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy.optimize as opt
import numpy as np

import concurrent.futures
import traceback
import multiprocessing


# Define the timeframes and corresponding database paths
timeframes = {
    '5m': 'fiveminutebtc',
    '15m': 'fifteenminutebtc',
    '1h': 'onehourbtc',
    '4h': 'fourhourbtc',
    '12h': 'twelvehourbtc',
    '1d': 'onedaybtc'
}

DB_PATH_DATA = './lobotomiser/data/btcdata.db'
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
    db_path = get_db_path(table_name)  # Fetch the correct database path using the function
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

def write_to_db(df, table_name, db_queue):
    if df.empty:
        print(f"mathers_iteration, Skipping write for {table_name} as the DataFrame is empty.")
        return

    # Round numeric columns to 2 decimal places
    numeric_cols = df.select_dtypes(include=['float64']).columns
    df[numeric_cols] = df[numeric_cols].round(2)
    
    # Structure the task to send to the mr_writer
    task = {
        "operation": "write_or_replace",
        "data": df.to_dict(orient="records"),  # Convert DataFrame to a list of dictionaries
        "table_name": table_name,
        "columns": df.columns.tolist()
    }
    
    # Put the task into the shared queue
    db_queue.put(task)
    db_queue.put(None)
    print("Mathers_iteration:Sentinel is here Neo")

    
       
    

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
    result = [series[0]]
    result.extend(
        alpha * series[n] + (1 - alpha) * result[n - 1]
        for n in range(1, len(series))
    )
    return result


def linear_interpolate(start, end, num_points):
    """Linearly interpolate between two values."""
    return [start + i * (end - start) / (num_points - 1) for i in range(num_points)]

def total_variation_denoising_alpha(y, alpha=10):
    """
    Apply Total Variation Denoising.
    
    Parameters:
    - y: Input data
    - alpha: Regularization parameter. Higher values produce smoother results.
    
    Returns:
    - Denoised data
    """
    N = len(y)
    D = np.eye(N) - np.eye(N, k=-1)
    D[0, 0] = 0
    D[N-1, N-2] = 0
    D[N-1, N-1] = 1
    
    def obj_func(x):
        return 0.5 * np.sum((x - y) ** 2) + alpha * np.sum(np.abs(D @ x))
    
    result = opt.minimize(obj_func, y, method='L-BFGS-B')
    
    return result.x

def total_variation_denoising_beta(y, alpha=15):
    """
    Apply Total Variation Denoising.
    
    Parameters:
    - y: Input data
    - alpha: Regularization parameter. Higher values produce smoother results.
    
    Returns:
    - Denoised data
    """
    N = len(y)
    D = np.eye(N) - np.eye(N, k=-1)
    D[0, 0] = 0
    D[N-1, N-2] = 0
    D[N-1, N-1] = 1
    
    def obj_func(x):
        return 0.5 * np.sum((x - y) ** 2) + alpha * np.sum(np.abs(D @ x))
    
    result = opt.minimize(obj_func, y, method='L-BFGS-B')
    
    return result.x

def process_pair(df, current_idx, next_pivot_idx):
    current_pivot = df.loc[current_idx, 'pivot']
    next_pivot_high_price = df.loc[next_pivot_idx, 'F_close']

    num_points = next_pivot_idx - current_idx + 1

    idx_range = df.index[df.index.isin(range(current_idx, next_pivot_idx + 1))]
    pd.set_option('display.float_format', '{:.2f}'.format)
    if current_pivot == 1:
        df.loc[idx_range, 'sellTarget'] = (next_pivot_high_price - df.loc[idx_range, 'F_close'])
        df.loc[idx_range, 'sellTarget_high'] = (next_pivot_high_price - df.loc[idx_range, 'F_high'])
        df.at[current_idx, 'actualClass'] = 1    
        df.loc[idx_range, 'sellTarget_smoothed'] = exponential_smoothing(df.loc[idx_range, 'sellTarget'].tolist())       
        sell_target_data = df.loc[idx_range, 'sellTarget'].to_numpy()
        df.loc[idx_range, 'sellTarget_denoised_alpha'] = total_variation_denoising_alpha(sell_target_data)  
        df.loc[idx_range, 'sellTarget_denoised_beta'] = total_variation_denoising_beta(sell_target_data)
        df.loc[idx_range, 'sellTarget_smoothed_high'] = exponential_smoothing(df.loc[idx_range, 'sellTarget_high'].tolist())       
        sell_target_data = df.loc[idx_range, 'sellTarget_high'].to_numpy()
        df.loc[idx_range, 'sellTarget_denoised_alpha_high'] = total_variation_denoising_alpha(sell_target_data)  
        df.loc[idx_range, 'sellTarget_denoised_beta_high'] = total_variation_denoising_beta(sell_target_data) 
        
           
    elif current_pivot == 2:
        df.loc[idx_range, 'buyTarget'] = (next_pivot_high_price - df.loc[idx_range, 'F_close'])
        df.loc[idx_range, 'buyTarget_low'] = (next_pivot_high_price - df.loc[idx_range, 'F_low'])
        df.at[current_idx, 'actualClass'] = 2        
        df.loc[idx_range, 'buyTarget_smoothed'] = exponential_smoothing(df.loc[idx_range, 'buyTarget'].tolist())
        buy_target_data = df.loc[idx_range, 'buyTarget'].to_numpy()
        df.loc[idx_range, 'buyTarget_denoised_alpha'] = total_variation_denoising_alpha(buy_target_data)
        df.loc[idx_range, 'buyTarget_denoised_beta'] = total_variation_denoising_beta(buy_target_data) 
        df.loc[idx_range, 'buyTarget_smoothed_low'] = exponential_smoothing(df.loc[idx_range, 'buyTarget_low'].tolist())
        buy_target_data = df.loc[idx_range, 'buyTarget_low'].to_numpy()
        df.loc[idx_range, 'buyTarget_denoised_alpha_low'] = total_variation_denoising_alpha(buy_target_data)
        df.loc[idx_range, 'buyTarget_denoised_beta_low'] = total_variation_denoising_beta(buy_target_data)
       
      

    return df.loc[idx_range]




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
    
    total_variation_denoising_alpha = [
        denoise_alphaOpen,
        denoise_alphaHigh,
        denoise_alphaLow,
        denoise_alphaClose
    ]
    total_variation_denoising_beta = [
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
    
    transformation_groups = [total_variation_denoising_alpha, total_variation_denoising_beta, transformations_F_HA]

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
            df.loc[idx:correct_idx, 'buyTargetP'] = (((next_pivot_high_price - df.loc[idx:correct_idx, 'F_close']) / df.loc[idx:correct_idx, 'F_close']) * 100)

    return df




def process_and_target(df, table_name, db_queue):
    print("mathers_iteration, processing")
    try:
        idx = 0
        state_indices = []
        current_accumulation = []
        current_state = None
        last_state_change_timestamp = None

        df['rsiState'] = 0

        while idx < len(df):
            row = df.iloc[idx]

            if current_state is None:
                if row['isOverbought'] == 1:
                    current_state = 1
                elif row['isOversold'] == 1:
                    current_state = 2
                idx += 1
                continue

            if (
                current_state == 1
                and row['isOversold'] == 1
                or current_state != 1
                and current_state == 2
                and row['isOverbought'] == 1
            ):
                current_accumulation.append(idx)
            elif current_state == 1 and row['isOverbought'] == 1:
                if current_accumulation:
                    extremum_idx = df.loc[current_accumulation, 'F_close'].idxmin()
                    state_indices.append(extremum_idx)
                    df.at[extremum_idx, 'rsiState'] = 1
                current_accumulation = []
                current_state = 2
            elif (
                current_state != 1
                and (current_state != 2 or row['isOversold'] == 1)
                and current_state == 2
            ):
                if current_accumulation:
                    extremum_idx = df.loc[current_accumulation, 'F_close'].idxmax()
                    state_indices.append(extremum_idx)
                    df.at[extremum_idx, 'rsiState'] = 2
                current_accumulation = []
                current_state = 1

            idx += 1

        if current_accumulation:
            if current_state == 1:
                extremum_idx = df.loc[current_accumulation, 'F_close'].idxmax()
            elif current_state == 2:
                extremum_idx = df.loc[current_accumulation, 'F_close'].idxmin()
            state_indices.append(extremum_idx)



        df.drop(columns=['isOverbought', 'isOversold'], inplace=True)
        df['pivot'] = 0

    except Exception as e:
        logging.error(f"Error occurred: {e}")

    df[['buyTarget', 'sellTarget', 'actualClass']] = [0, 0, 0]
    df_extremum = df.copy()

    # The following code will now calculate the extremums in df_extremum
    close_series = df_extremum['F_close']
    state_indices = df_extremum.index[df_extremum['rsiState'] != 0].tolist()

    # Iterating over state indices to adjust rsiState values
    for i in range(len(state_indices) - 1):
        current_idx = state_indices[i]
        next_idx = state_indices[i + 1]
        current_state = df.at[current_idx, 'rsiState']
        next_state = df.at[next_idx, 'rsiState']

        # Two consecutive oversold states: mark the highest high as overbought
        if current_state == 1 and next_state == 1:
            max_high_idx = df.loc[current_idx:next_idx, 'F_close'].idxmax()
            df.at[max_high_idx, 'rsiState'] = 2

        # Two consecutive overbought states: mark the lowest low as oversold
        elif current_state == 2 and next_state == 2:
            min_low_idx = df.loc[current_idx:next_idx, 'F_close'].idxmin()
            df.at[min_low_idx, 'rsiState'] = 1

    # Re-fetch state indices after adjustments
    state_indices = df.index[df['rsiState'] != 0].tolist()

    # Iterating over state indices again to refine rsiState values
    for i in range(len(state_indices) - 1):
        current_idx = state_indices[i]
        next_idx = state_indices[i + 1]
        current_state = df.at[current_idx, 'rsiState']
        next_state = df.at[next_idx, 'rsiState']

        # Oversold followed by overbought: mark the last oversold as extreme
        if current_state == 1 and next_state == 2:
            df.at[current_idx, 'rsiState'] = 1  

        # Overbought followed by oversold: mark the last overbought as extreme
        elif current_state == 2 and next_state == 1:
            df.at[current_idx, 'rsiState'] = 2

    # 2. Re-fetch state indices after all modifications are done
    state_indices = df.index[df['rsiState'] != 0].tolist()

    while True:
        made_adjustments = False

        for i in range(len(state_indices) - 1):
            current_idx = state_indices[i]
            next_idx = state_indices[i + 1]
            current_state = df.at[current_idx, 'rsiState']
            next_state = df.at[next_idx, 'rsiState']

            # Two consecutive oversold states: mark the highest high as overbought
            if current_state == 1 and next_state == 1:
                max_high_idx = df.loc[current_idx:next_idx, 'F_close'].idxmax()
                df.at[max_high_idx, 'rsiState'] = 2
                made_adjustments = True

            # Two consecutive overbought states: mark the lowest low as oversold
            elif current_state == 2 and next_state == 2:
                min_low_idx = df.loc[current_idx:next_idx, 'F_close'].idxmin()
                df.at[min_low_idx, 'rsiState'] = 1
                made_adjustments = True

        # If no adjustments were made, break out of the loop
        if not made_adjustments:
            break

        # Otherwise, refresh the state_indices and iterate again
        state_indices = df.index[df['rsiState'] != 0].tolist()

    # 3. Determine the pivots
    slices = [slice(start_idx, end_idx) for start_idx, end_idx in zip(state_indices[:-1], state_indices[1:])]
    for s in slices:
        if df.at[s.start, 'rsiState'] == 2:
            min_low_idx = close_series[s].idxmin()
            df.at[min_low_idx, 'pivot'] = 1
        else:
            max_high_idx = close_series[s].idxmax()
            df.at[max_high_idx, 'pivot'] = 2

    pivot_indices = df.index[df['pivot'] != 0].tolist()

    pairs = [(pivot_indices[i], pivot_indices[i + 1]) for i in range(len(pivot_indices) - 1)]

    for start_idx, end_idx in pairs:
        try:
            result_df = process_pair(df, start_idx, end_idx)
            df.update(result_df)           
        except Exception as e:
            logging.exception(f"Error processing pair {start_idx}, {end_idx}: {e}")

    #df = adjust_buy_targets(df)
    try:    
        df.drop(columns=['pivot'], inplace=True)
        df.drop(columns=['rsiState'], inplace=True)
        write_to_db(df, table_name, db_queue)
    except Exception as e:
        logging.exception(f"mathers_iteration, Error occurred while processing: {e}")  

    return table_name



async def main_mathers_iteration_async_logic(db_queue):
    setup_logging()
    
    df_dict = {}  # Initialize the dictionary outside the try block
    
    try:
        print("mathers_iteration, Starting main function, init:mathers_iteration.py")

        tasks = [main_processing_loop(table_name, rsiOverbought, rsiOversold) for table_name in timeframes.values()]
        results = await asyncio.gather(*tasks)

        for table, df_selected in results:
            if table and not df_selected.empty:
                df_dict[table] = df_selected
            else:
                print("mathers_iteration, Skipping processing for table due to not enough data.")


    except Exception as e:
        error_message = f"mathers_iteration, Exception occurred: {e}\n"
        error_message += traceback.format_exc()
        print(error_message)

    return df_dict  # Return the df_dict at the end

def main_mathers_iteration_logic(db_queue):
    setup_logging()

    processes = []
    loop = asyncio.get_event_loop()
    df_dict = loop.run_until_complete(main_mathers_iteration_async_logic(db_queue))

    if not df_dict:
        print("No data to process.")
        return

    for table_name in timeframes.values():
        if df_dict.get(table_name) is not None and not df_dict.get(table_name).empty:
            p = Process(target=process_and_target, args=(df_dict.get(table_name), table_name, db_queue))
            processes.append(p)
            p.start()

    for p in processes:
        p.join()

    print("Sentinel reached. Ending mathers_iteration.")
