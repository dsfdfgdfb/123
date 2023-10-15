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
import numpy as np
import concurrent.futures


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
    query = f"SELECT time FROM {table_name} WHERE actualClass != 0.0 ORDER BY time DESC LIMIT 1;"
    
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
    return df

def write_to_db(df, table_name, db_queue):
    if df.empty:
        print(f"mathers_iteration, Skipping write for {table_name} as the DataFrame is empty.")
        return
   # logging.info(f"mathers_iteration, Attempting to write data for {table_name} to DB at path: {get_db_path}")

    # Structure the task to send to the mr_writer
    task = {
        "operation": "write_or_replace",
        "data": df.to_dict(orient="records"),  # Convert DataFrame to a list of dictionaries
        "table_name": table_name,
        "columns": df.columns.tolist()
    }
    #logging.info(f"Mathers_iteration, Adding task to db_queue: {task}")
    # Put the task into the shared queue
    db_queue.put(task)
    db_queue.put(None)
    print(f"Mathers_iteration:Sentinel is here Neo")
    
       
    

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
    R = (4 * RSILine + 3 * RSILine.shift(1) + 2 * RSILine.shift(2) + RSILine.shift(3)) / rsiLength
    return R

async def main_processing_loop(table_name, rsiOverbought, rsiOversold):
    try:        
        df = await async_read_from_db(table_name) 
        df['F_close'] = df['F_close']
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

def process_chunk(chunk):
    # This function processes a chunk of data (multiple pairs)
    results = []
    for (df, current_idx, next_pivot_idx) in chunk:
        result = process_pair(df, current_idx, next_pivot_idx)
        results.append(result)
    return results

def main_parallel_processing(df, pairs):
    # Decide on the number of processes
    num_processes = cpu_count()  # Or however many cores you wish to utilize

    # Split the pairs into chunks
    chunk_size = len(pairs) // num_processes
    chunks = [pairs[i:i + chunk_size] for i in range(0, len(pairs), chunk_size)]

    with Pool(num_processes) as pool:
        all_results = pool.map(process_chunk, chunks)

    # Flatten the results and update the dataframe
    for result_chunk in all_results:
        for result_df in result_chunk:
            df.update(result_df)
    
    return df


def compute_heikin_ashi(df):
    """
    Compute Heikin Ashi candlesticks for the given dataframe.
    
    Parameters:
    - df: Dataframe containing 'F_high', 'F_low', and 'F_close' columns.
    
    Returns:
    - df with additional 'F_HA_high', 'F_HA_low', and 'F_HA_close' columns.
    """
    # Compute HA Close
    df['F_HA_close'] = (df['F_high'] + df['F_low'] + df['F_close']) / 3

    # For the first data point, HA Open is just the regular open
    df['F_HA_open'] = df['F_close'].shift(1)
    df['F_HA_open'].fillna((df['F_high'] + df['F_low']) / 2, inplace=True)

    # Compute HA High and HA Low
    df['F_HA_high'] = df[['F_high', 'F_HA_open', 'F_HA_close']].max(axis=1)
    df['F_HA_low'] = df[['F_low', 'F_HA_open', 'F_HA_close']].min(axis=1)
    
    # Compute subsequent HA Open values
    for i in range(1, len(df)):
        df.at[df.index[i], 'F_HA_open'] = (df['F_HA_open'][i-1] + df['F_HA_close'][i-1]) / 2

    return df[['F_HA_high', 'F_HA_low', 'F_HA_close']]


def exponential_smoothing(series, alpha=0.3):
    """Apply Exponential Smoothing."""
    result = [series[0]]
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result


def linear_interpolate(start, end, num_points):
    """Linearly interpolate between two values."""
    return [start + i * (end - start) / (num_points - 1) for i in range(num_points)]


def process_pair(df, current_idx, next_pivot_idx):
    df['smoothedClose'] = exponential_smoothing(df['F_close'].tolist())
    df['linearClose'] = df['F_close'].interpolate(method='linear')
    current_pivot = df.iloc[current_idx]['pivot']
    next_pivot_high_price = df.iloc[next_pivot_idx]['F_close']
    ha_df = compute_heikin_ashi(df)
    df['F_HA_high'] = ha_df['F_HA_high']
    df['F_HA_low'] = ha_df['F_HA_low']
    df['F_HA_close'] = ha_df['F_HA_close']

    
    num_points = next_pivot_idx - current_idx + 1
    
    if current_pivot == 1:        
        df.loc[current_idx:next_pivot_idx, 'sellTarget'] = next_pivot_high_price - df.loc[current_idx:next_pivot_idx, 'F_close']
        df.at[current_idx, 'actualClass'] = 1
        df.loc[current_idx:next_pivot_idx, 'sellTargetP'] = (((next_pivot_high_price - df.loc[current_idx:next_pivot_idx, 'F_close']) / df.loc[current_idx:next_pivot_idx, 'F_close']) * 100).round(3)
        
        # Apply linear interpolation on 'sellTarget'
        df.loc[current_idx:next_pivot_idx, 'linearSell'] = linear_interpolate(df.iloc[current_idx]['sellTarget'], df.iloc[next_pivot_idx]['sellTarget'], num_points)

    elif current_pivot == 2:        
        df.loc[current_idx:next_pivot_idx, 'buyTarget'] = next_pivot_high_price - df.loc[current_idx:next_pivot_idx, 'F_close']
        df.at[current_idx, 'actualClass'] = 2
        df.loc[current_idx:next_pivot_idx, 'buyTargetP'] = (((next_pivot_high_price - df.loc[current_idx:next_pivot_idx, 'F_close']) / df.loc[current_idx:next_pivot_idx, 'F_close']) * 100).round(3)
        
        # Apply linear interpolation on 'buyTarget'
        df.loc[current_idx:next_pivot_idx, 'linearBuy'] = linear_interpolate(df.iloc[current_idx]['buyTarget'], df.iloc[next_pivot_idx]['buyTarget'], num_points)
        
    # Apply transformations    
    df['sellTargetSmooth'] = exponential_smoothing(df['sellTarget'])   
    df['buyTargetSmooth'] = exponential_smoothing(df['buyTarget'])
    df['highSmooth'] = exponential_smoothing(df['F_high'])
    df['lowSmooth'] = exponential_smoothing(df['F_low'])
    df['closeSmooth'] = exponential_smoothing(df['F_close'])  
    df['linearHigh'] = df['F_high'].interpolate(method='linear')
    df['linearLow'] = df['F_low'].interpolate(method='linear')
    df['linearClose'] = df['F_close'].interpolate(method='linear')

    return df.iloc[current_idx:next_pivot_idx+1]


#there's likely around 3 pivots that are incorrect before entering in here. fuck them
def adjust_buy_targets(df):
    buy_rows = df[(df['actualClass'] == 2) & (df['buyTarget'] > 0)].index.tolist()
    
    for idx in buy_rows:
        current_close = df.loc[idx, 'F_close']
        # Find the next row that would result in a correct buyTarget
        correct_idx = next((i for i in range(idx + 1, len(df)) if df.loc[i, 'F_close'] < current_close), None)
        
        if correct_idx:
            # Adjust the actualClass for the incorrect and correct indices
            df.at[correct_idx, 'actualClass'] = 2
            df.at[idx, 'actualClass'] = 0

            # Calculate the correct buyTarget and buyTargetP values
            next_pivot_high_price = df.loc[correct_idx, 'F_close']
            df.loc[idx:correct_idx, 'buyTarget'] = next_pivot_high_price - df.loc[idx:correct_idx, 'F_close']
            df.loc[idx:correct_idx, 'buyTargetP'] = (((next_pivot_high_price - df.loc[idx:correct_idx, 'F_close']) / df.loc[idx:correct_idx, 'F_close']) * 100).round(3)

    return df




async def process_and_target(df, table_name, db_queue):
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

            if current_state == 1:
                if row['isOversold'] == 1:
                    current_accumulation.append(idx)
                elif row['isOverbought'] == 1:
                    if current_accumulation:
                        extremum_idx = df.loc[current_accumulation, 'F_close'].idxmin()
                        state_indices.append(extremum_idx)
                        df.at[extremum_idx, 'rsiState'] = 1
                    current_accumulation = []
                    current_state = 2      
            elif current_state == 2:
                if row['isOverbought'] == 1:
                    current_accumulation.append(idx)
                elif row['isOversold'] == 1:
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

    df[['buyTarget', 'sellTarget', 'buyTargetP', 'sellTargetP', 'actualClass']] = [0, 0, 0, 0, 0]
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
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_pair, df, start_idx, end_idx) for start_idx, end_idx in pairs]
        for future in concurrent.futures.as_completed(futures):
            try:
                result_df = future.result()
                df.update(result_df)
            except Exception as e:
                logging.exception(f"Error in thread: {e}")
   
    #df = adjust_buy_targets(df)
    try:    
        loop = asyncio.get_running_loop()
        df.drop(columns=['pivot'], inplace=True)
        df.drop(columns=['rsiState'], inplace=True)


        await loop.run_in_executor(None, write_to_db, df, table_name, db_queue)
    except Exception as e:
        logging.exception(f"mathers_iteration, Error occurred while processing: {e}")  



    return table_name
    


async def main_mathers_iteration_async_logic(db_queue):
    setup_logging()
    try:
        print(f"mathers_iteration, Starting main function, init:mathers_iteration.py")
       
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
                print(f"mathers_iteration, Skipping processing for table due to not enough data.")
                successful_processing = False  # Set the flag to False if there's an error or None value
                continue
            df_dict[table] = df_selected
                
        logging.debug(f"mathers_iteration, Gathered data. Now processing and targeting.")
        tasks = [process_and_target(df_dict.get(table_name), table_name, db_queue) for table_name in timeframes.values() if df_dict.get(table_name) is not None]
        
        

        dfs = await asyncio.gather(*tasks)
        


    except Exception as e:
        print(f"mathers_iteration, Exception occurred: {e}")

def main_mathers_iteration_logic(db_queue):
    setup_logging()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main_mathers_iteration_async_logic(db_queue))
    print("Sentinel reached. Ending mathers_iteration.")

if __name__ == "__main__":
    db_queue = Queue()   
    os._exit(0)