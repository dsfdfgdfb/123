
import sys
sys.path.insert(0, './lobotomiser/modules')
from collections import defaultdict
from multiprocessing import Queue, Process, Pool, cpu_count
import logging
from lobotomiser.dbqueue import mr_writer, get_db_path
from lobotomiser.botdbqueue import the_narrator, bot_db_path, determine_length_range
#from lobotomiser.indicators_1h_high import indicatorvalues_main_1h_h
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import cwt, morlet
import threading
import os
import traceback
import numpy as np
import pandas as pd
import warnings
import aiosqlite
import asyncio
import pyti.detrended_price_oscillator as DPO
import pyti.double_exponential_moving_average as DEMA
import pyti.average_true_range as ATR
import pyti.bollinger_bands as BB
import pyti.chande_momentum_oscillator as CHANDE
from pyti.moving_average_convergence_divergence import moving_average_convergence_divergence as MACD_function
import pyti.on_balance_volume as OBV
import pyti.rate_of_change as ROC
import pyti.relative_strength_index as RSI
import pyti.volume_oscillator as VO
import pyti.momentum as MOM
from pyti.moving_average_envelope import upper_band as mae_upper_band
from pyti.moving_average_envelope import center_band as mae_center_band
from pyti.moving_average_envelope import lower_band as mae_lower_band
import pyti.standard_deviation as SD
import pyti.standard_variance as SV
import pyti.triangular_moving_average as TMA
import pyti.triple_exponential_moving_average as TEMA
import pyti.volatility as VOLA
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from lobotomiser.modules.ta.volume import EaseOfMovementIndicator
#from lobotomiser.modules.ta.momentum import KAMAIndicator, kama
from lobotomiser.modules.ta.momentum import WilliamsRIndicator as WPR
from lobotomiser.modules.ta.trend import STCIndicator
from lobotomiser.modules.ta.volume import VolumeWeightedAveragePrice



DB_INDICATOR_VALUES = './lobotomiser/data/indicatorvalues.db'
timeframe = 'onehourbtc'
indicatortimeframes = 'standard_1h_high'




VALID_INTERVALS = ['5m', '15m', '1h', '4h', '12h', '1d']





standard_indicators = [

   

        'DPO_3_u_h_1h', 'DPO_4_u_h_1h', 'DPO_5_u_h_1h', 'DPO_6_u_h_1h', 'DPO_7_u_h_1h', 'DPO_8_u_h_1h',
    'DPO_9_u_h_1h', 'DPO_10_u_h_1h', 'DPO_11_u_h_1h', 'DPO_12_u_h_1h', 'DPO_13_u_h_1h', 'DPO_14_u_h_1h',
    'DPO_15_u_h_1h', 'DPO_16_u_h_1h', 'DPO_17_u_h_1h', 'DPO_18_u_h_1h', 'DPO_19_u_h_1h', 'DPO_20_u_h_1h',
    'DPO_21_u_h_1h', 'DPO_22_u_h_1h', 'DPO_23_u_h_1h', 'DPO_24_u_h_1h', 'DPO_25_u_h_1h', 'DPO_26_u_h_1h',
    'DPO_27_u_h_1h', 'DPO_28_u_h_1h', 'DPO_29_u_h_1h', 'DPO_30_u_h_1h', 'DPO_31_u_h_1h', 'DPO_32_u_h_1h',
    'DPO_33_u_h_1h', 'DPO_34_u_h_1h', 'DPO_35_u_h_1h', 'DPO_36_u_h_1h', 'DPO_37_u_h_1h', 'DPO_38_u_h_1h',
    'DPO_39_u_h_1h', 'DPO_40_u_h_1h', 'DPO_41_u_h_1h', 'DPO_42_u_h_1h', 'DPO_43_u_h_1h', 'DPO_44_u_h_1h',
    'DPO_45_u_h_1h', 'DPO_46_u_h_1h', 'DPO_47_u_h_1h', 'DPO_48_u_h_1h', 'DPO_49_u_h_1h', 'DPO_50_u_h_1h',
    'DPO_51_u_h_1h', 'DPO_52_u_h_1h', 'DPO_53_u_h_1h', 'DPO_54_u_h_1h', 'DPO_55_u_h_1h', 'DPO_56_u_h_1h',
    'DPO_57_u_h_1h', 'DPO_58_u_h_1h', 'DPO_59_u_h_1h', 'DPO_60_u_h_1h', 'DPO_61_u_h_1h', 'DPO_62_u_h_1h',
    'DPO_63_u_h_1h', 'DPO_64_u_h_1h', 'DPO_65_u_h_1h', 'DPO_66_u_h_1h', 'DPO_67_u_h_1h', 'DPO_68_u_h_1h',
    'DPO_69_u_h_1h', 'DPO_70_u_h_1h',

    'DEMA_3_u_h_1h', 'DEMA_4_u_h_1h', 'DEMA_5_u_h_1h', 'DEMA_6_u_h_1h', 'DEMA_7_u_h_1h', 'DEMA_8_u_h_1h',
    'DEMA_9_u_h_1h', 'DEMA_10_u_h_1h', 'DEMA_11_u_h_1h', 'DEMA_12_u_h_1h', 'DEMA_13_u_h_1h', 'DEMA_14_u_h_1h',
    'DEMA_15_u_h_1h', 'DEMA_16_u_h_1h', 'DEMA_17_u_h_1h', 'DEMA_18_u_h_1h', 'DEMA_19_u_h_1h', 'DEMA_20_u_h_1h',
    'DEMA_21_u_h_1h', 'DEMA_22_u_h_1h', 'DEMA_23_u_h_1h', 'DEMA_24_u_h_1h', 'DEMA_25_u_h_1h', 'DEMA_26_u_h_1h',
    'DEMA_27_u_h_1h', 'DEMA_28_u_h_1h', 'DEMA_29_u_h_1h', 'DEMA_30_u_h_1h', 'DEMA_31_u_h_1h', 'DEMA_32_u_h_1h',
    'DEMA_33_u_h_1h', 'DEMA_34_u_h_1h', 'DEMA_35_u_h_1h', 'DEMA_36_u_h_1h', 'DEMA_37_u_h_1h', 'DEMA_38_u_h_1h',
    'DEMA_39_u_h_1h', 'DEMA_40_u_h_1h', 'DEMA_41_u_h_1h', 'DEMA_42_u_h_1h', 'DEMA_43_u_h_1h', 'DEMA_44_u_h_1h',
    'DEMA_45_u_h_1h', 'DEMA_46_u_h_1h', 'DEMA_47_u_h_1h', 'DEMA_48_u_h_1h', 'DEMA_49_u_h_1h', 'DEMA_50_u_h_1h',
    'DEMA_51_u_h_1h', 'DEMA_52_u_h_1h', 'DEMA_53_u_h_1h', 'DEMA_54_u_h_1h', 'DEMA_55_u_h_1h', 'DEMA_56_u_h_1h',
    'DEMA_57_u_h_1h', 'DEMA_58_u_h_1h', 'DEMA_59_u_h_1h', 'DEMA_60_u_h_1h', 'DEMA_61_u_h_1h', 'DEMA_62_u_h_1h',
    'DEMA_63_u_h_1h', 'DEMA_64_u_h_1h', 'DEMA_65_u_h_1h', 'DEMA_66_u_h_1h', 'DEMA_67_u_h_1h', 'DEMA_68_u_h_1h',
    'DEMA_69_u_h_1h', 'DEMA_70_u_h_1h',

  

      'MAE_3_upper_u_h_1h', 'MAE_4_upper_u_h_1h', 'MAE_5_upper_u_h_1h', 'MAE_6_upper_u_h_1h', 'MAE_7_upper_u_h_1h', 'MAE_8_upper_u_h_1h',
    'MAE_9_upper_u_h_1h', 'MAE_10_upper_u_h_1h', 'MAE_11_upper_u_h_1h', 'MAE_12_upper_u_h_1h', 'MAE_13_upper_u_h_1h', 'MAE_14_upper_u_h_1h',
    'MAE_15_upper_u_h_1h', 'MAE_16_upper_u_h_1h', 'MAE_17_upper_u_h_1h', 'MAE_18_upper_u_h_1h', 'MAE_19_upper_u_h_1h', 'MAE_20_upper_u_h_1h',
    'MAE_21_upper_u_h_1h', 'MAE_22_upper_u_h_1h', 'MAE_23_upper_u_h_1h', 'MAE_24_upper_u_h_1h', 'MAE_25_upper_u_h_1h', 'MAE_26_upper_u_h_1h',
    'MAE_27_upper_u_h_1h', 'MAE_28_upper_u_h_1h', 'MAE_29_upper_u_h_1h', 'MAE_30_upper_u_h_1h', 'MAE_31_upper_u_h_1h', 'MAE_32_upper_u_h_1h',
    'MAE_33_upper_u_h_1h', 'MAE_34_upper_u_h_1h', 'MAE_35_upper_u_h_1h', 'MAE_36_upper_u_h_1h', 'MAE_37_upper_u_h_1h', 'MAE_38_upper_u_h_1h',
    'MAE_39_upper_u_h_1h', 'MAE_40_upper_u_h_1h', 'MAE_41_upper_u_h_1h', 'MAE_42_upper_u_h_1h', 'MAE_43_upper_u_h_1h', 'MAE_44_upper_u_h_1h',
    'MAE_45_upper_u_h_1h', 'MAE_46_upper_u_h_1h', 'MAE_47_upper_u_h_1h', 'MAE_48_upper_u_h_1h', 'MAE_49_upper_u_h_1h', 'MAE_50_upper_u_h_1h',
    'MAE_51_upper_u_h_1h', 'MAE_52_upper_u_h_1h', 'MAE_53_upper_u_h_1h', 'MAE_54_upper_u_h_1h', 'MAE_55_upper_u_h_1h', 'MAE_56_upper_u_h_1h',
    'MAE_57_upper_u_h_1h', 'MAE_58_upper_u_h_1h', 'MAE_59_upper_u_h_1h', 'MAE_60_upper_u_h_1h', 'MAE_61_upper_u_h_1h', 'MAE_62_upper_u_h_1h',
    'MAE_63_upper_u_h_1h', 'MAE_64_upper_u_h_1h', 'MAE_65_upper_u_h_1h', 'MAE_66_upper_u_h_1h', 'MAE_67_upper_u_h_1h', 'MAE_68_upper_u_h_1h',
    'MAE_69_upper_u_h_1h', 'MAE_70_upper_u_h_1h',

        'MAE_3_center_u_h_1h', 'MAE_4_center_u_h_1h', 'MAE_5_center_u_h_1h', 'MAE_6_center_u_h_1h', 'MAE_7_center_u_h_1h', 'MAE_8_center_u_h_1h',
    'MAE_9_center_u_h_1h', 'MAE_10_center_u_h_1h', 'MAE_11_center_u_h_1h', 'MAE_12_center_u_h_1h', 'MAE_13_center_u_h_1h', 'MAE_14_center_u_h_1h',
    'MAE_15_center_u_h_1h', 'MAE_16_center_u_h_1h', 'MAE_17_center_u_h_1h', 'MAE_18_center_u_h_1h', 'MAE_19_center_u_h_1h', 'MAE_20_center_u_h_1h',
    'MAE_21_center_u_h_1h', 'MAE_22_center_u_h_1h', 'MAE_23_center_u_h_1h', 'MAE_24_center_u_h_1h', 'MAE_25_center_u_h_1h', 'MAE_26_center_u_h_1h',
    'MAE_27_center_u_h_1h', 'MAE_28_center_u_h_1h', 'MAE_29_center_u_h_1h', 'MAE_30_center_u_h_1h', 'MAE_31_center_u_h_1h', 'MAE_32_center_u_h_1h',
    'MAE_33_center_u_h_1h', 'MAE_34_center_u_h_1h', 'MAE_35_center_u_h_1h', 'MAE_36_center_u_h_1h', 'MAE_37_center_u_h_1h', 'MAE_38_center_u_h_1h',
    'MAE_39_center_u_h_1h', 'MAE_40_center_u_h_1h', 'MAE_41_center_u_h_1h', 'MAE_42_center_u_h_1h', 'MAE_43_center_u_h_1h', 'MAE_44_center_u_h_1h',
    'MAE_45_center_u_h_1h', 'MAE_46_center_u_h_1h', 'MAE_47_center_u_h_1h', 'MAE_48_center_u_h_1h', 'MAE_49_center_u_h_1h', 'MAE_50_center_u_h_1h',
    'MAE_51_center_u_h_1h', 'MAE_52_center_u_h_1h', 'MAE_53_center_u_h_1h', 'MAE_54_center_u_h_1h', 'MAE_55_center_u_h_1h', 'MAE_56_center_u_h_1h',
    'MAE_57_center_u_h_1h', 'MAE_58_center_u_h_1h', 'MAE_59_center_u_h_1h', 'MAE_60_center_u_h_1h', 'MAE_61_center_u_h_1h', 'MAE_62_center_u_h_1h',
    'MAE_63_center_u_h_1h', 'MAE_64_center_u_h_1h', 'MAE_65_center_u_h_1h', 'MAE_66_center_u_h_1h', 'MAE_67_center_u_h_1h', 'MAE_68_center_u_h_1h',
    'MAE_69_center_u_h_1h', 'MAE_70_center_u_h_1h',

        'MAE_3_lower_u_h_1h', 'MAE_4_lower_u_h_1h', 'MAE_5_lower_u_h_1h', 'MAE_6_lower_u_h_1h', 'MAE_7_lower_u_h_1h', 'MAE_8_lower_u_h_1h',
    'MAE_9_lower_u_h_1h', 'MAE_10_lower_u_h_1h', 'MAE_11_lower_u_h_1h', 'MAE_12_lower_u_h_1h', 'MAE_13_lower_u_h_1h', 'MAE_14_lower_u_h_1h',
    'MAE_15_lower_u_h_1h', 'MAE_16_lower_u_h_1h', 'MAE_17_lower_u_h_1h', 'MAE_18_lower_u_h_1h', 'MAE_19_lower_u_h_1h', 'MAE_20_lower_u_h_1h',
    'MAE_21_lower_u_h_1h', 'MAE_22_lower_u_h_1h', 'MAE_23_lower_u_h_1h', 'MAE_24_lower_u_h_1h', 'MAE_25_lower_u_h_1h', 'MAE_26_lower_u_h_1h',
    'MAE_27_lower_u_h_1h', 'MAE_28_lower_u_h_1h', 'MAE_29_lower_u_h_1h', 'MAE_30_lower_u_h_1h', 'MAE_31_lower_u_h_1h', 'MAE_32_lower_u_h_1h',
    'MAE_33_lower_u_h_1h', 'MAE_34_lower_u_h_1h', 'MAE_35_lower_u_h_1h', 'MAE_36_lower_u_h_1h', 'MAE_37_lower_u_h_1h', 'MAE_38_lower_u_h_1h',
    'MAE_39_lower_u_h_1h', 'MAE_40_lower_u_h_1h', 'MAE_41_lower_u_h_1h', 'MAE_42_lower_u_h_1h', 'MAE_43_lower_u_h_1h', 'MAE_44_lower_u_h_1h',
    'MAE_45_lower_u_h_1h', 'MAE_46_lower_u_h_1h', 'MAE_47_lower_u_h_1h', 'MAE_48_lower_u_h_1h', 'MAE_49_lower_u_h_1h', 'MAE_50_lower_u_h_1h',
    'MAE_51_lower_u_h_1h', 'MAE_52_lower_u_h_1h', 'MAE_53_lower_u_h_1h', 'MAE_54_lower_u_h_1h', 'MAE_55_lower_u_h_1h', 'MAE_56_lower_u_h_1h',
    'MAE_57_lower_u_h_1h', 'MAE_58_lower_u_h_1h', 'MAE_59_lower_u_h_1h', 'MAE_60_lower_u_h_1h', 'MAE_61_lower_u_h_1h', 'MAE_62_lower_u_h_1h',
    'MAE_63_lower_u_h_1h', 'MAE_64_lower_u_h_1h', 'MAE_65_lower_u_h_1h', 'MAE_66_lower_u_h_1h', 'MAE_67_lower_u_h_1h', 'MAE_68_lower_u_h_1h',
    'MAE_69_lower_u_h_1h', 'MAE_70_lower_u_h_1h',


    'MOM_3_u_h_1h', 'MOM_4_u_h_1h', 'MOM_5_u_h_1h', 'MOM_6_u_h_1h', 'MOM_7_u_h_1h', 'MOM_8_u_h_1h',
    'MOM_9_u_h_1h', 'MOM_10_u_h_1h', 'MOM_11_u_h_1h', 'MOM_12_u_h_1h', 'MOM_13_u_h_1h', 'MOM_14_u_h_1h',
    'MOM_15_u_h_1h', 'MOM_16_u_h_1h', 'MOM_17_u_h_1h', 'MOM_18_u_h_1h', 'MOM_19_u_h_1h', 'MOM_20_u_h_1h',
    'MOM_21_u_h_1h', 'MOM_22_u_h_1h', 'MOM_23_u_h_1h', 'MOM_24_u_h_1h', 'MOM_25_u_h_1h', 'MOM_26_u_h_1h',
    'MOM_27_u_h_1h', 'MOM_28_u_h_1h', 'MOM_29_u_h_1h', 'MOM_30_u_h_1h', 'MOM_31_u_h_1h', 'MOM_32_u_h_1h',
    'MOM_33_u_h_1h', 'MOM_34_u_h_1h', 'MOM_35_u_h_1h', 'MOM_36_u_h_1h', 'MOM_37_u_h_1h', 'MOM_38_u_h_1h',
    'MOM_39_u_h_1h', 'MOM_40_u_h_1h', 'MOM_41_u_h_1h', 'MOM_42_u_h_1h', 'MOM_43_u_h_1h', 'MOM_44_u_h_1h',
    'MOM_45_u_h_1h', 'MOM_46_u_h_1h', 'MOM_47_u_h_1h', 'MOM_48_u_h_1h', 'MOM_49_u_h_1h', 'MOM_50_u_h_1h',
    'MOM_51_u_h_1h', 'MOM_52_u_h_1h', 'MOM_53_u_h_1h', 'MOM_54_u_h_1h', 'MOM_55_u_h_1h', 'MOM_56_u_h_1h',
    'MOM_57_u_h_1h', 'MOM_58_u_h_1h', 'MOM_59_u_h_1h', 'MOM_60_u_h_1h', 'MOM_61_u_h_1h', 'MOM_62_u_h_1h',
    'MOM_63_u_h_1h', 'MOM_64_u_h_1h', 'MOM_65_u_h_1h', 'MOM_66_u_h_1h', 'MOM_67_u_h_1h', 'MOM_68_u_h_1h',
    'MOM_69_u_h_1h', 'MOM_70_u_h_1h',


    'SD_3_u_h_1h', 'SD_4_u_h_1h', 'SD_5_u_h_1h', 'SD_6_u_h_1h', 'SD_7_u_h_1h', 'SD_8_u_h_1h',
    'SD_9_u_h_1h', 'SD_10_u_h_1h', 'SD_11_u_h_1h', 'SD_12_u_h_1h', 'SD_13_u_h_1h', 'SD_14_u_h_1h',
    'SD_15_u_h_1h', 'SD_16_u_h_1h', 'SD_17_u_h_1h', 'SD_18_u_h_1h', 'SD_19_u_h_1h', 'SD_20_u_h_1h',
    'SD_21_u_h_1h', 'SD_22_u_h_1h', 'SD_23_u_h_1h', 'SD_24_u_h_1h', 'SD_25_u_h_1h', 'SD_26_u_h_1h',
    'SD_27_u_h_1h', 'SD_28_u_h_1h', 'SD_29_u_h_1h', 'SD_30_u_h_1h', 'SD_31_u_h_1h', 'SD_32_u_h_1h',
    'SD_33_u_h_1h', 'SD_34_u_h_1h', 'SD_35_u_h_1h', 'SD_36_u_h_1h', 'SD_37_u_h_1h', 'SD_38_u_h_1h',
    'SD_39_u_h_1h', 'SD_40_u_h_1h', 'SD_41_u_h_1h', 'SD_42_u_h_1h', 'SD_43_u_h_1h', 'SD_44_u_h_1h',
    'SD_45_u_h_1h', 'SD_46_u_h_1h', 'SD_47_u_h_1h', 'SD_48_u_h_1h', 'SD_49_u_h_1h', 'SD_50_u_h_1h',
    'SD_51_u_h_1h', 'SD_52_u_h_1h', 'SD_53_u_h_1h', 'SD_54_u_h_1h', 'SD_55_u_h_1h', 'SD_56_u_h_1h',
    'SD_57_u_h_1h', 'SD_58_u_h_1h', 'SD_59_u_h_1h', 'SD_60_u_h_1h', 'SD_61_u_h_1h', 'SD_62_u_h_1h',
    'SD_63_u_h_1h', 'SD_64_u_h_1h', 'SD_65_u_h_1h', 'SD_66_u_h_1h', 'SD_67_u_h_1h', 'SD_68_u_h_1h',
    'SD_69_u_h_1h', 'SD_70_u_h_1h',

    'SV_3_u_h_1h', 'SV_4_u_h_1h', 'SV_5_u_h_1h', 'SV_6_u_h_1h', 'SV_7_u_h_1h', 'SV_8_u_h_1h',
    'SV_9_u_h_1h', 'SV_10_u_h_1h', 'SV_11_u_h_1h', 'SV_12_u_h_1h', 'SV_13_u_h_1h', 'SV_14_u_h_1h',
    'SV_15_u_h_1h', 'SV_16_u_h_1h', 'SV_17_u_h_1h', 'SV_18_u_h_1h', 'SV_19_u_h_1h', 'SV_20_u_h_1h',
    'SV_21_u_h_1h', 'SV_22_u_h_1h', 'SV_23_u_h_1h', 'SV_24_u_h_1h', 'SV_25_u_h_1h', 'SV_26_u_h_1h',
    'SV_27_u_h_1h', 'SV_28_u_h_1h', 'SV_29_u_h_1h', 'SV_30_u_h_1h', 'SV_31_u_h_1h', 'SV_32_u_h_1h',
    'SV_33_u_h_1h', 'SV_34_u_h_1h', 'SV_35_u_h_1h', 'SV_36_u_h_1h', 'SV_37_u_h_1h', 'SV_38_u_h_1h',
    'SV_39_u_h_1h', 'SV_40_u_h_1h', 'SV_41_u_h_1h', 'SV_42_u_h_1h', 'SV_43_u_h_1h', 'SV_44_u_h_1h',
    'SV_45_u_h_1h', 'SV_46_u_h_1h', 'SV_47_u_h_1h', 'SV_48_u_h_1h', 'SV_49_u_h_1h', 'SV_50_u_h_1h',
    'SV_51_u_h_1h', 'SV_52_u_h_1h', 'SV_53_u_h_1h', 'SV_54_u_h_1h', 'SV_55_u_h_1h', 'SV_56_u_h_1h',
    'SV_57_u_h_1h', 'SV_58_u_h_1h', 'SV_59_u_h_1h', 'SV_60_u_h_1h', 'SV_61_u_h_1h', 'SV_62_u_h_1h',
    'SV_63_u_h_1h', 'SV_64_u_h_1h', 'SV_65_u_h_1h', 'SV_66_u_h_1h', 'SV_67_u_h_1h', 'SV_68_u_h_1h',
    'SV_69_u_h_1h', 'SV_70_u_h_1h',

    'TMA_3_u_h_1h', 'TMA_4_u_h_1h', 'TMA_5_u_h_1h', 'TMA_6_u_h_1h', 'TMA_7_u_h_1h', 'TMA_8_u_h_1h',
    'TMA_9_u_h_1h', 'TMA_10_u_h_1h', 'TMA_11_u_h_1h', 'TMA_12_u_h_1h', 'TMA_13_u_h_1h', 'TMA_14_u_h_1h',
    'TMA_15_u_h_1h', 'TMA_16_u_h_1h', 'TMA_17_u_h_1h', 'TMA_18_u_h_1h', 'TMA_19_u_h_1h', 'TMA_20_u_h_1h',
    'TMA_21_u_h_1h', 'TMA_22_u_h_1h', 'TMA_23_u_h_1h', 'TMA_24_u_h_1h', 'TMA_25_u_h_1h', 'TMA_26_u_h_1h',
    'TMA_27_u_h_1h', 'TMA_28_u_h_1h', 'TMA_29_u_h_1h', 'TMA_30_u_h_1h', 'TMA_31_u_h_1h', 'TMA_32_u_h_1h',
    'TMA_33_u_h_1h', 'TMA_34_u_h_1h', 'TMA_35_u_h_1h', 'TMA_36_u_h_1h', 'TMA_37_u_h_1h', 'TMA_38_u_h_1h',
    'TMA_39_u_h_1h', 'TMA_40_u_h_1h', 'TMA_41_u_h_1h', 'TMA_42_u_h_1h', 'TMA_43_u_h_1h', 'TMA_44_u_h_1h',
    'TMA_45_u_h_1h', 'TMA_46_u_h_1h', 'TMA_47_u_h_1h', 'TMA_48_u_h_1h', 'TMA_49_u_h_1h', 'TMA_50_u_h_1h',
    'TMA_51_u_h_1h', 'TMA_52_u_h_1h', 'TMA_53_u_h_1h', 'TMA_54_u_h_1h', 'TMA_55_u_h_1h', 'TMA_56_u_h_1h',
    'TMA_57_u_h_1h', 'TMA_58_u_h_1h', 'TMA_59_u_h_1h', 'TMA_60_u_h_1h', 'TMA_61_u_h_1h', 'TMA_62_u_h_1h',
    'TMA_63_u_h_1h', 'TMA_64_u_h_1h', 'TMA_65_u_h_1h', 'TMA_66_u_h_1h', 'TMA_67_u_h_1h', 'TMA_68_u_h_1h',
    'TMA_69_u_h_1h', 'TMA_70_u_h_1h',

    'TEMA_3_u_h_1h', 'TEMA_4_u_h_1h', 'TEMA_5_u_h_1h', 'TEMA_6_u_h_1h', 'TEMA_7_u_h_1h', 'TEMA_8_u_h_1h',
    'TEMA_9_u_h_1h', 'TEMA_10_u_h_1h', 'TEMA_11_u_h_1h', 'TEMA_12_u_h_1h', 'TEMA_13_u_h_1h', 'TEMA_14_u_h_1h',
    'TEMA_15_u_h_1h', 'TEMA_16_u_h_1h', 'TEMA_17_u_h_1h', 'TEMA_18_u_h_1h', 'TEMA_19_u_h_1h', 'TEMA_20_u_h_1h',
    'TEMA_21_u_h_1h', 'TEMA_22_u_h_1h', 'TEMA_23_u_h_1h', 'TEMA_24_u_h_1h', 'TEMA_25_u_h_1h', 'TEMA_26_u_h_1h',
    'TEMA_27_u_h_1h', 'TEMA_28_u_h_1h', 'TEMA_29_u_h_1h', 'TEMA_30_u_h_1h', 'TEMA_31_u_h_1h', 'TEMA_32_u_h_1h',
    'TEMA_33_u_h_1h', 'TEMA_34_u_h_1h', 'TEMA_35_u_h_1h', 'TEMA_36_u_h_1h', 'TEMA_37_u_h_1h', 'TEMA_38_u_h_1h',
    'TEMA_39_u_h_1h', 'TEMA_40_u_h_1h', 'TEMA_41_u_h_1h', 'TEMA_42_u_h_1h', 'TEMA_43_u_h_1h', 'TEMA_44_u_h_1h',
    'TEMA_45_u_h_1h', 'TEMA_46_u_h_1h', 'TEMA_47_u_h_1h', 'TEMA_48_u_h_1h', 'TEMA_49_u_h_1h', 'TEMA_50_u_h_1h',
    'TEMA_51_u_h_1h', 'TEMA_52_u_h_1h', 'TEMA_53_u_h_1h', 'TEMA_54_u_h_1h', 'TEMA_55_u_h_1h', 'TEMA_56_u_h_1h',
    'TEMA_57_u_h_1h', 'TEMA_58_u_h_1h', 'TEMA_59_u_h_1h', 'TEMA_60_u_h_1h', 'TEMA_61_u_h_1h', 'TEMA_62_u_h_1h',
    'TEMA_63_u_h_1h', 'TEMA_64_u_h_1h', 'TEMA_65_u_h_1h', 'TEMA_66_u_h_1h', 'TEMA_67_u_h_1h', 'TEMA_68_u_h_1h',
    'TEMA_69_u_h_1h', 'TEMA_70_u_h_1h',

     

    'VOLA_3_u_h_1h', 'VOLA_4_u_h_1h', 'VOLA_5_u_h_1h', 'VOLA_6_u_h_1h', 'VOLA_7_u_h_1h', 'VOLA_8_u_h_1h',
    'VOLA_9_u_h_1h', 'VOLA_10_u_h_1h', 'VOLA_11_u_h_1h', 'VOLA_12_u_h_1h', 'VOLA_13_u_h_1h', 'VOLA_14_u_h_1h',
    'VOLA_15_u_h_1h', 'VOLA_16_u_h_1h', 'VOLA_17_u_h_1h', 'VOLA_18_u_h_1h', 'VOLA_19_u_h_1h', 'VOLA_20_u_h_1h',
    'VOLA_21_u_h_1h', 'VOLA_22_u_h_1h', 'VOLA_23_u_h_1h', 'VOLA_24_u_h_1h', 'VOLA_25_u_h_1h', 'VOLA_26_u_h_1h',
    'VOLA_27_u_h_1h', 'VOLA_28_u_h_1h', 'VOLA_29_u_h_1h', 'VOLA_30_u_h_1h', 'VOLA_31_u_h_1h', 'VOLA_32_u_h_1h',
    'VOLA_33_u_h_1h', 'VOLA_34_u_h_1h', 'VOLA_35_u_h_1h', 'VOLA_36_u_h_1h', 'VOLA_37_u_h_1h', 'VOLA_38_u_h_1h',
    'VOLA_39_u_h_1h', 'VOLA_40_u_h_1h', 'VOLA_41_u_h_1h', 'VOLA_42_u_h_1h', 'VOLA_43_u_h_1h', 'VOLA_44_u_h_1h',
    'VOLA_45_u_h_1h', 'VOLA_46_u_h_1h', 'VOLA_47_u_h_1h', 'VOLA_48_u_h_1h', 'VOLA_49_u_h_1h', 'VOLA_50_u_h_1h',
    'VOLA_51_u_h_1h', 'VOLA_52_u_h_1h', 'VOLA_53_u_h_1h', 'VOLA_54_u_h_1h', 'VOLA_55_u_h_1h', 'VOLA_56_u_h_1h',
    'VOLA_57_u_h_1h', 'VOLA_58_u_h_1h', 'VOLA_59_u_h_1h', 'VOLA_60_u_h_1h', 'VOLA_61_u_h_1h', 'VOLA_62_u_h_1h',
    'VOLA_63_u_h_1h', 'VOLA_64_u_h_1h', 'VOLA_65_u_h_1h', 'VOLA_66_u_h_1h', 'VOLA_67_u_h_1h', 'VOLA_68_u_h_1h',
    'VOLA_69_u_h_1h', 'VOLA_70_u_h_1h',

    'ATR_4_u_h_1h', 'ATR_5_u_h_1h', 'ATR_6_u_h_1h', 'ATR_7_u_h_1h', 'ATR_8_u_h_1h', 'ATR_9_u_h_1h', 'ATR_10_u_h_1h',
    'ATR_11_u_h_1h', 'ATR_12_u_h_1h', 'ATR_13_u_h_1h', 'ATR_14_u_h_1h', 'ATR_15_u_h_1h', 'ATR_16_u_h_1h',
    'ATR_17_u_h_1h', 'ATR_18_u_h_1h', 'ATR_19_u_h_1h', 'ATR_20_u_h_1h', 'ATR_21_u_h_1h', 'ATR_22_u_h_1h',
    'ATR_23_u_h_1h', 'ATR_24_u_h_1h', 'ATR_25_u_h_1h', 'ATR_26_u_h_1h', 'ATR_27_u_h_1h', 'ATR_28_u_h_1h',
    'ATR_29_u_h_1h', 'ATR_30_u_h_1h', 'ATR_31_u_h_1h', 'ATR_32_u_h_1h', 'ATR_33_u_h_1h', 'ATR_34_u_h_1h',
    'ATR_35_u_h_1h', 'ATR_36_u_h_1h', 'ATR_37_u_h_1h', 'ATR_38_u_h_1h', 'ATR_39_u_h_1h', 'ATR_40_u_h_1h',
    'ATR_41_u_h_1h', 'ATR_42_u_h_1h', 'ATR_43_u_h_1h', 'ATR_44_u_h_1h', 'ATR_45_u_h_1h', 'ATR_46_u_h_1h',
    'ATR_47_u_h_1h', 'ATR_48_u_h_1h', 'ATR_50_u_h_1h', 'ATR_51_u_h_1h', 'ATR_52_u_h_1h',
    'ATR_53_u_h_1h', 'ATR_54_u_h_1h', 'ATR_55_u_h_1h', 'ATR_56_u_h_1h', 'ATR_57_u_h_1h', 'ATR_58_u_h_1h',
    'ATR_59_u_h_1h', 'ATR_60_u_h_1h', 'ATR_61_u_h_1h', 'ATR_62_u_h_1h', 'ATR_63_u_h_1h', 'ATR_64_u_h_1h',
    'ATR_65_u_h_1h', 'ATR_66_u_h_1h', 'ATR_67_u_h_1h', 'ATR_68_u_h_1h', 'ATR_69_u_h_1h', 'ATR_70_u_h_1h',

   
    'BB_l_14_u_h_1h', 'BB_l_21_u_h_1h', 'BB_l_24_u_h_1h', 'BB_l_25_u_h_1h', 'BB_l_26_u_h_1h', 'BB_l_27_u_h_1h', 'BB_l_28_u_h_1h', 'BB_l_36_u_h_1h', 'BB_l_44_u_h_1h',
    'BB_l_48_u_h_1h', 'BB_l_49_u_h_1h', 'BB_l_5_u_h_1h', 'BB_m_23_u_h_1h', 'BB_m_45_u_h_1h', 'BB_m_9_u_h_1h', 'BB_u_12_u_h_1h', 'BB_u_3_u_h_1h',
    'BB_u_37_u_h_1h', 'BB_u_38_u_h_1h', 'BB_u_39_u_h_1h', 'BB_u_44_u_h_1h', 'BB_u_45_u_h_1h', 'BB_u_46_u_h_1h', 'BB_u_49_u_h_1h',  
    'CHANDE_12_u_h_1h', 'CHANDE_18_u_h_1h', 'CHANDE_19_u_h_1h', 'CHANDE_27_u_h_1h', 'CHANDE_28_u_h_1h', 'CHANDE_31_u_h_1h', 'CHANDE_32_u_h_1h', 
    'CHANDE_33_u_h_1h', 'CHANDE_36_u_h_1h', 'CHANDE_44_u_h_1h', 'CHANDE_48_u_h_1h', 'CHANDE_49_u_h_1h', 'CHANDE_5_u_h_1h', 'CHANDE_7_u_h_1h',
    'CHANDE_8_u_h_1h', 'MACD_3_u_h_1h', 'PB_13_u_h_1h', 'PB_3_u_h_1h', 'PB_4_u_h_1h', 'PB_5_u_h_1h', 'PB_8_u_h_1h', 'ROC_11_u_h_1h', 'ROC_12_u_h_1h',
    'ROC_13_u_h_1h', 'ROC_21_u_h_1h', 'ROC_22_u_h_1h', 'ROC_23_u_h_1h', 'ROC_25_u_h_1h', 'ROC_27_u_h_1h', 'ROC_30_u_h_1h', 'ROC_45_u_h_1h', 'ROC_7_u_h_1h',
    'RSI_12_u_h_1h', 'RSI_13_u_h_1h', 'RSI_14_u_h_1h', 'RSI_15_u_h_1h', 'RSI_21_u_h_1h', 'RSI_23_u_h_1h', 'RSI_25_u_h_1h', 'RSI_28_u_h_1h', 'RSI_3_u_h_1h',
    'RSI_34_u_h_1h', 'RSI_35_u_h_1h', 'RSI_4_u_h_1h', 'RSI_49_u_h_1h', 'RSI_5_u_h_1h', 'RSI_6_u_h_1h', 'RSI_7_u_h_1h', 'RSI_9_u_h_1h', 'VO_3_u_h_1h',

    'OBV_3_u_h_1h',

    
  
]
add_dynamism_to_series = [

   

        'DPO_3_d_h_1h', 'DPO_4_d_h_1h', 'DPO_5_d_h_1h', 'DPO_6_d_h_1h', 'DPO_7_d_h_1h', 'DPO_8_d_h_1h',
    'DPO_9_d_h_1h', 'DPO_10_d_h_1h', 'DPO_11_d_h_1h', 'DPO_12_d_h_1h', 'DPO_13_d_h_1h', 'DPO_14_d_h_1h',
    'DPO_15_d_h_1h', 'DPO_16_d_h_1h', 'DPO_17_d_h_1h', 'DPO_18_d_h_1h', 'DPO_19_d_h_1h', 'DPO_20_d_h_1h',
    'DPO_21_d_h_1h', 'DPO_22_d_h_1h', 'DPO_23_d_h_1h', 'DPO_24_d_h_1h', 'DPO_25_d_h_1h', 'DPO_26_d_h_1h',
    'DPO_27_d_h_1h', 'DPO_28_d_h_1h', 'DPO_29_d_h_1h', 'DPO_30_d_h_1h', 'DPO_31_d_h_1h', 'DPO_32_d_h_1h',
    'DPO_33_d_h_1h', 'DPO_34_d_h_1h', 'DPO_35_d_h_1h', 'DPO_36_d_h_1h', 'DPO_37_d_h_1h', 'DPO_38_d_h_1h',
    'DPO_39_d_h_1h', 'DPO_40_d_h_1h', 'DPO_41_d_h_1h', 'DPO_42_d_h_1h', 'DPO_43_d_h_1h', 'DPO_44_d_h_1h',
    'DPO_45_d_h_1h', 'DPO_46_d_h_1h', 'DPO_47_d_h_1h', 'DPO_48_d_h_1h', 'DPO_49_d_h_1h', 'DPO_50_d_h_1h',
    'DPO_51_d_h_1h', 'DPO_52_d_h_1h', 'DPO_53_d_h_1h', 'DPO_54_d_h_1h', 'DPO_55_d_h_1h', 'DPO_56_d_h_1h',
    'DPO_57_d_h_1h', 'DPO_58_d_h_1h', 'DPO_59_d_h_1h', 'DPO_60_d_h_1h', 'DPO_61_d_h_1h', 'DPO_62_d_h_1h',
    'DPO_63_d_h_1h', 'DPO_64_d_h_1h', 'DPO_65_d_h_1h', 'DPO_66_d_h_1h', 'DPO_67_d_h_1h', 'DPO_68_d_h_1h',
    'DPO_69_d_h_1h', 'DPO_70_d_h_1h',

            'DEMA_3_d_h_1h', 'DEMA_4_d_h_1h', 'DEMA_5_d_h_1h', 'DEMA_6_d_h_1h', 'DEMA_7_d_h_1h', 'DEMA_8_d_h_1h',
    'DEMA_9_d_h_1h', 'DEMA_10_d_h_1h', 'DEMA_11_d_h_1h', 'DEMA_12_d_h_1h', 'DEMA_13_d_h_1h', 'DEMA_14_d_h_1h',
    'DEMA_15_d_h_1h', 'DEMA_16_d_h_1h', 'DEMA_17_d_h_1h', 'DEMA_18_d_h_1h', 'DEMA_19_d_h_1h', 'DEMA_20_d_h_1h',
    'DEMA_21_d_h_1h', 'DEMA_22_d_h_1h', 'DEMA_23_d_h_1h', 'DEMA_24_d_h_1h', 'DEMA_25_d_h_1h', 'DEMA_26_d_h_1h',
    'DEMA_27_d_h_1h', 'DEMA_28_d_h_1h', 'DEMA_29_d_h_1h', 'DEMA_30_d_h_1h', 'DEMA_31_d_h_1h', 'DEMA_32_d_h_1h',
    'DEMA_33_d_h_1h', 'DEMA_34_d_h_1h', 'DEMA_35_d_h_1h', 'DEMA_36_d_h_1h', 'DEMA_37_d_h_1h', 'DEMA_38_d_h_1h',
    'DEMA_39_d_h_1h', 'DEMA_40_d_h_1h', 'DEMA_41_d_h_1h', 'DEMA_42_d_h_1h', 'DEMA_43_d_h_1h', 'DEMA_44_d_h_1h',
    'DEMA_45_d_h_1h', 'DEMA_46_d_h_1h', 'DEMA_47_d_h_1h', 'DEMA_48_d_h_1h', 'DEMA_49_d_h_1h', 'DEMA_50_d_h_1h',
    'DEMA_51_d_h_1h', 'DEMA_52_d_h_1h', 'DEMA_53_d_h_1h', 'DEMA_54_d_h_1h', 'DEMA_55_d_h_1h', 'DEMA_56_d_h_1h',
    'DEMA_57_d_h_1h', 'DEMA_58_d_h_1h', 'DEMA_59_d_h_1h', 'DEMA_60_d_h_1h', 'DEMA_61_d_h_1h', 'DEMA_62_d_h_1h',
    'DEMA_63_d_h_1h', 'DEMA_64_d_h_1h', 'DEMA_65_d_h_1h', 'DEMA_66_d_h_1h', 'DEMA_67_d_h_1h', 'DEMA_68_d_h_1h',
    'DEMA_69_d_h_1h', 'DEMA_70_d_h_1h',

    'ATR_4_d_h_1h', 'ATR_5_d_h_1h', 'ATR_6_d_h_1h', 'ATR_7_d_h_1h', 'ATR_8_d_h_1h', 'ATR_9_d_h_1h', 'ATR_10_d_h_1h',
      'ATR_13_d_h_1h',  'ATR_15_d_h_1h', 'ATR_16_d_h_1h',
    'ATR_17_d_h_1h', 'ATR_18_d_h_1h', 'ATR_19_d_h_1h', 'ATR_20_d_h_1h', 'ATR_21_d_h_1h', 'ATR_22_d_h_1h',
    'ATR_23_d_h_1h', 'ATR_24_d_h_1h', 'ATR_25_d_h_1h', 'ATR_27_d_h_1h', 'ATR_28_d_h_1h',
     'ATR_30_d_h_1h',  'ATR_32_d_h_1h', 'ATR_33_d_h_1h', 'ATR_34_d_h_1h',
    'ATR_35_d_h_1h', 'ATR_36_d_h_1h', 'ATR_37_d_h_1h', 'ATR_38_d_h_1h', 'ATR_39_d_h_1h', 'ATR_40_d_h_1h',
    'ATR_41_d_h_1h', 'ATR_42_d_h_1h', 'ATR_43_d_h_1h', 'ATR_45_d_h_1h', 'ATR_46_d_h_1h',
    'ATR_47_d_h_1h', 'ATR_48_d_h_1h', 'ATR_49_d_h_1h', 'ATR_50_d_h_1h', 'ATR_51_d_h_1h', 'ATR_52_d_h_1h',
    'ATR_53_d_h_1h', 'ATR_54_d_h_1h', 'ATR_55_d_h_1h', 'ATR_56_d_h_1h', 'ATR_57_d_h_1h', 'ATR_58_d_h_1h',
    'ATR_59_d_h_1h', 'ATR_60_d_h_1h', 'ATR_61_d_h_1h', 'ATR_62_d_h_1h', 'ATR_63_d_h_1h', 'ATR_64_d_h_1h',
    'ATR_65_d_h_1h', 'ATR_66_d_h_1h', 'ATR_67_d_h_1h', 'ATR_68_d_h_1h', 'ATR_69_d_h_1h', 'ATR_70_d_h_1h',

    'CHANDE_16_d_h_1h', 'CHANDE_17_d_h_1h', 'CHANDE_19_d_h_1h', 'CHANDE_26_d_h_1h', 'CHANDE_38_d_h_1h',
    'CHANDE_6_d_h_1h', 'CHANDE_7_d_h_1h', 'MACD_3_d_h_1h',
    'ROC_14_d_h_1h', 'ROC_18_d_h_1h', 'ROC_20_d_h_1h', 'ROC_24_d_h_1h', 'ROC_36_d_h_1h', 'ROC_4_d_h_1h', 'ROC_40_d_h_1h',
    'ROC_7_d_h_1h', 'RSI_15_d_h_1h', 'RSI_24_d_h_1h', 'RSI_26_d_h_1h', 'RSI_3_d_h_1h', 'RSI_31_d_h_1h', 'RSI_33_d_h_1h', 'RSI_39_d_h_1h',
    'RSI_4_d_h_1h', 'RSI_47_d_h_1h', 'RSI_48_d_h_1h', 'RSI_49_d_h_1h', 'RSI_5_d_h_1h', 'RSI_9_d_h_1h', 
    
   

    'OBV_3_d_h_1h',



]
apply_gaussian_smoothing = [

    'DPO_3_g_h_1h', 'DPO_4_g_h_1h', 'DPO_5_g_h_1h', 'DPO_6_g_h_1h', 'DPO_7_g_h_1h', 'DPO_8_g_h_1h',
    'DPO_9_g_h_1h', 'DPO_10_g_h_1h', 'DPO_11_g_h_1h', 'DPO_12_g_h_1h', 'DPO_13_g_h_1h', 'DPO_14_g_h_1h',
    'DPO_15_g_h_1h', 'DPO_16_g_h_1h', 'DPO_17_g_h_1h', 'DPO_18_g_h_1h', 'DPO_19_g_h_1h', 'DPO_20_g_h_1h',
    'DPO_21_g_h_1h', 'DPO_22_g_h_1h', 'DPO_23_g_h_1h', 'DPO_24_g_h_1h', 'DPO_25_g_h_1h', 'DPO_26_g_h_1h',
    'DPO_27_g_h_1h', 'DPO_28_g_h_1h', 'DPO_29_g_h_1h', 'DPO_30_g_h_1h', 'DPO_31_g_h_1h', 'DPO_32_g_h_1h',
    'DPO_33_g_h_1h', 'DPO_34_g_h_1h', 'DPO_35_g_h_1h', 'DPO_36_g_h_1h', 'DPO_37_g_h_1h', 'DPO_38_g_h_1h',
    'DPO_39_g_h_1h', 'DPO_40_g_h_1h', 'DPO_41_g_h_1h', 'DPO_42_g_h_1h', 'DPO_43_g_h_1h', 'DPO_44_g_h_1h',
    'DPO_45_g_h_1h', 'DPO_46_g_h_1h', 'DPO_47_g_h_1h', 'DPO_48_g_h_1h', 'DPO_49_g_h_1h', 'DPO_50_g_h_1h',
    'DPO_51_g_h_1h', 'DPO_52_g_h_1h', 'DPO_53_g_h_1h', 'DPO_54_g_h_1h', 'DPO_55_g_h_1h', 'DPO_56_g_h_1h',
    'DPO_57_g_h_1h', 'DPO_58_g_h_1h', 'DPO_59_g_h_1h', 'DPO_60_g_h_1h', 'DPO_61_g_h_1h', 'DPO_62_g_h_1h',
    'DPO_63_g_h_1h', 'DPO_64_g_h_1h', 'DPO_65_g_h_1h', 'DPO_66_g_h_1h', 'DPO_67_g_h_1h', 'DPO_68_g_h_1h',
    'DPO_69_g_h_1h', 'DPO_70_g_h_1h',

        'DEMA_3_g_h_1h', 'DEMA_4_g_h_1h', 'DEMA_5_g_h_1h', 'DEMA_6_g_h_1h', 'DEMA_7_g_h_1h', 'DEMA_8_g_h_1h',
    'DEMA_9_g_h_1h', 'DEMA_10_g_h_1h', 'DEMA_11_g_h_1h', 'DEMA_12_g_h_1h', 'DEMA_13_g_h_1h', 'DEMA_14_g_h_1h',
    'DEMA_15_g_h_1h', 'DEMA_16_g_h_1h', 'DEMA_17_g_h_1h', 'DEMA_18_g_h_1h', 'DEMA_19_g_h_1h', 'DEMA_20_g_h_1h',
    'DEMA_21_g_h_1h', 'DEMA_22_g_h_1h', 'DEMA_23_g_h_1h', 'DEMA_24_g_h_1h', 'DEMA_25_g_h_1h', 'DEMA_26_g_h_1h',
    'DEMA_27_g_h_1h', 'DEMA_28_g_h_1h', 'DEMA_29_g_h_1h', 'DEMA_30_g_h_1h', 'DEMA_31_g_h_1h', 'DEMA_32_g_h_1h',
    'DEMA_33_g_h_1h', 'DEMA_34_g_h_1h', 'DEMA_35_g_h_1h', 'DEMA_36_g_h_1h', 'DEMA_37_g_h_1h', 'DEMA_38_g_h_1h',
    'DEMA_39_g_h_1h', 'DEMA_40_g_h_1h', 'DEMA_41_g_h_1h', 'DEMA_42_g_h_1h', 'DEMA_43_g_h_1h', 'DEMA_44_g_h_1h',
    'DEMA_45_g_h_1h', 'DEMA_46_g_h_1h', 'DEMA_47_g_h_1h', 'DEMA_48_g_h_1h', 'DEMA_49_g_h_1h', 'DEMA_50_g_h_1h',
    'DEMA_51_g_h_1h', 'DEMA_52_g_h_1h', 'DEMA_53_g_h_1h', 'DEMA_54_g_h_1h', 'DEMA_55_g_h_1h', 'DEMA_56_g_h_1h',
    'DEMA_57_g_h_1h', 'DEMA_58_g_h_1h', 'DEMA_59_g_h_1h', 'DEMA_60_g_h_1h', 'DEMA_61_g_h_1h', 'DEMA_62_g_h_1h',
    'DEMA_63_g_h_1h', 'DEMA_64_g_h_1h', 'DEMA_65_g_h_1h', 'DEMA_66_g_h_1h', 'DEMA_67_g_h_1h', 'DEMA_68_g_h_1h',
    'DEMA_69_g_h_1h', 'DEMA_70_g_h_1h',

    'ATR_4_g_h_1h', 'ATR_5_g_h_1h', 'ATR_6_g_h_1h', 'ATR_8_g_h_1h', 'ATR_19_g_h_1h', 'ATR_20_g_h_1h', 'ATR_22_g_h_1h',
    'ATR_23_g_h_1h', 'ATR_25_g_h_1h', 'ATR_26_g_h_1h', 'ATR_28_g_h_1h',
    'ATR_29_g_h_1h', 'ATR_31_g_h_1h', 'ATR_32_g_h_1h', 'ATR_36_g_h_1h', 'ATR_39_g_h_1h', 
    'ATR_41_g_h_1h', 'ATR_42_g_h_1h', 'ATR_45_g_h_1h', 'ATR_46_g_h_1h',
    'ATR_48_g_h_1h', 'ATR_50_g_h_1h', 'ATR_51_g_h_1h', 
    'ATR_53_g_h_1h', 'ATR_54_g_h_1h', 'ATR_57_g_h_1h', 'ATR_60_g_h_1h', 'ATR_61_g_h_1h', 'ATR_62_g_h_1h', 'ATR_63_g_h_1h', 
    'ATR_65_g_h_1h', 'ATR_67_g_h_1h', 'ATR_68_g_h_1h', 'ATR_69_g_h_1h', 'ATR_70_g_h_1h',

    'CHANDE_22_g_h_1h', 'CHANDE_3_g_h_1h',
    'CHANDE_32_g_h_1h', 'CHANDE_33_g_h_1h', 'CHANDE_36_g_h_1h', 'CHANDE_37_g_h_1h', 'CHANDE_38_g_h_1h', 'CHANDE_42_g_h_1h', 'CHANDE_44_g_h_1h',
    'CHANDE_8_g_h_1h', 'ROC_12_g_h_1h', 'ROC_22_g_h_1h',
    'ROC_24_g_h_1h', 'ROC_3_g_h_1h', 'ROC_38_g_h_1h', 'ROC_6_g_h_1h', 'ROC_7_g_h_1h', 'ROC_8_g_h_1h', 'RSI_10_g_h_1h', 'RSI_16_g_h_1h', 'RSI_17_g_h_1h',
    'RSI_18_g_h_1h', 'RSI_19_g_h_1h', 'RSI_20_g_h_1h', 'RSI_34_g_h_1h', 'RSI_44_g_h_1h', 'RSI_48_g_h_1h', 'RSI_49_g_h_1h', 'RSI_5_g_h_1h',
    



    'OBV_3_g_h_1h',

     

]
apply_wavelet_transform = [

  
    'DPO_3_w_h_1h', 'DPO_4_w_h_1h', 'DPO_5_w_h_1h', 'DPO_6_w_h_1h', 'DPO_7_w_h_1h', 'DPO_8_w_h_1h',
    'DPO_9_w_h_1h', 'DPO_10_w_h_1h', 'DPO_11_w_h_1h', 'DPO_12_w_h_1h', 'DPO_13_w_h_1h', 'DPO_14_w_h_1h',
    'DPO_15_w_h_1h', 'DPO_16_w_h_1h', 'DPO_17_w_h_1h', 'DPO_18_w_h_1h', 'DPO_19_w_h_1h', 'DPO_20_w_h_1h',
    'DPO_21_w_h_1h', 'DPO_22_w_h_1h', 'DPO_23_w_h_1h', 'DPO_24_w_h_1h', 'DPO_25_w_h_1h', 'DPO_26_w_h_1h',
    'DPO_27_w_h_1h', 'DPO_28_w_h_1h', 'DPO_29_w_h_1h', 'DPO_30_w_h_1h', 'DPO_31_w_h_1h', 'DPO_32_w_h_1h',
    'DPO_33_w_h_1h', 'DPO_34_w_h_1h', 'DPO_35_w_h_1h', 'DPO_36_w_h_1h', 'DPO_37_w_h_1h', 'DPO_38_w_h_1h',
    'DPO_39_w_h_1h', 'DPO_40_w_h_1h', 'DPO_41_w_h_1h', 'DPO_42_w_h_1h', 'DPO_43_w_h_1h', 'DPO_44_w_h_1h',
    'DPO_45_w_h_1h', 'DPO_46_w_h_1h', 'DPO_47_w_h_1h', 'DPO_48_w_h_1h', 'DPO_49_w_h_1h', 'DPO_50_w_h_1h',
    'DPO_51_w_h_1h', 'DPO_52_w_h_1h', 'DPO_53_w_h_1h', 'DPO_54_w_h_1h', 'DPO_55_w_h_1h', 'DPO_56_w_h_1h',
    'DPO_57_w_h_1h', 'DPO_58_w_h_1h', 'DPO_59_w_h_1h', 'DPO_60_w_h_1h', 'DPO_61_w_h_1h', 'DPO_62_w_h_1h',
    'DPO_63_w_h_1h', 'DPO_64_w_h_1h', 'DPO_65_w_h_1h', 'DPO_66_w_h_1h', 'DPO_67_w_h_1h', 'DPO_68_w_h_1h',
    'DPO_69_w_h_1h', 'DPO_70_w_h_1h',

        'DEMA_3_w_h_1h', 'DEMA_4_w_h_1h', 'DEMA_5_w_h_1h', 'DEMA_6_w_h_1h', 'DEMA_7_w_h_1h', 'DEMA_8_w_h_1h',
    'DEMA_9_w_h_1h', 'DEMA_10_w_h_1h', 'DEMA_11_w_h_1h', 'DEMA_12_w_h_1h', 'DEMA_13_w_h_1h', 'DEMA_14_w_h_1h',
    'DEMA_15_w_h_1h', 'DEMA_16_w_h_1h', 'DEMA_17_w_h_1h', 'DEMA_18_w_h_1h', 'DEMA_19_w_h_1h', 'DEMA_20_w_h_1h',
    'DEMA_21_w_h_1h', 'DEMA_22_w_h_1h', 'DEMA_23_w_h_1h', 'DEMA_24_w_h_1h', 'DEMA_25_w_h_1h', 'DEMA_26_w_h_1h',
    'DEMA_27_w_h_1h', 'DEMA_28_w_h_1h', 'DEMA_29_w_h_1h', 'DEMA_30_w_h_1h', 'DEMA_31_w_h_1h', 'DEMA_32_w_h_1h',
    'DEMA_33_w_h_1h', 'DEMA_34_w_h_1h', 'DEMA_35_w_h_1h', 'DEMA_36_w_h_1h', 'DEMA_37_w_h_1h', 'DEMA_38_w_h_1h',
    'DEMA_39_w_h_1h', 'DEMA_40_w_h_1h', 'DEMA_41_w_h_1h', 'DEMA_42_w_h_1h', 'DEMA_43_w_h_1h', 'DEMA_44_w_h_1h',
    'DEMA_45_w_h_1h', 'DEMA_46_w_h_1h', 'DEMA_47_w_h_1h', 'DEMA_48_w_h_1h', 'DEMA_49_w_h_1h', 'DEMA_50_w_h_1h',
    'DEMA_51_w_h_1h', 'DEMA_52_w_h_1h', 'DEMA_53_w_h_1h', 'DEMA_54_w_h_1h', 'DEMA_55_w_h_1h', 'DEMA_56_w_h_1h',
    'DEMA_57_w_h_1h', 'DEMA_58_w_h_1h', 'DEMA_59_w_h_1h', 'DEMA_60_w_h_1h', 'DEMA_61_w_h_1h', 'DEMA_62_w_h_1h',
    'DEMA_63_w_h_1h', 'DEMA_64_w_h_1h', 'DEMA_65_w_h_1h', 'DEMA_66_w_h_1h', 'DEMA_67_w_h_1h', 'DEMA_68_w_h_1h',
    'DEMA_69_w_h_1h', 'DEMA_70_w_h_1h',

    'ATR_4_w_h_1h', 'ATR_5_w_h_1h', 'ATR_6_w_h_1h', 'ATR_7_w_h_1h', 'ATR_8_w_h_1h', 'ATR_9_w_h_1h', 'ATR_10_w_h_1h',
    'ATR_11_w_h_1h', 'ATR_12_w_h_1h', 'ATR_13_w_h_1h', 'ATR_14_w_h_1h', 'ATR_15_w_h_1h', 'ATR_16_w_h_1h',
    'ATR_17_w_h_1h', 'ATR_18_w_h_1h', 'ATR_19_w_h_1h', 'ATR_20_w_h_1h', 'ATR_21_w_h_1h', 'ATR_22_w_h_1h',
    'ATR_23_w_h_1h', 'ATR_24_w_h_1h', 'ATR_25_w_h_1h', 'ATR_26_w_h_1h', 'ATR_27_w_h_1h', 'ATR_28_w_h_1h',
    'ATR_29_w_h_1h', 'ATR_30_w_h_1h', 'ATR_31_w_h_1h', 'ATR_32_w_h_1h', 'ATR_33_w_h_1h', 'ATR_34_w_h_1h',
    'ATR_35_w_h_1h', 'ATR_36_w_h_1h', 'ATR_37_w_h_1h', 'ATR_38_w_h_1h', 'ATR_39_w_h_1h', 'ATR_40_w_h_1h',
    'ATR_41_w_h_1h', 'ATR_42_w_h_1h', 'ATR_43_w_h_1h', 'ATR_44_w_h_1h', 'ATR_45_w_h_1h', 'ATR_46_w_h_1h',
    'ATR_47_w_h_1h', 'ATR_48_w_h_1h', 'ATR_49_w_h_1h', 'ATR_50_w_h_1h', 'ATR_51_w_h_1h', 'ATR_52_w_h_1h',
    'ATR_53_w_h_1h', 'ATR_54_w_h_1h', 'ATR_55_w_h_1h', 'ATR_56_w_h_1h', 'ATR_57_w_h_1h', 'ATR_58_w_h_1h',
    'ATR_59_w_h_1h', 'ATR_60_w_h_1h', 'ATR_61_w_h_1h', 'ATR_62_w_h_1h', 'ATR_63_w_h_1h', 'ATR_64_w_h_1h',
    'ATR_65_w_h_1h', 'ATR_66_w_h_1h', 'ATR_67_w_h_1h', 'ATR_68_w_h_1h', 'ATR_69_w_h_1h', 'ATR_70_w_h_1h',

    'CHANDE_14_w_h_1h', 'CHANDE_20_w_h_1h',
    'CHANDE_22_w_h_1h', 'CHANDE_25_w_h_1h', 'CHANDE_29_w_h_1h', 'CHANDE_37_w_h_1h', 'CHANDE_4_w_h_1h', 'CHANDE_7_w_h_1h',
    'ROC_28_w_h_1h', 'ROC_29_w_h_1h',
    'RSI_44_w_h_1h', 
    

    'OBV_3_w_h_1h',


   


]

detrend_series = [

 

    'DPO_3_e_h_1h', 'DPO_4_e_h_1h', 'DPO_5_e_h_1h', 'DPO_6_e_h_1h', 'DPO_7_e_h_1h', 'DPO_8_e_h_1h',
    'DPO_9_e_h_1h', 'DPO_10_e_h_1h', 'DPO_11_e_h_1h', 'DPO_12_e_h_1h', 'DPO_13_e_h_1h', 'DPO_14_e_h_1h',
    'DPO_15_e_h_1h', 'DPO_16_e_h_1h', 'DPO_17_e_h_1h', 'DPO_18_e_h_1h', 'DPO_19_e_h_1h', 'DPO_20_e_h_1h',
    'DPO_21_e_h_1h', 'DPO_22_e_h_1h', 'DPO_23_e_h_1h', 'DPO_24_e_h_1h', 'DPO_25_e_h_1h', 'DPO_26_e_h_1h',
    'DPO_27_e_h_1h', 'DPO_28_e_h_1h', 'DPO_29_e_h_1h', 'DPO_30_e_h_1h', 'DPO_31_e_h_1h', 'DPO_32_e_h_1h',
    'DPO_33_e_h_1h', 'DPO_34_e_h_1h', 'DPO_35_e_h_1h', 'DPO_36_e_h_1h', 'DPO_37_e_h_1h', 'DPO_38_e_h_1h',
    'DPO_39_e_h_1h', 'DPO_40_e_h_1h', 'DPO_41_e_h_1h', 'DPO_42_e_h_1h', 'DPO_43_e_h_1h', 'DPO_44_e_h_1h',
    'DPO_45_e_h_1h', 'DPO_46_e_h_1h', 'DPO_47_e_h_1h', 'DPO_48_e_h_1h', 'DPO_49_e_h_1h', 'DPO_50_e_h_1h',
    'DPO_51_e_h_1h', 'DPO_52_e_h_1h', 'DPO_53_e_h_1h', 'DPO_54_e_h_1h', 'DPO_55_e_h_1h', 'DPO_56_e_h_1h',
    'DPO_57_e_h_1h', 'DPO_58_e_h_1h', 'DPO_59_e_h_1h', 'DPO_60_e_h_1h', 'DPO_61_e_h_1h', 'DPO_62_e_h_1h',
    'DPO_63_e_h_1h', 'DPO_64_e_h_1h', 'DPO_65_e_h_1h', 'DPO_66_e_h_1h', 'DPO_67_e_h_1h', 'DPO_68_e_h_1h',
    'DPO_69_e_h_1h', 'DPO_70_e_h_1h',

    'DEMA_3_e_h_1h', 'DEMA_4_e_h_1h', 'DEMA_5_e_h_1h', 'DEMA_6_e_h_1h', 'DEMA_7_e_h_1h', 'DEMA_8_e_h_1h',
    'DEMA_9_e_h_1h', 'DEMA_10_e_h_1h', 'DEMA_11_e_h_1h', 'DEMA_12_e_h_1h', 'DEMA_13_e_h_1h', 'DEMA_14_e_h_1h',
    'DEMA_15_e_h_1h', 'DEMA_16_e_h_1h', 'DEMA_17_e_h_1h', 'DEMA_18_e_h_1h', 'DEMA_19_e_h_1h', 'DEMA_20_e_h_1h',
    'DEMA_21_e_h_1h', 'DEMA_22_e_h_1h', 'DEMA_23_e_h_1h', 'DEMA_24_e_h_1h', 'DEMA_25_e_h_1h', 'DEMA_26_e_h_1h',
    'DEMA_27_e_h_1h', 'DEMA_28_e_h_1h', 'DEMA_29_e_h_1h', 'DEMA_30_e_h_1h', 'DEMA_31_e_h_1h', 'DEMA_32_e_h_1h',
    'DEMA_33_e_h_1h', 'DEMA_34_e_h_1h', 'DEMA_35_e_h_1h', 'DEMA_36_e_h_1h', 'DEMA_37_e_h_1h', 'DEMA_38_e_h_1h',
    'DEMA_39_e_h_1h', 'DEMA_40_e_h_1h', 'DEMA_41_e_h_1h', 'DEMA_42_e_h_1h', 'DEMA_43_e_h_1h', 'DEMA_44_e_h_1h',
    'DEMA_45_e_h_1h', 'DEMA_46_e_h_1h', 'DEMA_47_e_h_1h', 'DEMA_48_e_h_1h', 'DEMA_49_e_h_1h', 'DEMA_50_e_h_1h',
    'DEMA_51_e_h_1h', 'DEMA_52_e_h_1h', 'DEMA_53_e_h_1h', 'DEMA_54_e_h_1h', 'DEMA_55_e_h_1h', 'DEMA_56_e_h_1h',
    'DEMA_57_e_h_1h', 'DEMA_58_e_h_1h', 'DEMA_59_e_h_1h', 'DEMA_60_e_h_1h', 'DEMA_61_e_h_1h', 'DEMA_62_e_h_1h',
    'DEMA_63_e_h_1h', 'DEMA_64_e_h_1h', 'DEMA_65_e_h_1h', 'DEMA_66_e_h_1h', 'DEMA_67_e_h_1h', 'DEMA_68_e_h_1h',
    'DEMA_69_e_h_1h', 'DEMA_70_e_h_1h',

    'ATR_4_e_h_1h', 'ATR_5_e_h_1h', 'ATR_6_e_h_1h', 'ATR_7_e_h_1h', 'ATR_8_e_h_1h', 'ATR_9_e_h_1h', 'ATR_10_e_h_1h',
    'ATR_11_e_h_1h', 'ATR_12_e_h_1h', 'ATR_13_e_h_1h', 'ATR_14_e_h_1h', 'ATR_15_e_h_1h', 'ATR_16_e_h_1h',
    'ATR_17_e_h_1h', 'ATR_18_e_h_1h', 'ATR_19_e_h_1h', 'ATR_20_e_h_1h', 'ATR_21_e_h_1h', 'ATR_22_e_h_1h',
    'ATR_23_e_h_1h', 'ATR_24_e_h_1h', 'ATR_25_e_h_1h', 'ATR_26_e_h_1h', 'ATR_27_e_h_1h', 'ATR_28_e_h_1h',
    'ATR_29_e_h_1h', 'ATR_30_e_h_1h', 'ATR_31_e_h_1h', 'ATR_32_e_h_1h', 'ATR_33_e_h_1h', 'ATR_34_e_h_1h',
    'ATR_35_e_h_1h', 'ATR_36_e_h_1h', 'ATR_37_e_h_1h', 'ATR_38_e_h_1h', 'ATR_39_e_h_1h', 'ATR_40_e_h_1h',
    'ATR_41_e_h_1h', 'ATR_42_e_h_1h', 'ATR_43_e_h_1h', 'ATR_44_e_h_1h', 'ATR_45_e_h_1h', 'ATR_46_e_h_1h',
    'ATR_47_e_h_1h', 'ATR_48_e_h_1h', 'ATR_49_e_h_1h', 'ATR_50_e_h_1h', 'ATR_51_e_h_1h', 'ATR_52_e_h_1h',
    'ATR_53_e_h_1h', 'ATR_54_e_h_1h', 'ATR_55_e_h_1h', 'ATR_56_e_h_1h', 'ATR_57_e_h_1h', 'ATR_58_e_h_1h',
    'ATR_59_e_h_1h', 'ATR_60_e_h_1h', 'ATR_61_e_h_1h', 'ATR_62_e_h_1h', 'ATR_63_e_h_1h', 'ATR_64_e_h_1h',
    'ATR_65_e_h_1h', 'ATR_66_e_h_1h', 'ATR_67_e_h_1h', 'ATR_68_e_h_1h', 'ATR_69_e_h_1h', 'ATR_70_e_h_1h',    
    'CHANDE_12_e_h_1h',
    'CHANDE_3_e_h_1h', 'CHANDE_38_e_h_1h', 'CHANDE_4_e_h_1h', 'CHANDE_5_e_h_1h', 'PB_18_e_h_1h',
    'PB_4_e_h_1h', 'ROC_11_e_h_1h', 'ROC_30_e_h_1h', 'ROC_37_e_h_1h', 'ROC_42_e_h_1h', 'ROC_8_e_h_1h', 'RSI_12_e_h_1h', 'RSI_3_e_h_1h', 
    'RSI_40_e_h_1h', 'RSI_42_e_h_1h', 'RSI_44_e_h_1h', 'RSI_49_e_h_1h', 
    

    'OBV_3_e_h_1h',


  
    
]

indicators_dict = {
    'standard_1h_high': standard_indicators,
    'add_dynamism_to_series_1h_high': add_dynamism_to_series,
    'apply_gaussian_smoothing_1h_high': apply_gaussian_smoothing,
    'apply_wavelet_transform_1h_high': apply_wavelet_transform,
    'detrend_series_1h_high': detrend_series
}
###creates an insane amount of indicator datapoints for each timestamp, sends it to be saved in the indicator values database. should be looped to keep it updated
def setup_logging():
    formatter = logging.Formatter('%(asctime)s - %(processName)-10s %(levelname)-8s %(message)s') # Added asctime for timestamps

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)    

    logger = logging.getLogger()
    logger.setLevel(logging.debug)

    logger.addHandler(console_handler)

    warnings.showwarning = custom_warning_handler
    
    # List of modules to suppress warnings for
    modules_to_suppress = [
      
        'pyti.average_true_range',
        'pyti.bollinger_bands',
        'pyti.dpo_momentum_oscillator',
        'pyti.chaikin_money_flow',
        'pyti.commodity_channel_index',
        'pyti.money_flow_index',
        'pyti.moving_average_convergence_divergence',
        'pyti.on_balance_volume',
        'pyti.rate_of_change',
        'pyti.relative_strength_index',
        'pyti.volume_oscillator',
        'pyti.williams_percent_r',
        'ta.volume',
        'ta.momentum',
        'ta.trend'
    ]
    
    for module in modules_to_suppress:
        warnings.filterwarnings('ignore', category=RuntimeWarning, module=module)


async def create_indicator_tables(botdb_queue):
    try:
        for table_name in indicators_dict.keys():
            table_name_with_suffix = f"{table_name}"  # Add the timeframe suffix to the table name
            # Generate the indicator columns based on the indicators for the current category
            indicator_columns = ["time INTEGER PRIMARY KEY"]
            indicator_columns.extend(
                f"{indicator} INTEGER"
                for indicator in indicators_dict[table_name]
            )
            indicator_columns_str = ", ".join(indicator_columns)

            # Construct the task and add it to the queue
            task = {
                "operation": "create_table",
                "table_name": table_name_with_suffix,
                "schema": indicator_columns_str
            }
            botdb_queue.put(task)

        botdb_queue.put(None)
    except Exception as e:
        logging.error(f"Error queuing table creation tasks: {e}")

###checks the database, see when the last timestamp was, if new data exists it gets processed and should bring enough data for rolling calculations
async def get_start_timestamp_for_indicators():
    start_timestamps = {}

    # Connect to the 'indicator' database
    bot_db_filepath = bot_db_path("standard_1h_high")

    indicators_dict = {
        'standard_1h_high': standard_indicators,
        'add_dynamism_to_series_1h_high': add_dynamism_to_series,
        'apply_gaussian_smoothing_1h_high': apply_gaussian_smoothing,
        'apply_wavelet_transform_1h_high': apply_wavelet_transform,
        'detrend_series_1h_high': detrend_series
    }

    async with aiosqlite.connect(bot_db_filepath) as readdb:
        async with readdb.cursor() as cursor:
            for table_name in indicators_dict:  # Using table_name directly
                print(f"Fetching data from: {bot_db_filepath} for table: {table_name}")



            # Fetch column names from the table
                await cursor.execute(f"PRAGMA table_info({table_name})")
                columns_info = await cursor.fetchall()

                # Get the last timestamp (sorted in descending order) where time has data
                await cursor.execute(f"SELECT time FROM {table_name} WHERE time IS NOT NULL ORDER BY time DESC LIMIT 1")
                result = await cursor.fetchone()
                start_time = result[0] if result else None

                # Determine the row number (rank) of this timestamp
                if start_time:
                    await cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE time <= ?", (start_time,))
                    rank_result = await cursor.fetchone()
                    rank = rank_result[0] if rank_result else None

                    # Get the total number of rows in the table
                    await cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    total_rows_result = await cursor.fetchone()
                    total_rows = total_rows_result[0] if total_rows_result else 0

                    # Check if there's data to calculate the adjusted rank
                    if rank is not None:
                        if length_range := determine_length_range(table_name):
                            max_window_size = max(length_range)
                        else:
                            max_window_size = 0
                        adjusted_rank = rank - 100

                        # Ensure the adjusted rank doesn't exceed the total number of rows
                        adjusted_rank = min(adjusted_rank, total_rows - 1)

                        # Fetch the timestamp of the adjusted rank
                        await cursor.execute(f"SELECT time FROM {table_name} ORDER BY time LIMIT 1 OFFSET ?", (adjusted_rank,))
                        adjusted_result = await cursor.fetchone()
                        adjusted_start_time = adjusted_result[0] if adjusted_result else None
                    else:
                        adjusted_start_time = None
                else:
                    adjusted_start_time = None

                start_timestamps[table_name] = adjusted_start_time

                # Print the adjusted start timestamp for the current table
                print(f"Adjusted start timestamp for {table_name}: {adjusted_start_time}")

        return start_timestamps, standard_indicators, add_dynamism_to_series, apply_gaussian_smoothing, apply_wavelet_transform, detrend_series





def determine_long_slow_periods(window_size):
    if length_range := determine_length_range(window_size):
        return length_range[0], length_range[-1]  # first and last values in the range
    else:
        return None, None

def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
    logging.warning(f'{filename}:{lineno}: {category.__name__}: {message}')

warnings.showwarning = custom_warning_handler

def add_standard_indicators(data_df, window_size):
    indicator_data = {}

    for indicator in standard_indicators:
        indicator_name, _, _ = indicator.split('_')
        column_name = f"{indicator_name}_{window_size}_u_h"
        indicator_data[indicator_name] = pd.Series(indicator_name.lower() + '_values')[column_name].tolist()

    return pd.DataFrame(indicator_data)

def add_dynamism_to_series(series, period):
    series = pd.Series(series)
    jh = series.rolling(period).max()
    jl = series.rolling(period).min()
    jc = (0.5 * (jh - jl))
    Hiline = jh - jc * 0.1  # DZbuy
    Loline = jl + jc * 0.1  # DZsell
    result = (Hiline + Loline) / 2
    result.fillna(0, inplace=True)
    return result



def apply_ema_to_series(series, period):
    result = series.ewm(span=period, adjust=False).mean()
    result.fillna(0, inplace=True)
    return result


sigma_default = 1.0

def apply_gaussian_smoothing(series, window_size, sigma=sigma_default):
    kernel = np.exp(-np.linspace(-(window_size // 2), window_size // 2, window_size) ** 2 / (2 * sigma ** 2))
    kernel /= np.sum(kernel)
    result = np.convolve(series, kernel, mode='same')
    result[np.isnan(result)] = 0
    return result


def standardize_series(series):
    result = (series - series.mean()) / series.std()
    result.fillna(0, inplace=True)
    return result

def detrend_series(series, period):
    trend = series.rolling(period).mean()
    trend.fillna(0, inplace=True)
    detrended = series - trend
    detrended.fillna(0, inplace=True)
    return detrended

def seasonal(series, window_size):
    from statsmodels.tsa.seasonal import seasonal_decompose
    result = seasonal_decompose(series, model='additive', period=window_size, extrapolate_trend='freq')
    residual = result.resid
    residual.fillna(0, inplace=True)
    return residual

def apply_wavelet_transform(series, width=30):
    transformed_series = cwt(series, morlet, np.arange(1, width+1))
    magnitude = np.abs(transformed_series)
    summed_magnitude = np.sum(magnitude, axis=0)
    summed_magnitude[np.isnan(summed_magnitude)] = 0
    return summed_magnitude



def calculate_standard(price_data_df):

    print("Starting calculate_standard for calculate_standard: 1h_high")

    indicator_values = {}
  
    close_prices = price_data_df['F_close'].tolist()
    high_prices = price_data_df['F_high'].tolist()
    low_prices = price_data_df['F_low'].tolist()   
    volume = price_data_df['F_volume'].tolist()    
    time_values = price_data_df['time'].tolist()
    indicator_values['time'] = time_values
    high_series = pd.Series(high_prices)
    low_series = pd.Series(low_prices)
    close_series = pd.Series(close_prices)  
    volume_series = pd.Series(volume)

    try:
       
        
     


        dpo_periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]        
        dpo_values_dict = {period: DPO.detrended_price_oscillator(high_prices, period) for period in dpo_periods}
        for period, dpo_values in dpo_values_dict.items():
            key = f'DPO_{period}_u_h_1h'
            dpo_values = np.where(np.isfinite(dpo_values), dpo_values, 0.0)
            rounded_and_int_values = (pd.Series(dpo_values))
            indicator_values[key] = rounded_and_int_values.tolist()

        dema_periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]        
        dema_values_dict = {period: DEMA.double_exponential_moving_average(high_prices, period) for period in dema_periods}
        for period, dema_values in dema_values_dict.items():
            key = f'dema_{period}_u_h_1h'
            dema_values = np.where(np.isfinite(dema_values), dema_values, 0.0)
            rounded_and_int_values = (pd.Series(dema_values))
            indicator_values[key] = rounded_and_int_values.tolist()
            

        mom_periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]        
        mom_values_dict = {period: MOM.momentum(high_prices, period) for period in mom_periods}
        for period, mom_values in mom_values_dict.items():
            key = f'MOM_{period}_u_h_1h'
            mom_values = np.where(np.isfinite(mom_values), mom_values, 0.0)
            rounded_and_int_values = (pd.Series(mom_values))
            indicator_values[key] = rounded_and_int_values.tolist()

        mae_periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]        
        mae_values_dict = {period: mae_upper_band(high_prices, period, 1) for period in mae_periods}
        for period, mae_values in mae_values_dict.items():
            key = f'MAE_{period}_upper_u_h_1h'
            mae_values = np.where(np.isfinite(mae_values), mae_values, 0.0)
            rounded_and_int_values = (pd.Series(mae_values))
            indicator_values[key] = rounded_and_int_values.tolist()

      

        mae_periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]        
        mae_values_dict = {period: mae_lower_band(high_prices, period, 1) for period in mae_periods}
        for period, mae_values in mae_values_dict.items():
            key = f'MAE_{period}_lower_u_h_1h'
            mae_values = np.where(np.isfinite(mae_values), mae_values, 0.0)
            rounded_and_int_values = (pd.Series(mae_values))
            indicator_values[key] = rounded_and_int_values.tolist()

        mae_periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]        
        mae_values_dict = {period: mae_center_band(high_prices, period) for period in mae_periods}
        for period, mae_values in mae_values_dict.items():
            key = f'MAE_{period}_center_u_h_1h'
            mae_values = np.where(np.isfinite(mae_values), mae_values, 0.0)
            rounded_and_int_values = (pd.Series(mae_values))
            indicator_values[key] = rounded_and_int_values.tolist()

        sd_periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]        
        sd_values_dict = {period: SD.standard_deviation(high_prices, period) for period in sd_periods}
        for period, sd_values in sd_values_dict.items():
            key = f'SD_{period}_u_h_1h'
            sd_values = np.where(np.isfinite(sd_values), sd_values, 0.0)
            rounded_and_int_values = (pd.Series(sd_values))
            indicator_values[key] = rounded_and_int_values.tolist()

        sv_periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]        
        sv_values_dict = {period: SV.standard_variance(high_prices, period) for period in sv_periods}
        for period, sv_values in sv_values_dict.items():
            key = f'SV_{period}_u_h_1h'
            sv_values = np.where(np.isfinite(sv_values), sv_values, 0.0)
            rounded_and_int_values = (pd.Series(sv_values))
            indicator_values[key] = rounded_and_int_values.tolist()

        tma_periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]        
        tma_values_dict = {period: TMA.triangular_moving_average(high_prices, period) for period in tma_periods}
        for period, tma_values in tma_values_dict.items():
            key = f'TMA_{period}_u_h_1h'
            tma_values = np.where(np.isfinite(tma_values), tma_values, 0.0)
            rounded_and_int_values = (pd.Series(tma_values))
            indicator_values[key] = rounded_and_int_values.tolist()

        tema_periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]        
        tema_values_dict = {period: TEMA.triple_exponential_moving_average(high_prices, period) for period in tema_periods}
        for period, tema_values in tema_values_dict.items():
            key = f'TEMA_{period}_u_h_1h'
            tema_values = np.where(np.isfinite(tema_values), tema_values, 0.0)
            rounded_and_int_values = (pd.Series(tema_values))
            indicator_values[key] = rounded_and_int_values.tolist()
            
        vola_periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]        
        vola_values_dict = {period: VOLA.volatility(high_prices, period) for period in vola_periods}
        for period, vola_values in vola_values_dict.items():
            key = f'VOLA_{period}_u_h_1h'
            vola_values = np.where(np.isfinite(vola_values), vola_values, 0.0)
            rounded_and_int_values = (pd.Series(vola_values))
            indicator_values[key] = rounded_and_int_values.tolist()



        atr_periods = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
        atr_values_dict = {period: ATR.average_true_range(high_prices, period) for period in atr_periods}
        for period, atr_values in atr_values_dict.items():
            key = f'ATR_{period}_u_h_1h'
            atr_values = np.where(np.isfinite(atr_values), atr_values, 0.0)                    
            rounded_and_int_values = (pd.Series(atr_values))             
            indicator_values[key] = rounded_and_int_values.tolist()  
            atr_values = np.where(np.isfinite(atr_values), atr_values, 0.0)
            rounded_and_int_values = (pd.Series(atr_values))
            indicator_values[key] = rounded_and_int_values.tolist()                             
            
       
        bb_periods_l = [14, 21, 24, 25, 26, 27, 28, 36, 44, 48, 49, 5]
        bb_values_l_dict = {period: BB.lower_bollinger_band(high_prices, period) for period in bb_periods_l}
        for period, l_bb_values in bb_values_l_dict.items():
            key = f'BB_l_{period}_u_h_1h'
            l_bb_values = np.where(np.isfinite(l_bb_values), l_bb_values, 0.0)
            rounded_and_int_values = (pd.Series(l_bb_values))
            indicator_values[key] = rounded_and_int_values.tolist()

       

        bb_periods_m = [23, 45, 9]
        bb_values_m_dict = {period: BB.lower_bollinger_band(high_prices, period) for period in bb_periods_m}
        for period, m_bb_values in bb_values_m_dict.items():
            key = f'BB_m_{period}_u_h_1h'
            m_bb_values = np.where(np.isfinite(m_bb_values), m_bb_values, 0.0)
            rounded_and_int_values = (pd.Series(m_bb_values))
            indicator_values[key] = rounded_and_int_values.tolist()


        bb_periods_u = [12, 3, 37, 38, 39, 44, 45, 46, 49]
        bb_values_u_dict = {period: BB.upper_bollinger_band(high_prices, period) for period in bb_periods_u} 
        for period, u_bb_values in bb_values_u_dict.items():
            key = f'BB_u_{period}_u_h_1h'           
            u_bb_values = np.where(np.isfinite(u_bb_values), u_bb_values, 0.0)            
            rounded_and_int_values = (pd.Series(u_bb_values))
            indicator_values[key] = rounded_and_int_values.tolist()
            



        chande_periods = [12, 18, 19, 27, 28, 31, 32, 33, 36, 44, 48, 49, 5, 7, 8]
        chande_values_dict = {period: CHANDE.chande_momentum_oscillator(high_prices, period) for period in chande_periods}
        for period, chande_values in chande_values_dict.items():
            key = f'CHANDE_{period}_u_h_1h'
            chande_values = np.where(np.isfinite(chande_values), chande_values, 0.0)
            rounded_and_int_values = (pd.Series(chande_values))          
            indicator_values[key] = rounded_and_int_values.tolist()
            
        # MACD
        macd_values_3 = MACD_function(high_prices, 3, 21).tolist()
        macd_values_3 = np.where(np.isfinite(macd_values_3), macd_values_3, 0.0)
        rounded_and_int_values = (pd.Series(macd_values_3))
        indicator_values['MACD_3_u_h_1h'] = rounded_and_int_values


        # OBV
        obv_periods = [3]
        obv_values_dict = {period: OBV.on_balance_volume(high_prices, volume) for period in obv_periods}
        for period, obv_values in obv_values_dict.items():
            key = f'OBV_{period}_u_h_1h'
            obv_values = np.where(np.isfinite(obv_values), obv_values, 0.0)
            rounded_and_int_values = (pd.Series(obv_values))
            indicator_values[key] = rounded_and_int_values.tolist()

        # Percent B
        pb_periods = [13, 3, 4, 5, 8]
        pb_values_dict = {period: BB.percent_b(high_prices, period) for period in pb_periods}
        for period, pb_values in pb_values_dict.items():
            key = f'PB_{period}_u_h_1h'
            pb_values = np.where(np.isfinite(pb_values), pb_values, 0.0)
            rounded_and_int_values = (pd.Series(pb_values)).tolist()
            indicator_values[key] = rounded_and_int_values

        # ROC
        roc_periods = [11, 12, 13, 21, 22, 23, 25, 27, 30, 45, 7]
        roc_values_dict = {period: ROC.rate_of_change(high_prices, period) for period in roc_periods}
        for period, roc_values in roc_values_dict.items():
            key = f'ROC_{period}_u_h_1h'
            roc_values = np.where(np.isfinite(roc_values), roc_values, 0.0)
            rounded_and_int_values = (pd.Series(roc_values))
            indicator_values[key] = rounded_and_int_values

      
        # RSI
        rsi_periods = [12, 13, 14, 15, 21, 23, 25, 28, 3, 34, 35, 4, 49, 5, 6, 7, 9]
        rsi_values_dict = {period: RSI.relative_strength_index(high_prices, period) for period in rsi_periods}
        for period, rsi_values in rsi_values_dict.items():
            key = f'RSI_{period}_u_h_1h'
            rsi_values = np.where(np.isfinite(rsi_values), rsi_values, 0.0)
            rounded_and_int_values = (pd.Series(rsi_values)).tolist()
            indicator_values[key] = rounded_and_int_values


        # VO (Assuming VO is a constant value of 3 for now)
        vo_3_values = np.array([3] * len(low_prices))
        vo_3_values = np.where(np.isfinite(vo_3_values), vo_3_values, 0.0)
        rounded_and_int_values = (pd.Series(vo_3_values)).tolist()
        indicator_values['VO_3_u_h_1h'] = rounded_and_int_values

              
      


    except Exception as e:
        tb = traceback.format_exc()
        print(f"Error occurred: {e}\n{tb} 1h_high")
    
    # Create a DataFrame from the indicator values
    standard_df = pd.DataFrame(indicator_values)

    print("Finished calculate_indicators for interval: 1h_high")

    return standard_df




def calculate_dynamism(price_data_df):
    print("Starting calculate_dynamism for interval: 1h_high")

    indicator_values = {}

    
    close_prices = price_data_df['F_close'].tolist()
    high_prices = price_data_df['F_high'].tolist()
    low_prices = price_data_df['F_low'].tolist()   
    volume = price_data_df['F_volume'].tolist()    
    time_values = price_data_df['time'].tolist()
    indicator_values['time'] = time_values
    high_series = pd.Series(high_prices)
    low_series = pd.Series(low_prices)
    close_series = pd.Series(close_prices)   
    volume_series = pd.Series(volume)
    
    try:
      

        dpo_periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
        for period in dpo_periods:
            dpo_values = pd.Series(DPO.detrended_price_oscillator(high_prices, period))
            dpo_dynamic_values = add_dynamism_to_series(dpo_values, period)
            key = f'DPO_{period}_d_h_1h'
            dynamic_values = np.where(np.isfinite(dpo_dynamic_values), dpo_dynamic_values, 0.0)
            indicator_values[key] = dynamic_values

        dema_periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
        dema_values = pd.Series(DEMA.double_exponential_moving_average(high_prices, period))
        dema_values_dict = {period: add_dynamism_to_series(dema_values, period) for period in dema_periods}
        for period, dema_dynamic_values in dema_values_dict.items():
            key = f'dema_{period}_d_h_1h'
            dema_dynamic_values = np.where(np.isfinite(dema_dynamic_values), dema_dynamic_values, 0.0)
            indicator_values[key] = dynamic_values


        # ATR
        atr_periods = [4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
        atr_values_dict = {period: ATR.average_true_range(high_prices, period) for period in atr_periods}
        for period, atr_values in atr_values_dict.items():
            key = f'ATR_{period}_d_h_1h'
            atr_values = np.where(np.isfinite(atr_values), atr_values, 0.0) 
            dynamic_values = add_dynamism_to_series(pd.Series(atr_values), period)
            indicator_values[key] = dynamic_values
            
        # CHANDE
        chande_periods = [16, 17, 19, 26, 38, 6, 7]
        chande_values_dict = {period: CHANDE.chande_momentum_oscillator(high_prices, period) for period in chande_periods}
        for period, chande_values in chande_values_dict.items():
            key = f'CHANDE_{period}_d_h_1h'
            chande_values = np.where(np.isfinite(chande_values), chande_values, 0.0) 
            dynamic_values = add_dynamism_to_series(pd.Series(chande_values), period)
            indicator_values[key] = dynamic_values

        


        # MACD
        macd_values_3 = MACD_function(high_prices, 3, 21)
        macd_values = np.where(np.isfinite(macd_values_3), macd_values_3, 0.0)
        dynamic_values = add_dynamism_to_series(pd.Series(macd_values), 3)
        indicator_values['MACD_3_d_h_1h'] = dynamic_values

        # OBV
        obv_periods = [3]
        obv_values = pd.Series(OBV.on_balance_volume(high_prices, volume))
        obv_values_dict = {period: add_dynamism_to_series(obv_values, period) for period in obv_periods}
        for period, obv_dynamic_values in obv_values_dict.items():
            key = f'OBV_{period}_d_h_1h'
            dynamic_values = np.where(np.isfinite(obv_dynamic_values), obv_dynamic_values, 0.0)
            indicator_values[key] = dynamic_values

            
     # ROC
        roc_periods = [14, 18, 20, 24, 36, 4, 40, 7]
        roc_values_dict = {period: ROC.rate_of_change(high_prices, period) for period in roc_periods}
        for period, roc_values in roc_values_dict.items():
            key = f'ROC_{period}_d_h_1h'
            roc_values = np.where(np.isfinite(roc_values), roc_values, 0.0)
            dynamic_values = add_dynamism_to_series(pd.Series(roc_values), period)
            indicator_values[key] = dynamic_values

        # RSI
        rsi_periods = [15, 24, 26, 3, 31, 33, 39, 4, 47, 48, 49, 5, 9]
        rsi_values_dict = {period: RSI.relative_strength_index(high_prices, period) for period in rsi_periods}
        for period, rsi_values in rsi_values_dict.items():
            key = f'RSI_{period}_d_h_1h'
            rsi_values = np.where(np.isfinite(rsi_values), rsi_values, 0.0)
            dynamic_values = add_dynamism_to_series(pd.Series(rsi_values), period)
            indicator_values[key] = dynamic_values



            

    except Exception as e:
        tb = traceback.format_exc()
        print(f"Error occurred: {e}\n{tb} 1h_high")

    # Create a DataFrame from the indicator values
    dynamism_df = pd.DataFrame(indicator_values)

    print("Finished calculate_indicators for interval: 1h_high")

    return dynamism_df



def calculate_gaussian(price_data_df):
    print("Starting calculate_gaussian for 1h_high")
   
    indicator_values = {}

    close_prices = price_data_df['F_close'].tolist()
    high_prices = price_data_df['F_high'].tolist()
    low_prices = price_data_df['F_low'].tolist()   
    volume = price_data_df['F_volume'].tolist()    
    time_values = price_data_df['time'].tolist()
    indicator_values['time'] = time_values
    high_series = pd.Series(high_prices)
    low_series = pd.Series(low_prices)
    close_series = pd.Series(close_prices)   
    volume_series = pd.Series(volume)
        
    try:
    

        dpo_periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
        dpo_values_dict = {period: DPO.detrended_price_oscillator(high_prices, period) for period in dpo_periods}
        for period, dpo_values in dpo_values_dict.items():
            key = f'DPO_{period}_g_h_1h'
            dpo_values = np.where(np.isfinite(dpo_values), dpo_values, 0.0)
            gaussian_values = apply_gaussian_smoothing(pd.Series(dpo_values), period)
            indicator_values[key] = gaussian_values

        dema_periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
        dema_values_dict = {period: DEMA.double_exponential_moving_average(high_prices, period) for period in dema_periods}
        for period, dema_values in dema_values_dict.items():
            key = f'DEMA_{period}_g_h_1h'
            dema_values = np.where(np.isfinite(dema_values), dema_values, 0.0)
            gaussian_values = apply_gaussian_smoothing(pd.Series(dema_values), period)
            indicator_values[key] = gaussian_values

        # ATR
        atr_periods = [4, 5, 6, 8, 19, 20, 22, 23, 25, 26, 28, 29, 31, 32, 36, 39, 41, 42, 45, 46, 48, 50,
        51, 53, 54, 57, 60, 61, 62, 63, 65, 67, 68, 69, 70]
        atr_values_dict = {period: ATR.average_true_range(high_prices, period) for period in atr_periods}
        for period, atr_values in atr_values_dict.items():
            key = f'ATR_{period}_g_h_1h'
            atr_values = np.where(np.isfinite(atr_values), atr_values, 0.0)
            atr_smoothed = apply_gaussian_smoothing(pd.Series(atr_values), period)
            indicator_values[key] = gaussian_values

      
        chande_periods = [22, 3, 32, 33, 36, 37, 38, 42, 44, 8]
        chande_values_dict = {period: CHANDE.chande_momentum_oscillator(high_prices, period) for period in chande_periods}
        for period, chande_values in chande_values_dict.items():
            key = f'CHANDE_{period}_g_h_1h'
            chande_values = np.where(np.isfinite(chande_values), chande_values, 0.0)
            gaussian_values= apply_gaussian_smoothing(pd.Series(chande_values), period)
            indicator_values[key] = gaussian_values


            

        # OBV
        obv_periods = [3]
        obv_values_dict = {period: OBV.on_balance_volume(high_prices, volume) for period in obv_periods}
        for period, obv_values in obv_values_dict.items():
            key = f'OBV_{period}_g_h_1h'
            obv_values = np.where(np.isfinite(obv_values), obv_values, 0.0)
            gaussian_values = apply_gaussian_smoothing(pd.Series(obv_values), period)
            indicator_values[key] = gaussian_values


        # ROC
        roc_periods = [12, 22, 24, 3, 38, 6, 7, 8]
        roc_values_dict = {period: ROC.rate_of_change(high_prices, period) for period in roc_periods}
        for period, roc_values in roc_values_dict.items():
            key = f'ROC_{period}_g_h_1h'
            roc_values = np.where(np.isfinite(roc_values), roc_values, 0.0)
            gaussian_values = apply_gaussian_smoothing(pd.Series(roc_values), period)
            indicator_values[key] = gaussian_values

        # RSI
        rsi_periods = [10, 16, 17, 18, 19, 20, 34, 44, 48, 49, 5]
        rsi_values_dict = {period: RSI.relative_strength_index(high_prices, period) for period in rsi_periods}
        for period, rsi_values in rsi_values_dict.items():
            key = f'RSI_{period}_g_h_1h'
            rsi_values = np.where(np.isfinite(rsi_values), rsi_values, 0.0)
            gaussian_values = apply_gaussian_smoothing(pd.Series(rsi_values), period)
            indicator_values[key] = gaussian_values


 
      
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Error occurred: {e}\n{tb} 1h_high")

    # Create a DataFrame from the indicator values
    gaussian_df = pd.DataFrame(indicator_values)


    print("Finished calculate_gaussian for interval: 1h_high")

    return gaussian_df

        
def calculate_wavelet(price_data_df):
    print("Starting calculate_wavelet for interval: 1h_high")
    
    indicator_values = {}

    close_prices = price_data_df['F_close'].tolist()
    high_prices = price_data_df['F_high'].tolist()
    low_prices = price_data_df['F_low'].tolist()   
    volume = price_data_df['F_volume'].tolist()    
    time_values = price_data_df['time'].tolist()
    indicator_values['time'] = time_values
    high_series = pd.Series(high_prices)
    low_series = pd.Series(low_prices)
    close_series = pd.Series(close_prices)   
    volume_series = pd.Series(volume)
   
   
    try:
        

        dpo_periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
        dpo_values_dict = {period:  DPO.detrended_price_oscillator(high_prices, period) for period in dpo_periods}
        for period, dpo_values in dpo_values_dict.items():
            key = f'DPO_{period}_w_h_1h'
            dpo_values = np.where(np.isfinite(dpo_values), dpo_values, 0.0)
            transformed_values = apply_wavelet_transform(pd.Series(dpo_values), period)
            indicator_values[key] = transformed_values

        dema_periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
        dema_values_dict = {period: DEMA.double_exponential_moving_average(high_prices, period) for period in dema_periods}
        for period, dema_values in dema_values_dict.items():
            key = f'DEMA_{period}_w_h_1h'
            dema_values = np.where(np.isfinite(dema_values), dema_values, 0.0)
            transformed_values = apply_wavelet_transform(pd.Series(dema_values), period)
            indicator_values[key] = transformed_values

        # ATR
        atr_periods = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
        atr_values_dict = {period: ATR.average_true_range(high_prices, period) for period in atr_periods}
        for period, atr_values in atr_values_dict.items():
            key = f'ATR_{period}_w_h_1h'
            atr_values = np.where(np.isfinite(atr_values), atr_values, 0.0)
            transformed_values = apply_wavelet_transform(pd.Series(atr_values), period)
            indicator_values[key] = transformed_values
                
        # CHANDE
        chande_periods = [14, 20, 22, 25, 29, 37, 4, 7]
        chande_values_dict = {period: CHANDE.chande_momentum_oscillator(high_prices, period) for period in chande_periods}
        for period, chande_values in chande_values_dict.items():
            key = f'CHANDE_{period}_w_h_1h'
            chande_values = np.where(np.isfinite(chande_values), chande_values, 0.0)
            transformed_values = apply_wavelet_transform(pd.Series(chande_values), period)
            indicator_values[key] = transformed_values


        # OBV
        obv_periods = [3]
        obv_values_dict = {period: OBV.on_balance_volume(high_prices, volume) for period in obv_periods}
        for period, obv_values in obv_values_dict.items():
            key = f'OBV_{period}_w_h_1h'
            obv_values = np.where(np.isfinite(obv_values), obv_values, 0.0)
            transformed_values = apply_wavelet_transform(pd.Series(obv_values), period)
            indicator_values[key] = transformed_values

        # ROC
        roc_periods = [28, 29]
        roc_values_dict = {period: ROC.rate_of_change(high_prices, period) for period in roc_periods}
        for period, roc_values in roc_values_dict.items():
            key = f'ROC_{period}_w_h_1h'
            roc_values = np.where(np.isfinite(roc_values), roc_values, 0.0)
            transformed_values = apply_wavelet_transform(pd.Series(roc_values), period)
            indicator_values[key] = transformed_values

        # RSI
        rsi_periods = [44]
        rsi_values_dict = {period: RSI.relative_strength_index(high_prices, period) for period in rsi_periods}
        for period, rsi_values in rsi_values_dict.items():
            key = f'RSI_{period}_w_h_1h'
            rsi_values = np.where(np.isfinite(rsi_values), rsi_values, 0.0)
            transformed_values = apply_wavelet_transform(pd.Series(rsi_values), period)
            indicator_values[key] = transformed_values

      
    
        
     
    
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Error occurred: {e}\n{tb} 1h_high")

    # Create a DataFrame from the indicator values
    wavelet_df = pd.DataFrame(indicator_values)    

    print("Finished calculate_wavelet for interval: 1h_high")

    return wavelet_df

def calculate_detrend(price_data_df):
    print("Starting calculate_detrend for interval: 1h_high")

    indicator_values = {}
    
    close_prices = price_data_df['F_close'].tolist()
    high_prices = price_data_df['F_high'].tolist()
    low_prices = price_data_df['F_low'].tolist()   
    volume = price_data_df['F_volume'].tolist()    
    time_values = price_data_df['time'].tolist()
    indicator_values['time'] = time_values
    high_series = pd.Series(high_prices)
    low_series = pd.Series(low_prices)
    close_series = pd.Series(close_prices)   
    volume_series = pd.Series(volume)

    try:  

        
       
     
        # ATR
        atr_periods = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
        atr_values_dict = {period: ATR.average_true_range(high_prices, period) for period in atr_periods}
        for period, atr_values in atr_values_dict.items():
            key = f'ATR_{period}_e_h_1h'
            atr_values = np.where(np.isfinite(atr_values), atr_values, 0.0)
            detrended_values = detrend_series(pd.Series(atr_values), period)
            indicator_values[key] = detrended_values

            
        dpo_periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
        dpo_values_dict = {period: DPO.detrended_price_oscillator(high_prices, period) for period in dpo_periods}
        for period, dpo_dynamic_values in dpo_values_dict.items():
            key = f'dpo_{period}_e_h_1h'
            dpo_values = np.where(np.isfinite(dpo_dynamic_values), dpo_dynamic_values, 0.0)
            detrended_values = detrend_series(pd.Series(dpo_values), period)
            indicator_values[key] = detrended_values

        dema_periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
        dema_values_dict = {period: DEMA.double_exponential_moving_average(high_prices, period) for period in dema_periods}
        for period, dema_dynamic_values in dema_values_dict.items():
            key = f'DEMA_{period}_e_h_1h'
            dema_values = np.where(np.isfinite(dema_dynamic_values), dema_dynamic_values, 0.0)
            detrended_values = detrend_series(pd.Series(dema_values), period)
            indicator_values[key] = detrended_values
              

                # CHANDE
        chande_periods = [12, 3, 38, 4, 5]
        chande_values_dict = {period: CHANDE.chande_momentum_oscillator(high_prices, period) for period in chande_periods}
        for period, chande_values in chande_values_dict.items():
            key = f'CHANDE_{period}_e_h_1h'
            chande_values = np.where(np.isfinite(chande_values), chande_values, 0.0)
            detrended_values = detrend_series(pd.Series(chande_values), period)
            indicator_values[key] = detrended_values

            


        # OBV
        obv_periods = [3]
        obv_values_dict = {period: OBV.on_balance_volume(high_prices, volume) for period in obv_periods}
        for period, obv_values in obv_values_dict.items():
            key = f'OBV_{period}_e_h_1h'
            obv_values = np.where(np.isfinite(obv_values), obv_values, 0.0)
            detrended_values = detrend_series(pd.Series(obv_values), period)
            indicator_values[key] = detrended_values


        # ROC
        roc_periods = [11, 30, 37, 42, 8]
        roc_values_dict = {period: ROC.rate_of_change(high_prices, period) for period in roc_periods}
        for period, roc_values in roc_values_dict.items():
            key = f'ROC_{period}_e_h_1h'
            roc_values = np.where(np.isfinite(roc_values), roc_values, 0.0)
            detrended_values = detrend_series(pd.Series(roc_values), period)
            indicator_values[key] = detrended_values


        # RSI
        rsi_periods = [12, 3, 40, 42, 44, 49]
        rsi_values_dict = {period: RSI.relative_strength_index(high_prices, period) for period in rsi_periods}
        for period, rsi_values in rsi_values_dict.items():
            key = f'RSI_{period}_e_h_1h'
            rsi_values = np.where(np.isfinite(rsi_values), rsi_values, 0.0)
            detrended_values = detrend_series(pd.Series(rsi_values), period)
            indicator_values[key] = detrended_values


            
      

        # Percent B
        pb_periods = [18,4]
        pb_values_dict = {period: BB.percent_b(high_prices, period) for period in pb_periods}
        for period, pb_values in pb_values_dict.items():
            key = f'PB_{period}_e_h_1h'
            pb_values = np.where(np.isfinite(pb_values), pb_values, 0.0)
            detrended_values = detrend_series(pd.Series(pb_values), period)
            indicator_values[key] = detrended_values
            



    except Exception as e:
        tb = traceback.format_exc()
        print(f"Error occurred: {e}\n{tb} 15m_high")

    # Create a DataFrame from the indicator values
    detrend_df = pd.DataFrame(indicator_values)


    print("Finished calculate_detrend for interval: 15m_high")

    return detrend_df


async def fetch_data_from_db(global_start_timestamps):
    
    print("Starting fetch_data_from_db for timeframe: 15m_high")
    table_name = timeframe
    indicator_table_name = indicatortimeframes

    earliest_timestamp = global_start_timestamps.get(indicator_table_name)

    db_path = get_db_path(table_name)
    async with aiosqlite.connect(db_path) as db:
        async with db.cursor() as cursor:
            if earliest_timestamp:
                await cursor.execute(f"SELECT time, F_open, F_high, F_low, F_close, F_volume FROM {table_name} WHERE time >= ?", (earliest_timestamp,))
            else:
                await cursor.execute(f"SELECT time, F_open, F_high, F_low, F_close, F_volume FROM {table_name}")

            rows = await cursor.fetchall()
            columns = [column[0] for column in cursor.description]
            data_df = pd.DataFrame(rows, columns=columns).sort_values(by="time")     
    print("Finished fetch_data_from_db for timeframe: 15m_high ", len(data_df))
    
    return data_df

def manage_process(func_name, function, args):
    print(f"Starting {func_name} on a new thread 15m_high...")
    function(*args)
    print(f"Completed {func_name} 15m_high.")

def run_parallel_calculations(data_df):
    results = {}
    
    with ProcessPoolExecutor() as executor:
        tasks = {
            "standard_1h_high": executor.submit(calculate_standard, data_df),
            "add_dynamism_to_series_1h_high": executor.submit(calculate_dynamism, data_df),
            "apply_gaussian_smoothing_1h_high": executor.submit(calculate_gaussian, data_df),
            "apply_wavelet_transform_1h_high": executor.submit(calculate_wavelet, data_df),
            "detrend_series_1h_high": executor.submit(calculate_detrend, data_df)
        }
        
        for key, future in tasks.items():
            results[key] = future.result()
        
    print("returned parallel calc_15m_high")
    return results



async def process_indicator_series(botdb_queue, global_start_timestamps): 
    print(f"Fetching data from database for indicator_15m_high")
    data_df = await fetch_data_from_db(global_start_timestamps)
    print(f"Fetched {len(data_df)} 15m_high")
    
    dfs_dict = run_parallel_calculations(data_df)

    for key, dfs in dfs_dict.items():
        dfs = dfs.applymap(lambda x: round(x, 2) if isinstance(x, (float, int)) else x)
        dfs = dfs.fillna(0).astype(int)
        print(f"Finished processing {key} 15m_high")

        # Check if DataFrame is empty and skip the rest if it is
        if dfs.empty:
            print(f"No results for {key}. Skipping_15m_high.")
            continue

        # Add each row of the DataFrame to the queue
        table_name = key  # Directly using the key as the table name
        batch_size = 1000
        data_batch = []

        for _, row in dfs.iterrows():
            data_dict = row.to_dict()
            data_batch.append(data_dict)
            if len(data_batch) == batch_size:
                task = {
                    "operation": "write_or_replace",
                    "data": data_batch,
                    "table_name": table_name,
                    "columns": list(data_dict.keys())
                }
                botdb_queue.put(task)
                data_batch = []

        if data_batch:  # handle the remaining rows
            task = {
                "operation": "write_or_replace",
                "data": data_batch,
                "table_name": table_name,
                "columns": list(data_dict.keys())
            }
            botdb_queue.put(task)

    print(f"Sentinel Indicatorvalues_1h, high")
    botdb_queue.put(None)


async def indicatorvalues_async_main(botdb_queue, table_created_event_1h_h):
    df_indicators = {}  # This is now a dictionary
    print("Starting indicatorvalues_async_main_1h_high")
    await create_indicator_tables(botdb_queue)
    print("awaiting create table_1h_high")
    table_created_event_1h_h.wait()
    await asyncio.sleep(10)
    print("awaiting await event_1h_high")
    global_start_timestamps, *_ = await get_start_timestamp_for_indicators()


    print("getting timestamp_1h_high")
    await process_indicator_series(botdb_queue, global_start_timestamps)
    #manage_process_thread = threading.Thread(target=manage_process, args=("indicatorvalues_main_1h_h", (botdb_queue), table_created_event_1h_h))
    #manage_process_thread.start()
    #manage_process_thread.join()

def indicatorvalues_main_1h_h(botdb_queue, table_created_event_1h_h):
    print("Indicator values running_1h_high")
    #setup_logging()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(indicatorvalues_async_main(botdb_queue, table_created_event_1h_h))
    botdb_queue.put(None) 
    loop.close()
    time.sleep(3)    
    print("Indicator values aren't what they used to be_1h_high")