
import sys
sys.path.insert(0, './lobotomiser/modules')
from collections import defaultdict
from multiprocessing import Queue, Process, Pool, cpu_count
import logging
from lobotomiser.dbqueue import mr_writer, get_db_path
from lobotomiser.botdbqueue import the_narrator, bot_db_path, determine_length_range
#from lobotomiser.indicators_15m_high import indicatorvalues_main_15m_c
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
import pyti.aroon as AROON
import pyti.accumulation_distribution as ACC
import pyti.detrended_price_oscillator as DPO
import pyti.double_exponential_moving_average as DEMA
from pyti.keltner_bands import upper_band as kelt_upper_band
from pyti.keltner_bands import center_band as kelt_center_band
from pyti.keltner_bands import lower_band as kelt_lower_band
import pyti.average_true_range as ATR
import pyti.bollinger_bands as BB
import pyti.chande_momentum_oscillator as CHANDE
import pyti.chaikin_money_flow as CMF
import pyti.commodity_channel_index as CCI
import pyti.money_flow_index as MFI
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
import pyti.ultimate_oscillator as UO
import pyti.volatility as VOLA
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from lobotomiser.modules.ta.volume import EaseOfMovementIndicator
#from lobotomiser.modules.ta.momentum import KAMAIndicator, kama
from lobotomiser.modules.ta.momentum import WilliamsRIndicator as WPR
from lobotomiser.modules.ta.trend import STCIndicator
from lobotomiser.modules.ta.volume import VolumeWeightedAveragePrice



DB_INDICATOR_VALUES = './lobotomiser/data/indicatorvalues.db'
timeframe = 'fiveminutebtc'
indicatortimeframes = 'standard_5m_close'




VALID_INTERVALS = ['5m', '15m', '1h', '4h', '12h', '1d']





standard_indicators = [


        'DPO_3_u_c', 'DPO_4_u_c', 'DPO_5_u_c', 'DPO_6_u_c', 'DPO_7_u_c', 'DPO_8_u_c',
    'DPO_9_u_c', 'DPO_10_u_c', 'DPO_11_u_c', 'DPO_12_u_c', 'DPO_13_u_c', 'DPO_14_u_c',
    'DPO_15_u_c', 'DPO_16_u_c', 'DPO_17_u_c', 'DPO_18_u_c', 'DPO_19_u_c', 'DPO_20_u_c',
    'DPO_21_u_c', 'DPO_22_u_c', 'DPO_23_u_c', 'DPO_24_u_c', 'DPO_25_u_c', 'DPO_26_u_c',
    'DPO_27_u_c', 'DPO_28_u_c', 'DPO_29_u_c', 'DPO_30_u_c', 'DPO_31_u_c', 'DPO_32_u_c',
    'DPO_33_u_c', 'DPO_34_u_c', 'DPO_35_u_c', 'DPO_36_u_c', 'DPO_37_u_c', 'DPO_38_u_c',
    'DPO_39_u_c', 'DPO_40_u_c', 'DPO_41_u_c', 'DPO_42_u_c', 'DPO_43_u_c', 'DPO_44_u_c',
    'DPO_45_u_c', 'DPO_46_u_c', 'DPO_47_u_c', 'DPO_48_u_c', 'DPO_49_u_c', 'DPO_50_u_c',
    'DPO_51_u_c', 'DPO_52_u_c', 'DPO_53_u_c', 'DPO_54_u_c', 'DPO_55_u_c', 'DPO_56_u_c',
    'DPO_57_u_c', 'DPO_58_u_c', 'DPO_59_u_c', 'DPO_60_u_c', 'DPO_61_u_c', 'DPO_62_u_c',
    'DPO_63_u_c', 'DPO_64_u_c', 'DPO_65_u_c', 'DPO_66_u_c', 'DPO_67_u_c', 'DPO_68_u_c',
    'DPO_69_u_c', 'DPO_70_u_c',

    'DEMA_3_u_c', 'DEMA_4_u_c', 'DEMA_5_u_c', 'DEMA_6_u_c', 'DEMA_7_u_c', 'DEMA_8_u_c',
    'DEMA_9_u_c', 'DEMA_10_u_c', 'DEMA_11_u_c', 'DEMA_12_u_c', 'DEMA_13_u_c', 'DEMA_14_u_c',
    'DEMA_15_u_c', 'DEMA_16_u_c', 'DEMA_17_u_c', 'DEMA_18_u_c', 'DEMA_19_u_c', 'DEMA_20_u_c',
    'DEMA_21_u_c', 'DEMA_22_u_c', 'DEMA_23_u_c', 'DEMA_24_u_c', 'DEMA_25_u_c', 'DEMA_26_u_c',
    'DEMA_27_u_c', 'DEMA_28_u_c', 'DEMA_29_u_c', 'DEMA_30_u_c', 'DEMA_31_u_c', 'DEMA_32_u_c',
    'DEMA_33_u_c', 'DEMA_34_u_c', 'DEMA_35_u_c', 'DEMA_36_u_c', 'DEMA_37_u_c', 'DEMA_38_u_c',
    'DEMA_39_u_c', 'DEMA_40_u_c', 'DEMA_41_u_c', 'DEMA_42_u_c', 'DEMA_43_u_c', 'DEMA_44_u_c',
    'DEMA_45_u_c', 'DEMA_46_u_c', 'DEMA_47_u_c', 'DEMA_48_u_c', 'DEMA_49_u_c', 'DEMA_50_u_c',
    'DEMA_51_u_c', 'DEMA_52_u_c', 'DEMA_53_u_c', 'DEMA_54_u_c', 'DEMA_55_u_c', 'DEMA_56_u_c',
    'DEMA_57_u_c', 'DEMA_58_u_c', 'DEMA_59_u_c', 'DEMA_60_u_c', 'DEMA_61_u_c', 'DEMA_62_u_c',
    'DEMA_63_u_c', 'DEMA_64_u_c', 'DEMA_65_u_c', 'DEMA_66_u_c', 'DEMA_67_u_c', 'DEMA_68_u_c',
    'DEMA_69_u_c', 'DEMA_70_u_c',

      'MAE_3_upper_u_c', 'MAE_4_upper_u_c', 'MAE_5_upper_u_c', 'MAE_6_upper_u_c', 'MAE_7_upper_u_c', 'MAE_8_upper_u_c',
    'MAE_9_upper_u_c', 'MAE_10_upper_u_c', 'MAE_11_upper_u_c', 'MAE_12_upper_u_c', 'MAE_13_upper_u_c', 'MAE_14_upper_u_c',
    'MAE_15_upper_u_c', 'MAE_16_upper_u_c', 'MAE_17_upper_u_c', 'MAE_18_upper_u_c', 'MAE_19_upper_u_c', 'MAE_20_upper_u_c',
    'MAE_21_upper_u_c', 'MAE_22_upper_u_c', 'MAE_23_upper_u_c', 'MAE_24_upper_u_c', 'MAE_25_upper_u_c', 'MAE_26_upper_u_c',
    'MAE_27_upper_u_c', 'MAE_28_upper_u_c', 'MAE_29_upper_u_c', 'MAE_30_upper_u_c', 'MAE_31_upper_u_c', 'MAE_32_upper_u_c',
    'MAE_33_upper_u_c', 'MAE_34_upper_u_c', 'MAE_35_upper_u_c', 'MAE_36_upper_u_c', 'MAE_37_upper_u_c', 'MAE_38_upper_u_c',
    'MAE_39_upper_u_c', 'MAE_40_upper_u_c', 'MAE_41_upper_u_c', 'MAE_42_upper_u_c', 'MAE_43_upper_u_c', 'MAE_44_upper_u_c',
    'MAE_45_upper_u_c', 'MAE_46_upper_u_c', 'MAE_47_upper_u_c', 'MAE_48_upper_u_c', 'MAE_49_upper_u_c', 'MAE_50_upper_u_c',
    'MAE_51_upper_u_c', 'MAE_52_upper_u_c', 'MAE_53_upper_u_c', 'MAE_54_upper_u_c', 'MAE_55_upper_u_c', 'MAE_56_upper_u_c',
    'MAE_57_upper_u_c', 'MAE_58_upper_u_c', 'MAE_59_upper_u_c', 'MAE_60_upper_u_c', 'MAE_61_upper_u_c', 'MAE_62_upper_u_c',
    'MAE_63_upper_u_c', 'MAE_64_upper_u_c', 'MAE_65_upper_u_c', 'MAE_66_upper_u_c', 'MAE_67_upper_u_c', 'MAE_68_upper_u_c',
    'MAE_69_upper_u_c', 'MAE_70_upper_u_c',

        'MAE_3_center_u_c', 'MAE_4_center_u_c', 'MAE_5_center_u_c', 'MAE_6_center_u_c', 'MAE_7_center_u_c', 'MAE_8_center_u_c',
    'MAE_9_center_u_c', 'MAE_10_center_u_c', 'MAE_11_center_u_c', 'MAE_12_center_u_c', 'MAE_13_center_u_c', 'MAE_14_center_u_c',
    'MAE_15_center_u_c', 'MAE_16_center_u_c', 'MAE_17_center_u_c', 'MAE_18_center_u_c', 'MAE_19_center_u_c', 'MAE_20_center_u_c',
    'MAE_21_center_u_c', 'MAE_22_center_u_c', 'MAE_23_center_u_c', 'MAE_24_center_u_c', 'MAE_25_center_u_c', 'MAE_26_center_u_c',
    'MAE_27_center_u_c', 'MAE_28_center_u_c', 'MAE_29_center_u_c', 'MAE_30_center_u_c', 'MAE_31_center_u_c', 'MAE_32_center_u_c',
    'MAE_33_center_u_c', 'MAE_34_center_u_c', 'MAE_35_center_u_c', 'MAE_36_center_u_c', 'MAE_37_center_u_c', 'MAE_38_center_u_c',
    'MAE_39_center_u_c', 'MAE_40_center_u_c', 'MAE_41_center_u_c', 'MAE_42_center_u_c', 'MAE_43_center_u_c', 'MAE_44_center_u_c',
    'MAE_45_center_u_c', 'MAE_46_center_u_c', 'MAE_47_center_u_c', 'MAE_48_center_u_c', 'MAE_49_center_u_c', 'MAE_50_center_u_c',
    'MAE_51_center_u_c', 'MAE_52_center_u_c', 'MAE_53_center_u_c', 'MAE_54_center_u_c', 'MAE_55_center_u_c', 'MAE_56_center_u_c',
    'MAE_57_center_u_c', 'MAE_58_center_u_c', 'MAE_59_center_u_c', 'MAE_60_center_u_c', 'MAE_61_center_u_c', 'MAE_62_center_u_c',
    'MAE_63_center_u_c', 'MAE_64_center_u_c', 'MAE_65_center_u_c', 'MAE_66_center_u_c', 'MAE_67_center_u_c', 'MAE_68_center_u_c',
    'MAE_69_center_u_c', 'MAE_70_center_u_c',

    'MAE_3_lower_u_c', 'MAE_4_lower_u_c', 'MAE_5_lower_u_c', 'MAE_6_lower_u_c', 'MAE_7_lower_u_c', 'MAE_8_lower_u_c',
    'MAE_9_lower_u_c', 'MAE_10_lower_u_c', 'MAE_11_lower_u_c', 'MAE_12_lower_u_c', 'MAE_13_lower_u_c', 'MAE_14_lower_u_c',
    'MAE_15_lower_u_c', 'MAE_16_lower_u_c', 'MAE_17_lower_u_c', 'MAE_18_lower_u_c', 'MAE_19_lower_u_c', 'MAE_20_lower_u_c',
    'MAE_21_lower_u_c', 'MAE_22_lower_u_c', 'MAE_23_lower_u_c', 'MAE_24_lower_u_c', 'MAE_25_lower_u_c', 'MAE_26_lower_u_c',
    'MAE_27_lower_u_c', 'MAE_28_lower_u_c', 'MAE_29_lower_u_c', 'MAE_30_lower_u_c', 'MAE_31_lower_u_c', 'MAE_32_lower_u_c',
    'MAE_33_lower_u_c', 'MAE_34_lower_u_c', 'MAE_35_lower_u_c', 'MAE_36_lower_u_c', 'MAE_37_lower_u_c', 'MAE_38_lower_u_c',
    'MAE_39_lower_u_c', 'MAE_40_lower_u_c', 'MAE_41_lower_u_c', 'MAE_42_lower_u_c', 'MAE_43_lower_u_c', 'MAE_44_lower_u_c',
    'MAE_45_lower_u_c', 'MAE_46_lower_u_c', 'MAE_47_lower_u_c', 'MAE_48_lower_u_c', 'MAE_49_lower_u_c', 'MAE_50_lower_u_c',
    'MAE_51_lower_u_c', 'MAE_52_lower_u_c', 'MAE_53_lower_u_c', 'MAE_54_lower_u_c', 'MAE_55_lower_u_c', 'MAE_56_lower_u_c',
    'MAE_57_lower_u_c', 'MAE_58_lower_u_c', 'MAE_59_lower_u_c', 'MAE_60_lower_u_c', 'MAE_61_lower_u_c', 'MAE_62_lower_u_c',
    'MAE_63_lower_u_c', 'MAE_64_lower_u_c', 'MAE_65_lower_u_c', 'MAE_66_lower_u_c', 'MAE_67_lower_u_c', 'MAE_68_lower_u_c',
    'MAE_69_lower_u_c', 'MAE_70_lower_u_c',


    'MOM_3_u_c', 'MOM_4_u_c', 'MOM_5_u_c', 'MOM_6_u_c', 'MOM_7_u_c', 'MOM_8_u_c',
    'MOM_9_u_c', 'MOM_10_u_c', 'MOM_11_u_c', 'MOM_12_u_c', 'MOM_13_u_c', 'MOM_14_u_c',
    'MOM_15_u_c', 'MOM_16_u_c', 'MOM_17_u_c', 'MOM_18_u_c', 'MOM_19_u_c', 'MOM_20_u_c',
    'MOM_21_u_c', 'MOM_22_u_c', 'MOM_23_u_c', 'MOM_24_u_c', 'MOM_25_u_c', 'MOM_26_u_c',
    'MOM_27_u_c', 'MOM_28_u_c', 'MOM_29_u_c', 'MOM_30_u_c', 'MOM_31_u_c', 'MOM_32_u_c',
    'MOM_33_u_c', 'MOM_34_u_c', 'MOM_35_u_c', 'MOM_36_u_c', 'MOM_37_u_c', 'MOM_38_u_c',
    'MOM_39_u_c', 'MOM_40_u_c', 'MOM_41_u_c', 'MOM_42_u_c', 'MOM_43_u_c', 'MOM_44_u_c',
    'MOM_45_u_c', 'MOM_46_u_c', 'MOM_47_u_c', 'MOM_48_u_c', 'MOM_49_u_c', 'MOM_50_u_c',
    'MOM_51_u_c', 'MOM_52_u_c', 'MOM_53_u_c', 'MOM_54_u_c', 'MOM_55_u_c', 'MOM_56_u_c',
    'MOM_57_u_c', 'MOM_58_u_c', 'MOM_59_u_c', 'MOM_60_u_c', 'MOM_61_u_c', 'MOM_62_u_c',
    'MOM_63_u_c', 'MOM_64_u_c', 'MOM_65_u_c', 'MOM_66_u_c', 'MOM_67_u_c', 'MOM_68_u_c',
    'MOM_69_u_c', 'MOM_70_u_c',

    'MAE_3_u_c', 'MAE_4_u_c', 'MAE_5_u_c', 'MAE_6_u_c', 'MAE_7_u_c', 'MAE_8_u_c',
    'MAE_9_u_c', 'MAE_10_u_c', 'MAE_11_u_c', 'MAE_12_u_c', 'MAE_13_u_c', 'MAE_14_u_c',
    'MAE_15_u_c', 'MAE_16_u_c', 'MAE_17_u_c', 'MAE_18_u_c', 'MAE_19_u_c', 'MAE_20_u_c',
    'MAE_21_u_c', 'MAE_22_u_c', 'MAE_23_u_c', 'MAE_24_u_c', 'MAE_25_u_c', 'MAE_26_u_c',
    'MAE_27_u_c', 'MAE_28_u_c', 'MAE_29_u_c', 'MAE_30_u_c', 'MAE_31_u_c', 'MAE_32_u_c',
    'MAE_33_u_c', 'MAE_34_u_c', 'MAE_35_u_c', 'MAE_36_u_c', 'MAE_37_u_c', 'MAE_38_u_c',
    'MAE_39_u_c', 'MAE_40_u_c', 'MAE_41_u_c', 'MAE_42_u_c', 'MAE_43_u_c', 'MAE_44_u_c',
    'MAE_45_u_c', 'MAE_46_u_c', 'MAE_47_u_c', 'MAE_48_u_c', 'MAE_49_u_c', 'MAE_50_u_c',
    'MAE_51_u_c', 'MAE_52_u_c', 'MAE_53_u_c', 'MAE_54_u_c', 'MAE_55_u_c', 'MAE_56_u_c',
    'MAE_57_u_c', 'MAE_58_u_c', 'MAE_59_u_c', 'MAE_60_u_c', 'MAE_61_u_c', 'MAE_62_u_c',
    'MAE_63_u_c', 'MAE_64_u_c', 'MAE_65_u_c', 'MAE_66_u_c', 'MAE_67_u_c', 'MAE_68_u_c',
    'MAE_69_u_c', 'MAE_70_u_c',

    'SD_3_u_c', 'SD_4_u_c', 'SD_5_u_c', 'SD_6_u_c', 'SD_7_u_c', 'SD_8_u_c',
    'SD_9_u_c', 'SD_10_u_c', 'SD_11_u_c', 'SD_12_u_c', 'SD_13_u_c', 'SD_14_u_c',
    'SD_15_u_c', 'SD_16_u_c', 'SD_17_u_c', 'SD_18_u_c', 'SD_19_u_c', 'SD_20_u_c',
    'SD_21_u_c', 'SD_22_u_c', 'SD_23_u_c', 'SD_24_u_c', 'SD_25_u_c', 'SD_26_u_c',
    'SD_27_u_c', 'SD_28_u_c', 'SD_29_u_c', 'SD_30_u_c', 'SD_31_u_c', 'SD_32_u_c',
    'SD_33_u_c', 'SD_34_u_c', 'SD_35_u_c', 'SD_36_u_c', 'SD_37_u_c', 'SD_38_u_c',
    'SD_39_u_c', 'SD_40_u_c', 'SD_41_u_c', 'SD_42_u_c', 'SD_43_u_c', 'SD_44_u_c',
    'SD_45_u_c', 'SD_46_u_c', 'SD_47_u_c', 'SD_48_u_c', 'SD_49_u_c', 'SD_50_u_c',
    'SD_51_u_c', 'SD_52_u_c', 'SD_53_u_c', 'SD_54_u_c', 'SD_55_u_c', 'SD_56_u_c',
    'SD_57_u_c', 'SD_58_u_c', 'SD_59_u_c', 'SD_60_u_c', 'SD_61_u_c', 'SD_62_u_c',
    'SD_63_u_c', 'SD_64_u_c', 'SD_65_u_c', 'SD_66_u_c', 'SD_67_u_c', 'SD_68_u_c',
    'SD_69_u_c', 'SD_70_u_c',

    'SV_3_u_c', 'SV_4_u_c', 'SV_5_u_c', 'SV_6_u_c', 'SV_7_u_c', 'SV_8_u_c',
    'SV_9_u_c', 'SV_10_u_c', 'SV_11_u_c', 'SV_12_u_c', 'SV_13_u_c', 'SV_14_u_c',
    'SV_15_u_c', 'SV_16_u_c', 'SV_17_u_c', 'SV_18_u_c', 'SV_19_u_c', 'SV_20_u_c',
    'SV_21_u_c', 'SV_22_u_c', 'SV_23_u_c', 'SV_24_u_c', 'SV_25_u_c', 'SV_26_u_c',
    'SV_27_u_c', 'SV_28_u_c', 'SV_29_u_c', 'SV_30_u_c', 'SV_31_u_c', 'SV_32_u_c',
    'SV_33_u_c', 'SV_34_u_c', 'SV_35_u_c', 'SV_36_u_c', 'SV_37_u_c', 'SV_38_u_c',
    'SV_39_u_c', 'SV_40_u_c', 'SV_41_u_c', 'SV_42_u_c', 'SV_43_u_c', 'SV_44_u_c',
    'SV_45_u_c', 'SV_46_u_c', 'SV_47_u_c', 'SV_48_u_c', 'SV_49_u_c', 'SV_50_u_c',
    'SV_51_u_c', 'SV_52_u_c', 'SV_53_u_c', 'SV_54_u_c', 'SV_55_u_c', 'SV_56_u_c',
    'SV_57_u_c', 'SV_58_u_c', 'SV_59_u_c', 'SV_60_u_c', 'SV_61_u_c', 'SV_62_u_c',
    'SV_63_u_c', 'SV_64_u_c', 'SV_65_u_c', 'SV_66_u_c', 'SV_67_u_c', 'SV_68_u_c',
    'SV_69_u_c', 'SV_70_u_c',

    'TMA_3_u_c', 'TMA_4_u_c', 'TMA_5_u_c', 'TMA_6_u_c', 'TMA_7_u_c', 'TMA_8_u_c',
    'TMA_9_u_c', 'TMA_10_u_c', 'TMA_11_u_c', 'TMA_12_u_c', 'TMA_13_u_c', 'TMA_14_u_c',
    'TMA_15_u_c', 'TMA_16_u_c', 'TMA_17_u_c', 'TMA_18_u_c', 'TMA_19_u_c', 'TMA_20_u_c',
    'TMA_21_u_c', 'TMA_22_u_c', 'TMA_23_u_c', 'TMA_24_u_c', 'TMA_25_u_c', 'TMA_26_u_c',
    'TMA_27_u_c', 'TMA_28_u_c', 'TMA_29_u_c', 'TMA_30_u_c', 'TMA_31_u_c', 'TMA_32_u_c',
    'TMA_33_u_c', 'TMA_34_u_c', 'TMA_35_u_c', 'TMA_36_u_c', 'TMA_37_u_c', 'TMA_38_u_c',
    'TMA_39_u_c', 'TMA_40_u_c', 'TMA_41_u_c', 'TMA_42_u_c', 'TMA_43_u_c', 'TMA_44_u_c',
    'TMA_45_u_c', 'TMA_46_u_c', 'TMA_47_u_c', 'TMA_48_u_c', 'TMA_49_u_c', 'TMA_50_u_c',
    'TMA_51_u_c', 'TMA_52_u_c', 'TMA_53_u_c', 'TMA_54_u_c', 'TMA_55_u_c', 'TMA_56_u_c',
    'TMA_57_u_c', 'TMA_58_u_c', 'TMA_59_u_c', 'TMA_60_u_c', 'TMA_61_u_c', 'TMA_62_u_c',
    'TMA_63_u_c', 'TMA_64_u_c', 'TMA_65_u_c', 'TMA_66_u_c', 'TMA_67_u_c', 'TMA_68_u_c',
    'TMA_69_u_c', 'TMA_70_u_c',

    'TEMA_3_u_c', 'TEMA_4_u_c', 'TEMA_5_u_c', 'TEMA_6_u_c', 'TEMA_7_u_c', 'TEMA_8_u_c',
    'TEMA_9_u_c', 'TEMA_10_u_c', 'TEMA_11_u_c', 'TEMA_12_u_c', 'TEMA_13_u_c', 'TEMA_14_u_c',
    'TEMA_15_u_c', 'TEMA_16_u_c', 'TEMA_17_u_c', 'TEMA_18_u_c', 'TEMA_19_u_c', 'TEMA_20_u_c',
    'TEMA_21_u_c', 'TEMA_22_u_c', 'TEMA_23_u_c', 'TEMA_24_u_c', 'TEMA_25_u_c', 'TEMA_26_u_c',
    'TEMA_27_u_c', 'TEMA_28_u_c', 'TEMA_29_u_c', 'TEMA_30_u_c', 'TEMA_31_u_c', 'TEMA_32_u_c',
    'TEMA_33_u_c', 'TEMA_34_u_c', 'TEMA_35_u_c', 'TEMA_36_u_c', 'TEMA_37_u_c', 'TEMA_38_u_c',
    'TEMA_39_u_c', 'TEMA_40_u_c', 'TEMA_41_u_c', 'TEMA_42_u_c', 'TEMA_43_u_c', 'TEMA_44_u_c',
    'TEMA_45_u_c', 'TEMA_46_u_c', 'TEMA_47_u_c', 'TEMA_48_u_c', 'TEMA_49_u_c', 'TEMA_50_u_c',
    'TEMA_51_u_c', 'TEMA_52_u_c', 'TEMA_53_u_c', 'TEMA_54_u_c', 'TEMA_55_u_c', 'TEMA_56_u_c',
    'TEMA_57_u_c', 'TEMA_58_u_c', 'TEMA_59_u_c', 'TEMA_60_u_c', 'TEMA_61_u_c', 'TEMA_62_u_c',
    'TEMA_63_u_c', 'TEMA_64_u_c', 'TEMA_65_u_c', 'TEMA_66_u_c', 'TEMA_67_u_c', 'TEMA_68_u_c',
    'TEMA_69_u_c', 'TEMA_70_u_c',
    
    'VOLA_3_u_c', 'VOLA_4_u_c', 'VOLA_5_u_c', 'VOLA_6_u_c', 'VOLA_7_u_c', 'VOLA_8_u_c',
    'VOLA_9_u_c', 'VOLA_10_u_c', 'VOLA_11_u_c', 'VOLA_12_u_c', 'VOLA_13_u_c', 'VOLA_14_u_c',
    'VOLA_15_u_c', 'VOLA_16_u_c', 'VOLA_17_u_c', 'VOLA_18_u_c', 'VOLA_19_u_c', 'VOLA_20_u_c',
    'VOLA_21_u_c', 'VOLA_22_u_c', 'VOLA_23_u_c', 'VOLA_24_u_c', 'VOLA_25_u_c', 'VOLA_26_u_c',
    'VOLA_27_u_c', 'VOLA_28_u_c', 'VOLA_29_u_c', 'VOLA_30_u_c', 'VOLA_31_u_c', 'VOLA_32_u_c',
    'VOLA_33_u_c', 'VOLA_34_u_c', 'VOLA_35_u_c', 'VOLA_36_u_c', 'VOLA_37_u_c', 'VOLA_38_u_c',
    'VOLA_39_u_c', 'VOLA_40_u_c', 'VOLA_41_u_c', 'VOLA_42_u_c', 'VOLA_43_u_c', 'VOLA_44_u_c',
    'VOLA_45_u_c', 'VOLA_46_u_c', 'VOLA_47_u_c', 'VOLA_48_u_c', 'VOLA_49_u_c', 'VOLA_50_u_c',
    'VOLA_51_u_c', 'VOLA_52_u_c', 'VOLA_53_u_c', 'VOLA_54_u_c', 'VOLA_55_u_c', 'VOLA_56_u_c',
    'VOLA_57_u_c', 'VOLA_58_u_c', 'VOLA_59_u_c', 'VOLA_60_u_c', 'VOLA_61_u_c', 'VOLA_62_u_c',
    'VOLA_63_u_c', 'VOLA_64_u_c', 'VOLA_65_u_c', 'VOLA_66_u_c', 'VOLA_67_u_c', 'VOLA_68_u_c',
    'VOLA_69_u_c', 'VOLA_70_u_c',

    'ATR_4_u_c', 'ATR_5_u_c', 'ATR_6_u_c', 'ATR_7_u_c', 'ATR_8_u_c', 'ATR_9_u_c', 'ATR_10_u_c',
    'ATR_11_u_c', 'ATR_12_u_c', 'ATR_13_u_c', 'ATR_14_u_c', 'ATR_15_u_c', 'ATR_16_u_c',
    'ATR_17_u_c', 'ATR_18_u_c', 'ATR_19_u_c', 'ATR_20_u_c', 'ATR_21_u_c', 'ATR_22_u_c',
    'ATR_23_u_c', 'ATR_24_u_c', 'ATR_25_u_c', 'ATR_26_u_c', 'ATR_27_u_c', 'ATR_28_u_c',
    'ATR_29_u_c', 'ATR_30_u_c', 'ATR_31_u_c', 'ATR_32_u_c', 'ATR_33_u_c', 'ATR_34_u_c',
    'ATR_35_u_c', 'ATR_36_u_c', 'ATR_37_u_c', 'ATR_38_u_c', 'ATR_39_u_c', 'ATR_40_u_c',
    'ATR_41_u_c', 'ATR_42_u_c', 'ATR_43_u_c', 'ATR_44_u_c', 'ATR_45_u_c', 'ATR_46_u_c',
    'ATR_47_u_c', 'ATR_48_u_c', 'ATR_50_u_c', 'ATR_51_u_c', 'ATR_52_u_c',
    'ATR_53_u_c', 'ATR_54_u_c', 'ATR_55_u_c', 'ATR_56_u_c', 'ATR_57_u_c', 'ATR_58_u_c',
    'ATR_59_u_c', 'ATR_60_u_c', 'ATR_61_u_c', 'ATR_62_u_c', 'ATR_63_u_c', 'ATR_64_u_c',
    'ATR_65_u_c', 'ATR_66_u_c', 'ATR_67_u_c', 'ATR_68_u_c', 'ATR_69_u_c', 'ATR_70_u_c',

   
    'BB_l_14_u_c', 'BB_l_21_u_c', 'BB_l_24_u_c', 'BB_l_25_u_c', 'BB_l_26_u_c', 'BB_l_27_u_c', 'BB_l_28_u_c', 'BB_l_36_u_c', 'BB_l_44_u_c',
    'BB_l_48_u_c', 'BB_l_49_u_c', 'BB_l_5_u_c', 'BB_m_23_u_c', 'BB_m_45_u_c', 'BB_m_9_u_c', 'BB_u_12_u_c', 'BB_u_3_u_c',
    'BB_u_37_u_c', 'BB_u_38_u_c', 'BB_u_39_u_c', 'BB_u_44_u_c', 'BB_u_45_u_c', 'BB_u_46_u_c', 'BB_u_49_u_c',  
    'CHANDE_12_u_c', 'CHANDE_18_u_c', 'CHANDE_19_u_c', 'CHANDE_27_u_c', 'CHANDE_28_u_c', 'CHANDE_31_u_c', 'CHANDE_32_u_c', 'CCI_7_u_c', 
    'CHANDE_33_u_c', 'CHANDE_36_u_c', 'CHANDE_44_u_c', 'CHANDE_48_u_c', 'CHANDE_49_u_c', 'CHANDE_5_u_c', 'CHANDE_7_u_c',
    'CHANDE_8_u_c', 'CMF_31_u_c', 'CMF_33_u_c', 'CMF_34_u_c', 'MACD_3_u_c', 'MFI_19_u_c', 'MFI_49_u_c',
    'MFI_6_u_c', 'PB_13_u_c', 'PB_3_u_c', 'PB_4_u_c', 'PB_5_u_c', 'PB_8_u_c', 'ROC_11_u_c', 'ROC_12_u_c',
    'ROC_13_u_c', 'ROC_21_u_c', 'ROC_22_u_c', 'ROC_23_u_c', 'ROC_25_u_c', 'ROC_27_u_c', 'ROC_30_u_c', 'ROC_45_u_c', 'ROC_7_u_c',
    'RSI_12_u_c', 'RSI_13_u_c', 'RSI_14_u_c', 'RSI_15_u_c', 'RSI_21_u_c', 'RSI_23_u_c', 'RSI_25_u_c', 'RSI_28_u_c', 'RSI_3_u_c',
    'RSI_34_u_c', 'RSI_35_u_c', 'RSI_4_u_c', 'RSI_49_u_c', 'RSI_5_u_c', 'RSI_6_u_c', 'RSI_7_u_c', 'RSI_9_u_c', 'SEOM_8_u_c',
    'VO_3_u_c',

    'OBV_3_u_c', 'OBV_4_u_c', 'OBV_5_u_c', 'OBV_6_u_c', 'OBV_7_u_c', 'OBV_8_u_c',
    'OBV_9_u_c', 'OBV_10_u_c', 'OBV_11_u_c', 'OBV_12_u_c', 'OBV_13_u_c', 'OBV_14_u_c',
    'OBV_15_u_c', 'OBV_16_u_c', 'OBV_17_u_c', 'OBV_18_u_c', 'OBV_19_u_c', 'OBV_20_u_c',
    'OBV_21_u_c', 'OBV_22_u_c', 'OBV_23_u_c', 'OBV_24_u_c', 'OBV_25_u_c', 'OBV_26_u_c',
    'OBV_27_u_c', 'OBV_28_u_c', 'OBV_29_u_c', 'OBV_30_u_c', 'OBV_31_u_c', 'OBV_32_u_c',
    'OBV_33_u_c', 'OBV_34_u_c', 'OBV_35_u_c', 'OBV_36_u_c', 'OBV_37_u_c', 'OBV_38_u_c',
    'OBV_39_u_c', 'OBV_40_u_c', 'OBV_41_u_c', 'OBV_42_u_c', 'OBV_43_u_c', 'OBV_44_u_c',
    'OBV_45_u_c', 'OBV_46_u_c', 'OBV_47_u_c', 'OBV_48_u_c', 'OBV_49_u_c', 'OBV_50_u_c',
    'OBV_51_u_c', 'OBV_52_u_c', 'OBV_53_u_c', 'OBV_54_u_c', 'OBV_55_u_c', 'OBV_56_u_c',
    'OBV_57_u_c', 'OBV_58_u_c', 'OBV_59_u_c', 'OBV_60_u_c', 'OBV_61_u_c', 'OBV_62_u_c',
    'OBV_63_u_c', 'OBV_64_u_c', 'OBV_65_u_c', 'OBV_66_u_c', 'OBV_67_u_c', 'OBV_68_u_c',
    'OBV_69_u_c', 'OBV_70_u_c'

    
  
]
add_dynamism_to_series = [

    'ACC_3_d_c', 'ACC_4_d_c', 'ACC_5_d_c', 'ACC_6_d_c', 'ACC_7_d_c', 'ACC_8_d_c',
    'ACC_9_d_c', 'ACC_10_d_c', 'ACC_11_d_c', 'ACC_12_d_c', 'ACC_13_d_c', 'ACC_14_d_c',
    'ACC_15_d_c', 'ACC_16_d_c', 'ACC_17_d_c', 'ACC_18_d_c', 'ACC_19_d_c', 'ACC_20_d_c',
    'ACC_21_d_c', 'ACC_22_d_c', 'ACC_23_d_c', 'ACC_24_d_c', 'ACC_25_d_c', 'ACC_26_d_c',
    'ACC_27_d_c', 'ACC_28_d_c', 'ACC_29_d_c', 'ACC_30_d_c', 'ACC_31_d_c', 'ACC_32_d_c',
    'ACC_33_d_c', 'ACC_34_d_c', 'ACC_35_d_c', 'ACC_36_d_c', 'ACC_37_d_c', 'ACC_38_d_c',
    'ACC_39_d_c', 'ACC_40_d_c', 'ACC_41_d_c', 'ACC_42_d_c', 'ACC_43_d_c', 'ACC_44_d_c',
    'ACC_45_d_c', 'ACC_46_d_c', 'ACC_47_d_c', 'ACC_48_d_c', 'ACC_49_d_c', 'ACC_50_d_c',
    'ACC_51_d_c', 'ACC_52_d_c', 'ACC_53_d_c', 'ACC_54_d_c', 'ACC_55_d_c', 'ACC_56_d_c',
    'ACC_57_d_c', 'ACC_58_d_c', 'ACC_59_d_c', 'ACC_60_d_c', 'ACC_61_d_c', 'ACC_62_d_c',
    'ACC_63_d_c', 'ACC_64_d_c', 'ACC_65_d_c', 'ACC_66_d_c', 'ACC_67_d_c', 'ACC_68_d_c',
    'ACC_69_d_c', 'ACC_70_d_c',

        'DPO_3_d_c', 'DPO_4_d_c', 'DPO_5_d_c', 'DPO_6_d_c', 'DPO_7_d_c', 'DPO_8_d_c',
    'DPO_9_d_c', 'DPO_10_d_c', 'DPO_11_d_c', 'DPO_12_d_c', 'DPO_13_d_c', 'DPO_14_d_c',
    'DPO_15_d_c', 'DPO_16_d_c', 'DPO_17_d_c', 'DPO_18_d_c', 'DPO_19_d_c', 'DPO_20_d_c',
    'DPO_21_d_c', 'DPO_22_d_c', 'DPO_23_d_c', 'DPO_24_d_c', 'DPO_25_d_c', 'DPO_26_d_c',
    'DPO_27_d_c', 'DPO_28_d_c', 'DPO_29_d_c', 'DPO_30_d_c', 'DPO_31_d_c', 'DPO_32_d_c',
    'DPO_33_d_c', 'DPO_34_d_c', 'DPO_35_d_c', 'DPO_36_d_c', 'DPO_37_d_c', 'DPO_38_d_c',
    'DPO_39_d_c', 'DPO_40_d_c', 'DPO_41_d_c', 'DPO_42_d_c', 'DPO_43_d_c', 'DPO_44_d_c',
    'DPO_45_d_c', 'DPO_46_d_c', 'DPO_47_d_c', 'DPO_48_d_c', 'DPO_49_d_c', 'DPO_50_d_c',
    'DPO_51_d_c', 'DPO_52_d_c', 'DPO_53_d_c', 'DPO_54_d_c', 'DPO_55_d_c', 'DPO_56_d_c',
    'DPO_57_d_c', 'DPO_58_d_c', 'DPO_59_d_c', 'DPO_60_d_c', 'DPO_61_d_c', 'DPO_62_d_c',
    'DPO_63_d_c', 'DPO_64_d_c', 'DPO_65_d_c', 'DPO_66_d_c', 'DPO_67_d_c', 'DPO_68_d_c',
    'DPO_69_d_c', 'DPO_70_d_c',

            'DEMA_3_d_c', 'DEMA_4_d_c', 'DEMA_5_d_c', 'DEMA_6_d_c', 'DEMA_7_d_c', 'DEMA_8_d_c',
    'DEMA_9_d_c', 'DEMA_10_d_c', 'DEMA_11_d_c', 'DEMA_12_d_c', 'DEMA_13_d_c', 'DEMA_14_d_c',
    'DEMA_15_d_c', 'DEMA_16_d_c', 'DEMA_17_d_c', 'DEMA_18_d_c', 'DEMA_19_d_c', 'DEMA_20_d_c',
    'DEMA_21_d_c', 'DEMA_22_d_c', 'DEMA_23_d_c', 'DEMA_24_d_c', 'DEMA_25_d_c', 'DEMA_26_d_c',
    'DEMA_27_d_c', 'DEMA_28_d_c', 'DEMA_29_d_c', 'DEMA_30_d_c', 'DEMA_31_d_c', 'DEMA_32_d_c',
    'DEMA_33_d_c', 'DEMA_34_d_c', 'DEMA_35_d_c', 'DEMA_36_d_c', 'DEMA_37_d_c', 'DEMA_38_d_c',
    'DEMA_39_d_c', 'DEMA_40_d_c', 'DEMA_41_d_c', 'DEMA_42_d_c', 'DEMA_43_d_c', 'DEMA_44_d_c',
    'DEMA_45_d_c', 'DEMA_46_d_c', 'DEMA_47_d_c', 'DEMA_48_d_c', 'DEMA_49_d_c', 'DEMA_50_d_c',
    'DEMA_51_d_c', 'DEMA_52_d_c', 'DEMA_53_d_c', 'DEMA_54_d_c', 'DEMA_55_d_c', 'DEMA_56_d_c',
    'DEMA_57_d_c', 'DEMA_58_d_c', 'DEMA_59_d_c', 'DEMA_60_d_c', 'DEMA_61_d_c', 'DEMA_62_d_c',
    'DEMA_63_d_c', 'DEMA_64_d_c', 'DEMA_65_d_c', 'DEMA_66_d_c', 'DEMA_67_d_c', 'DEMA_68_d_c',
    'DEMA_69_d_c', 'DEMA_70_d_c',

    'ATR_4_d_c', 'ATR_5_d_c', 'ATR_6_d_c', 'ATR_7_d_c', 'ATR_8_d_c', 'ATR_9_d_c', 'ATR_10_d_c',
      'ATR_13_d_c',  'ATR_15_d_c', 'ATR_16_d_c',
    'ATR_17_d_c', 'ATR_18_d_c', 'ATR_19_d_c', 'ATR_20_d_c', 'ATR_21_d_c', 'ATR_22_d_c',
    'ATR_23_d_c', 'ATR_24_d_c', 'ATR_25_d_c', 'ATR_27_d_c', 'ATR_28_d_c',
     'ATR_30_d_c',  'ATR_32_d_c', 'ATR_33_d_c', 'ATR_34_d_c',
    'ATR_35_d_c', 'ATR_36_d_c', 'ATR_37_d_c', 'ATR_38_d_c', 'ATR_39_d_c', 'ATR_40_d_c',
    'ATR_41_d_c', 'ATR_42_d_c', 'ATR_43_d_c', 'ATR_45_d_c', 'ATR_46_d_c',
    'ATR_47_d_c', 'ATR_48_d_c', 'ATR_49_d_c', 'ATR_50_d_c', 'ATR_51_d_c', 'ATR_52_d_c',
    'ATR_53_d_c', 'ATR_54_d_c', 'ATR_55_d_c', 'ATR_56_d_c', 'ATR_57_d_c', 'ATR_58_d_c',
    'ATR_59_d_c', 'ATR_60_d_c', 'ATR_61_d_c', 'ATR_62_d_c', 'ATR_63_d_c', 'ATR_64_d_c',
    'ATR_65_d_c', 'ATR_66_d_c', 'ATR_67_d_c', 'ATR_68_d_c', 'ATR_69_d_c', 'ATR_70_d_c',

    'AROON_DOWN_21_d_c', 'AROON_DOWN_24_d_c', 'AROON_DOWN_43_d_c', 'AROON_DOWN_48_d_c', 'AROON_DOWN_49_d_c',
    'AROON_DOWN_8_d_c', 'AROON_UP_13_d_c', 'AROON_UP_17_d_c', 'AROON_UP_32_d_c', 'AROON_UP_48_d_c',
    'AROON_UP_7_d_c', 
    'CHANDE_16_d_c', 'CHANDE_17_d_c', 'CHANDE_19_d_c', 'CHANDE_26_d_c', 'CHANDE_38_d_c',
    'CHANDE_6_d_c', 'CHANDE_7_d_c', 'CMF_21_d_c', 'CMF_22_d_c', 'CMF_31_d_c', 'EOM_3_d_c', 'EOM_4_d_c', 'EOM_46_d_c',
    'MACD_3_d_c', 'MFI_27_d_c', 'MFI_45_d_c',
    'ROC_14_d_c', 'ROC_18_d_c', 'ROC_20_d_c', 'ROC_24_d_c', 'ROC_36_d_c', 'ROC_4_d_c', 'ROC_40_d_c',
    'ROC_7_d_c', 'RSI_15_d_c', 'RSI_24_d_c', 'RSI_26_d_c', 'RSI_3_d_c', 'RSI_31_d_c', 'RSI_33_d_c', 'RSI_39_d_c',
    'RSI_4_d_c', 'RSI_47_d_c', 'RSI_48_d_c', 'RSI_49_d_c', 'RSI_5_d_c', 'RSI_9_d_c', 'SCHAFF_13_d_c', 'SCHAFF_40_d_c',
    'VO_10_d_c', 'VO_11_d_c', 'VO_12_d_c', 'VO_13_d_c', 'VO_19_d_c', 'VO_20_d_c', 'VO_21_d_c', 'VO_25_d_c',
    'VWAP_30_d_c', 'VWAP_34_d_c', 'VWAP_49_d_c',

    'OBV_3_d_c', 'OBV_4_d_c', 'OBV_5_d_c', 'OBV_6_d_c', 'OBV_7_d_c', 'OBV_8_d_c',
    'OBV_9_d_c', 'OBV_10_d_c', 'OBV_11_d_c', 'OBV_12_d_c', 'OBV_13_d_c', 'OBV_14_d_c',
    'OBV_15_d_c', 'OBV_16_d_c', 'OBV_17_d_c', 'OBV_18_d_c', 'OBV_19_d_c', 'OBV_20_d_c',
    'OBV_21_d_c', 'OBV_22_d_c', 'OBV_23_d_c', 'OBV_24_d_c', 'OBV_25_d_c', 'OBV_26_d_c',
    'OBV_27_d_c', 'OBV_28_d_c', 'OBV_29_d_c', 'OBV_30_d_c', 'OBV_31_d_c', 'OBV_32_d_c',
    'OBV_33_d_c', 'OBV_34_d_c', 'OBV_35_d_c', 'OBV_36_d_c', 'OBV_37_d_c', 'OBV_38_d_c',
    'OBV_39_d_c', 'OBV_40_d_c', 'OBV_41_d_c', 'OBV_42_d_c', 'OBV_43_d_c', 'OBV_44_d_c',
    'OBV_45_d_c', 'OBV_46_d_c', 'OBV_47_d_c', 'OBV_48_d_c', 'OBV_49_d_c', 'OBV_50_d_c',
    'OBV_51_d_c', 'OBV_52_d_c', 'OBV_53_d_c', 'OBV_54_d_c', 'OBV_55_d_c', 'OBV_56_d_c',
    'OBV_57_d_c', 'OBV_58_d_c', 'OBV_59_d_c', 'OBV_60_d_c', 'OBV_61_d_c', 'OBV_62_d_c',
    'OBV_63_d_c', 'OBV_64_d_c', 'OBV_65_d_c', 'OBV_66_d_c', 'OBV_67_d_c', 'OBV_68_d_c',
    'OBV_69_d_c', 'OBV_70_d_c',




]
apply_gaussian_smoothing = [

    'ACC_3_g_c', 'ACC_4_g_c', 'ACC_5_g_c', 'ACC_6_g_c', 'ACC_7_g_c', 'ACC_8_g_c',
    'ACC_9_g_c', 'ACC_10_g_c', 'ACC_11_g_c', 'ACC_12_g_c', 'ACC_13_g_c', 'ACC_14_g_c',
    'ACC_15_g_c', 'ACC_16_g_c', 'ACC_17_g_c', 'ACC_18_g_c', 'ACC_19_g_c', 'ACC_20_g_c',
    'ACC_21_g_c', 'ACC_22_g_c', 'ACC_23_g_c', 'ACC_24_g_c', 'ACC_25_g_c', 'ACC_26_g_c',
    'ACC_27_g_c', 'ACC_28_g_c', 'ACC_29_g_c', 'ACC_30_g_c', 'ACC_31_g_c', 'ACC_32_g_c',
    'ACC_33_g_c', 'ACC_34_g_c', 'ACC_35_g_c', 'ACC_36_g_c', 'ACC_37_g_c', 'ACC_38_g_c',
    'ACC_39_g_c', 'ACC_40_g_c', 'ACC_41_g_c', 'ACC_42_g_c', 'ACC_43_g_c', 'ACC_44_g_c',
    'ACC_45_g_c', 'ACC_46_g_c', 'ACC_47_g_c', 'ACC_48_g_c', 'ACC_49_g_c', 'ACC_50_g_c',
    'ACC_51_g_c', 'ACC_52_g_c', 'ACC_53_g_c', 'ACC_54_g_c', 'ACC_55_g_c', 'ACC_56_g_c',
    'ACC_57_g_c', 'ACC_58_g_c', 'ACC_59_g_c', 'ACC_60_g_c', 'ACC_61_g_c', 'ACC_62_g_c',
    'ACC_63_g_c', 'ACC_64_g_c', 'ACC_65_g_c', 'ACC_66_g_c', 'ACC_67_g_c', 'ACC_68_g_c',
    'ACC_69_g_c', 'ACC_70_g_c',

    'DPO_3_g_c', 'DPO_4_g_c', 'DPO_5_g_c', 'DPO_6_g_c', 'DPO_7_g_c', 'DPO_8_g_c',
    'DPO_9_g_c', 'DPO_10_g_c', 'DPO_11_g_c', 'DPO_12_g_c', 'DPO_13_g_c', 'DPO_14_g_c',
    'DPO_15_g_c', 'DPO_16_g_c', 'DPO_17_g_c', 'DPO_18_g_c', 'DPO_19_g_c', 'DPO_20_g_c',
    'DPO_21_g_c', 'DPO_22_g_c', 'DPO_23_g_c', 'DPO_24_g_c', 'DPO_25_g_c', 'DPO_26_g_c',
    'DPO_27_g_c', 'DPO_28_g_c', 'DPO_29_g_c', 'DPO_30_g_c', 'DPO_31_g_c', 'DPO_32_g_c',
    'DPO_33_g_c', 'DPO_34_g_c', 'DPO_35_g_c', 'DPO_36_g_c', 'DPO_37_g_c', 'DPO_38_g_c',
    'DPO_39_g_c', 'DPO_40_g_c', 'DPO_41_g_c', 'DPO_42_g_c', 'DPO_43_g_c', 'DPO_44_g_c',
    'DPO_45_g_c', 'DPO_46_g_c', 'DPO_47_g_c', 'DPO_48_g_c', 'DPO_49_g_c', 'DPO_50_g_c',
    'DPO_51_g_c', 'DPO_52_g_c', 'DPO_53_g_c', 'DPO_54_g_c', 'DPO_55_g_c', 'DPO_56_g_c',
    'DPO_57_g_c', 'DPO_58_g_c', 'DPO_59_g_c', 'DPO_60_g_c', 'DPO_61_g_c', 'DPO_62_g_c',
    'DPO_63_g_c', 'DPO_64_g_c', 'DPO_65_g_c', 'DPO_66_g_c', 'DPO_67_g_c', 'DPO_68_g_c',
    'DPO_69_g_c', 'DPO_70_g_c',

        'DEMA_3_g_c', 'DEMA_4_g_c', 'DEMA_5_g_c', 'DEMA_6_g_c', 'DEMA_7_g_c', 'DEMA_8_g_c',
    'DEMA_9_g_c', 'DEMA_10_g_c', 'DEMA_11_g_c', 'DEMA_12_g_c', 'DEMA_13_g_c', 'DEMA_14_g_c',
    'DEMA_15_g_c', 'DEMA_16_g_c', 'DEMA_17_g_c', 'DEMA_18_g_c', 'DEMA_19_g_c', 'DEMA_20_g_c',
    'DEMA_21_g_c', 'DEMA_22_g_c', 'DEMA_23_g_c', 'DEMA_24_g_c', 'DEMA_25_g_c', 'DEMA_26_g_c',
    'DEMA_27_g_c', 'DEMA_28_g_c', 'DEMA_29_g_c', 'DEMA_30_g_c', 'DEMA_31_g_c', 'DEMA_32_g_c',
    'DEMA_33_g_c', 'DEMA_34_g_c', 'DEMA_35_g_c', 'DEMA_36_g_c', 'DEMA_37_g_c', 'DEMA_38_g_c',
    'DEMA_39_g_c', 'DEMA_40_g_c', 'DEMA_41_g_c', 'DEMA_42_g_c', 'DEMA_43_g_c', 'DEMA_44_g_c',
    'DEMA_45_g_c', 'DEMA_46_g_c', 'DEMA_47_g_c', 'DEMA_48_g_c', 'DEMA_49_g_c', 'DEMA_50_g_c',
    'DEMA_51_g_c', 'DEMA_52_g_c', 'DEMA_53_g_c', 'DEMA_54_g_c', 'DEMA_55_g_c', 'DEMA_56_g_c',
    'DEMA_57_g_c', 'DEMA_58_g_c', 'DEMA_59_g_c', 'DEMA_60_g_c', 'DEMA_61_g_c', 'DEMA_62_g_c',
    'DEMA_63_g_c', 'DEMA_64_g_c', 'DEMA_65_g_c', 'DEMA_66_g_c', 'DEMA_67_g_c', 'DEMA_68_g_c',
    'DEMA_69_g_c', 'DEMA_70_g_c',

    'ATR_4_g_c', 'ATR_5_g_c', 'ATR_6_g_c', 'ATR_8_g_c', 'ATR_19_g_c', 'ATR_20_g_c', 'ATR_22_g_c',
    'ATR_23_g_c', 'ATR_25_g_c', 'ATR_26_g_c', 'ATR_28_g_c',
    'ATR_29_g_c', 'ATR_31_g_c', 'ATR_32_g_c', 'ATR_36_g_c', 'ATR_39_g_c', 
    'ATR_41_g_c', 'ATR_42_g_c', 'ATR_45_g_c', 'ATR_46_g_c',
    'ATR_48_g_c', 'ATR_50_g_c', 'ATR_51_g_c', 
    'ATR_53_g_c', 'ATR_54_g_c', 'ATR_57_g_c', 'ATR_60_g_c', 'ATR_61_g_c', 'ATR_62_g_c', 'ATR_63_g_c', 
    'ATR_65_g_c', 'ATR_67_g_c', 'ATR_68_g_c', 'ATR_69_g_c', 'ATR_70_g_c',    
    'CCI_13_g_c', 'CCI_30_g_c', 'CHANDE_22_g_c', 'CHANDE_3_g_c',
    'CHANDE_32_g_c', 'CHANDE_33_g_c', 'CHANDE_36_g_c', 'CHANDE_37_g_c', 'CHANDE_38_g_c', 'CHANDE_42_g_c', 'CHANDE_44_g_c',
    'CHANDE_8_g_c', 'CMF_21_g_c', 'CMF_32_g_c', 'EOM_5_g_c', 'MFI_10_g_c', 'ROC_12_g_c', 'ROC_22_g_c',
    'ROC_24_g_c', 'ROC_3_g_c', 'ROC_38_g_c', 'ROC_6_g_c', 'ROC_7_g_c', 'ROC_8_g_c', 'RSI_10_g_c', 'RSI_16_g_c', 'RSI_17_g_c',
    'RSI_18_g_c', 'RSI_19_g_c', 'RSI_20_g_c', 'RSI_34_g_c', 'RSI_44_g_c', 'RSI_48_g_c', 'RSI_49_g_c', 'RSI_5_g_c', 'SEOM_10_g_c',
    'SEOM_7_g_c', 'VO_3_g_c', 'VO_46_g_c', 'VO_48_g_c', 'VO_5_g_c', 'VO_7_g_c', 'VO_9_g_c', 'VWAP_48_g_c',



    'OBV_3_g_c', 'OBV_4_g_c', 'OBV_5_g_c', 'OBV_6_g_c', 'OBV_7_g_c', 'OBV_8_g_c',
    'OBV_9_g_c', 'OBV_10_g_c', 'OBV_11_g_c', 'OBV_12_g_c', 'OBV_13_g_c', 'OBV_14_g_c',
    'OBV_15_g_c', 'OBV_16_g_c', 'OBV_17_g_c', 'OBV_18_g_c', 'OBV_19_g_c', 'OBV_20_g_c',
    'OBV_21_g_c', 'OBV_22_g_c', 'OBV_23_g_c', 'OBV_24_g_c', 'OBV_25_g_c', 'OBV_26_g_c',
    'OBV_27_g_c', 'OBV_28_g_c', 'OBV_29_g_c', 'OBV_30_g_c', 'OBV_31_g_c', 'OBV_32_g_c',
    'OBV_33_g_c', 'OBV_34_g_c', 'OBV_35_g_c', 'OBV_36_g_c', 'OBV_37_g_c', 'OBV_38_g_c',
    'OBV_39_g_c', 'OBV_40_g_c', 'OBV_41_g_c', 'OBV_42_g_c', 'OBV_43_g_c', 'OBV_44_g_c',
    'OBV_45_g_c', 'OBV_46_g_c', 'OBV_47_g_c', 'OBV_48_g_c', 'OBV_49_g_c', 'OBV_50_g_c',
    'OBV_51_g_c', 'OBV_52_g_c', 'OBV_53_g_c', 'OBV_54_g_c', 'OBV_55_g_c', 'OBV_56_g_c',
    'OBV_57_g_c', 'OBV_58_g_c', 'OBV_59_g_c', 'OBV_60_g_c', 'OBV_61_g_c', 'OBV_62_g_c',
    'OBV_63_g_c', 'OBV_64_g_c', 'OBV_65_g_c', 'OBV_66_g_c', 'OBV_67_g_c', 'OBV_68_g_c',
    'OBV_69_g_c', 'OBV_70_g_c',


     

]
apply_wavelet_transform = [

    'ACC_3_w_c', 'ACC_4_w_c', 'ACC_5_w_c', 'ACC_6_w_c', 'ACC_7_w_c', 'ACC_8_w_c',
    'ACC_9_w_c', 'ACC_10_w_c', 'ACC_11_w_c', 'ACC_12_w_c', 'ACC_13_w_c', 'ACC_14_w_c',
    'ACC_15_w_c', 'ACC_16_w_c', 'ACC_17_w_c', 'ACC_18_w_c', 'ACC_19_w_c', 'ACC_20_w_c',
    'ACC_21_w_c', 'ACC_22_w_c', 'ACC_23_w_c', 'ACC_24_w_c', 'ACC_25_w_c', 'ACC_26_w_c',
    'ACC_27_w_c', 'ACC_28_w_c', 'ACC_29_w_c', 'ACC_30_w_c', 'ACC_31_w_c', 'ACC_32_w_c',
    'ACC_33_w_c', 'ACC_34_w_c', 'ACC_35_w_c', 'ACC_36_w_c', 'ACC_37_w_c', 'ACC_38_w_c',
    'ACC_39_w_c', 'ACC_40_w_c', 'ACC_41_w_c', 'ACC_42_w_c', 'ACC_43_w_c', 'ACC_44_w_c',
    'ACC_45_w_c', 'ACC_46_w_c', 'ACC_47_w_c', 'ACC_48_w_c', 'ACC_49_w_c', 'ACC_50_w_c',
    'ACC_51_w_c', 'ACC_52_w_c', 'ACC_53_w_c', 'ACC_54_w_c', 'ACC_55_w_c', 'ACC_56_w_c',
    'ACC_57_w_c', 'ACC_58_w_c', 'ACC_59_w_c', 'ACC_60_w_c', 'ACC_61_w_c', 'ACC_62_w_c',
    'ACC_63_w_c', 'ACC_64_w_c', 'ACC_65_w_c', 'ACC_66_w_c', 'ACC_67_w_c', 'ACC_68_w_c',
    'ACC_69_w_c', 'ACC_70_w_c',

    'DPO_3_w_c', 'DPO_4_w_c', 'DPO_5_w_c', 'DPO_6_w_c', 'DPO_7_w_c', 'DPO_8_w_c',
    'DPO_9_w_c', 'DPO_10_w_c', 'DPO_11_w_c', 'DPO_12_w_c', 'DPO_13_w_c', 'DPO_14_w_c',
    'DPO_15_w_c', 'DPO_16_w_c', 'DPO_17_w_c', 'DPO_18_w_c', 'DPO_19_w_c', 'DPO_20_w_c',
    'DPO_21_w_c', 'DPO_22_w_c', 'DPO_23_w_c', 'DPO_24_w_c', 'DPO_25_w_c', 'DPO_26_w_c',
    'DPO_27_w_c', 'DPO_28_w_c', 'DPO_29_w_c', 'DPO_30_w_c', 'DPO_31_w_c', 'DPO_32_w_c',
    'DPO_33_w_c', 'DPO_34_w_c', 'DPO_35_w_c', 'DPO_36_w_c', 'DPO_37_w_c', 'DPO_38_w_c',
    'DPO_39_w_c', 'DPO_40_w_c', 'DPO_41_w_c', 'DPO_42_w_c', 'DPO_43_w_c', 'DPO_44_w_c',
    'DPO_45_w_c', 'DPO_46_w_c', 'DPO_47_w_c', 'DPO_48_w_c', 'DPO_49_w_c', 'DPO_50_w_c',
    'DPO_51_w_c', 'DPO_52_w_c', 'DPO_53_w_c', 'DPO_54_w_c', 'DPO_55_w_c', 'DPO_56_w_c',
    'DPO_57_w_c', 'DPO_58_w_c', 'DPO_59_w_c', 'DPO_60_w_c', 'DPO_61_w_c', 'DPO_62_w_c',
    'DPO_63_w_c', 'DPO_64_w_c', 'DPO_65_w_c', 'DPO_66_w_c', 'DPO_67_w_c', 'DPO_68_w_c',
    'DPO_69_w_c', 'DPO_70_w_c',

        'DEMA_3_w_c', 'DEMA_4_w_c', 'DEMA_5_w_c', 'DEMA_6_w_c', 'DEMA_7_w_c', 'DEMA_8_w_c',
    'DEMA_9_w_c', 'DEMA_10_w_c', 'DEMA_11_w_c', 'DEMA_12_w_c', 'DEMA_13_w_c', 'DEMA_14_w_c',
    'DEMA_15_w_c', 'DEMA_16_w_c', 'DEMA_17_w_c', 'DEMA_18_w_c', 'DEMA_19_w_c', 'DEMA_20_w_c',
    'DEMA_21_w_c', 'DEMA_22_w_c', 'DEMA_23_w_c', 'DEMA_24_w_c', 'DEMA_25_w_c', 'DEMA_26_w_c',
    'DEMA_27_w_c', 'DEMA_28_w_c', 'DEMA_29_w_c', 'DEMA_30_w_c', 'DEMA_31_w_c', 'DEMA_32_w_c',
    'DEMA_33_w_c', 'DEMA_34_w_c', 'DEMA_35_w_c', 'DEMA_36_w_c', 'DEMA_37_w_c', 'DEMA_38_w_c',
    'DEMA_39_w_c', 'DEMA_40_w_c', 'DEMA_41_w_c', 'DEMA_42_w_c', 'DEMA_43_w_c', 'DEMA_44_w_c',
    'DEMA_45_w_c', 'DEMA_46_w_c', 'DEMA_47_w_c', 'DEMA_48_w_c', 'DEMA_49_w_c', 'DEMA_50_w_c',
    'DEMA_51_w_c', 'DEMA_52_w_c', 'DEMA_53_w_c', 'DEMA_54_w_c', 'DEMA_55_w_c', 'DEMA_56_w_c',
    'DEMA_57_w_c', 'DEMA_58_w_c', 'DEMA_59_w_c', 'DEMA_60_w_c', 'DEMA_61_w_c', 'DEMA_62_w_c',
    'DEMA_63_w_c', 'DEMA_64_w_c', 'DEMA_65_w_c', 'DEMA_66_w_c', 'DEMA_67_w_c', 'DEMA_68_w_c',
    'DEMA_69_w_c', 'DEMA_70_w_c',

    'ATR_4_w_c', 'ATR_5_w_c', 'ATR_6_w_c', 'ATR_7_w_c', 'ATR_8_w_c', 'ATR_9_w_c', 'ATR_10_w_c',
    'ATR_11_w_c', 'ATR_12_w_c', 'ATR_13_w_c', 'ATR_14_w_c', 'ATR_15_w_c', 'ATR_16_w_c',
    'ATR_17_w_c', 'ATR_18_w_c', 'ATR_19_w_c', 'ATR_20_w_c', 'ATR_21_w_c', 'ATR_22_w_c',
    'ATR_23_w_c', 'ATR_24_w_c', 'ATR_25_w_c', 'ATR_26_w_c', 'ATR_27_w_c', 'ATR_28_w_c',
    'ATR_29_w_c', 'ATR_30_w_c', 'ATR_31_w_c', 'ATR_32_w_c', 'ATR_33_w_c', 'ATR_34_w_c',
    'ATR_35_w_c', 'ATR_36_w_c', 'ATR_37_w_c', 'ATR_38_w_c', 'ATR_39_w_c', 'ATR_40_w_c',
    'ATR_41_w_c', 'ATR_42_w_c', 'ATR_43_w_c', 'ATR_44_w_c', 'ATR_45_w_c', 'ATR_46_w_c',
    'ATR_47_w_c', 'ATR_48_w_c', 'ATR_49_w_c', 'ATR_50_w_c', 'ATR_51_w_c', 'ATR_52_w_c',
    'ATR_53_w_c', 'ATR_54_w_c', 'ATR_55_w_c', 'ATR_56_w_c', 'ATR_57_w_c', 'ATR_58_w_c',
    'ATR_59_w_c', 'ATR_60_w_c', 'ATR_61_w_c', 'ATR_62_w_c', 'ATR_63_w_c', 'ATR_64_w_c',
    'ATR_65_w_c', 'ATR_66_w_c', 'ATR_67_w_c', 'ATR_68_w_c', 'ATR_69_w_c', 'ATR_70_w_c',

    'AROON_DOWN_10_w_c', 'AROON_DOWN_26_w_c', 'AROON_DOWN_44_w_c', 'AROON_UP_39_w_c', 'AROON_UP_4_w_c',
    'AROON_UP_42_w_c', 'AROON_UP_43_w_c', 'AROON_UP_44_w_c', 'AROON_UP_5_w_c', 'AROON_UP_6_w_c', 'AROON_UP_7_w_c',
    'AROON_UP_8_w_c', 'CHANDE_14_w_c', 'CHANDE_20_w_c',
    'CHANDE_22_w_c', 'CHANDE_25_w_c', 'CHANDE_29_w_c', 'CHANDE_37_w_c', 'CHANDE_4_w_c', 'CHANDE_7_w_c', 'CMF_11_w_c',
    'CMF_40_w_c', 'CMF_42_w_c', 'MFI_12_w_c', 'MFI_37_w_c', 'MFI_38_w_c', 'MFI_42_w_c', 'MFI_43_w_c', 'MFI_44_w_c',
    'ROC_28_w_c', 'ROC_29_w_c',
    'RSI_44_w_c', 'SCHAFF_19_w_c', 'SEOM_10_w_c', 'SEOM_26_w_c', 'SEOM_4_w_c', 'SEOM_41_w_c', 'SEOM_45_w_c', 'SEOM_46_w_c',
    'SEOM_48_w_c', 'SEOM_5_w_c', 'VO_10_w_c', 'VO_18_w_c', 'VO_4_w_c', 'VO_44_w_c',
    

    'OBV_3_w_c', 'OBV_4_w_c', 'OBV_5_w_c', 'OBV_6_w_c', 'OBV_7_w_c', 'OBV_8_w_c',
    'OBV_9_w_c', 'OBV_10_w_c', 'OBV_11_w_c', 'OBV_12_w_c', 'OBV_13_w_c', 'OBV_14_w_c',
    'OBV_15_w_c', 'OBV_16_w_c', 'OBV_17_w_c', 'OBV_18_w_c', 'OBV_19_w_c', 'OBV_20_w_c',
    'OBV_21_w_c', 'OBV_22_w_c', 'OBV_23_w_c', 'OBV_24_w_c', 'OBV_25_w_c', 'OBV_26_w_c',
    'OBV_27_w_c', 'OBV_28_w_c', 'OBV_29_w_c', 'OBV_30_w_c', 'OBV_31_w_c', 'OBV_32_w_c',
    'OBV_33_w_c', 'OBV_34_w_c', 'OBV_35_w_c', 'OBV_36_w_c', 'OBV_37_w_c', 'OBV_38_w_c',
    'OBV_39_w_c', 'OBV_40_w_c', 'OBV_41_w_c', 'OBV_42_w_c', 'OBV_43_w_c', 'OBV_44_w_c',
    'OBV_45_w_c', 'OBV_46_w_c', 'OBV_47_w_c', 'OBV_48_w_c', 'OBV_49_w_c', 'OBV_50_w_c',
    'OBV_51_w_c', 'OBV_52_w_c', 'OBV_53_w_c', 'OBV_54_w_c', 'OBV_55_w_c', 'OBV_56_w_c',
    'OBV_57_w_c', 'OBV_58_w_c', 'OBV_59_w_c', 'OBV_60_w_c', 'OBV_61_w_c', 'OBV_62_w_c',
    'OBV_63_w_c', 'OBV_64_w_c', 'OBV_65_w_c', 'OBV_66_w_c', 'OBV_67_w_c', 'OBV_68_w_c',
    'OBV_69_w_c', 'OBV_70_w_c',


   


]

detrend_series = [

    'ACC_3_e_c', 'ACC_4_e_c', 'ACC_5_e_c', 'ACC_6_e_c', 'ACC_7_e_c', 'ACC_8_e_c',
    'ACC_9_e_c', 'ACC_10_e_c', 'ACC_11_e_c', 'ACC_12_e_c', 'ACC_13_e_c', 'ACC_14_e_c',
    'ACC_15_e_c', 'ACC_16_e_c', 'ACC_17_e_c', 'ACC_18_e_c', 'ACC_19_e_c', 'ACC_20_e_c',
    'ACC_21_e_c', 'ACC_22_e_c', 'ACC_23_e_c', 'ACC_24_e_c', 'ACC_25_e_c', 'ACC_26_e_c',
    'ACC_27_e_c', 'ACC_28_e_c', 'ACC_29_e_c', 'ACC_30_e_c', 'ACC_31_e_c', 'ACC_32_e_c',
    'ACC_33_e_c', 'ACC_34_e_c', 'ACC_35_e_c', 'ACC_36_e_c', 'ACC_37_e_c', 'ACC_38_e_c',
    'ACC_39_e_c', 'ACC_40_e_c', 'ACC_41_e_c', 'ACC_42_e_c', 'ACC_43_e_c', 'ACC_44_e_c',
    'ACC_45_e_c', 'ACC_46_e_c', 'ACC_47_e_c', 'ACC_48_e_c', 'ACC_49_e_c', 'ACC_50_e_c',
    'ACC_51_e_c', 'ACC_52_e_c', 'ACC_53_e_c', 'ACC_54_e_c', 'ACC_55_e_c', 'ACC_56_e_c',
    'ACC_57_e_c', 'ACC_58_e_c', 'ACC_59_e_c', 'ACC_60_e_c', 'ACC_61_e_c', 'ACC_62_e_c',
    'ACC_63_e_c', 'ACC_64_e_c', 'ACC_65_e_c', 'ACC_66_e_c', 'ACC_67_e_c', 'ACC_68_e_c',
    'ACC_69_e_c', 'ACC_70_e_c',

    'DPO_3_e_c', 'DPO_4_e_c', 'DPO_5_e_c', 'DPO_6_e_c', 'DPO_7_e_c', 'DPO_8_e_c',
    'DPO_9_e_c', 'DPO_10_e_c', 'DPO_11_e_c', 'DPO_12_e_c', 'DPO_13_e_c', 'DPO_14_e_c',
    'DPO_15_e_c', 'DPO_16_e_c', 'DPO_17_e_c', 'DPO_18_e_c', 'DPO_19_e_c', 'DPO_20_e_c',
    'DPO_21_e_c', 'DPO_22_e_c', 'DPO_23_e_c', 'DPO_24_e_c', 'DPO_25_e_c', 'DPO_26_e_c',
    'DPO_27_e_c', 'DPO_28_e_c', 'DPO_29_e_c', 'DPO_30_e_c', 'DPO_31_e_c', 'DPO_32_e_c',
    'DPO_33_e_c', 'DPO_34_e_c', 'DPO_35_e_c', 'DPO_36_e_c', 'DPO_37_e_c', 'DPO_38_e_c',
    'DPO_39_e_c', 'DPO_40_e_c', 'DPO_41_e_c', 'DPO_42_e_c', 'DPO_43_e_c', 'DPO_44_e_c',
    'DPO_45_e_c', 'DPO_46_e_c', 'DPO_47_e_c', 'DPO_48_e_c', 'DPO_49_e_c', 'DPO_50_e_c',
    'DPO_51_e_c', 'DPO_52_e_c', 'DPO_53_e_c', 'DPO_54_e_c', 'DPO_55_e_c', 'DPO_56_e_c',
    'DPO_57_e_c', 'DPO_58_e_c', 'DPO_59_e_c', 'DPO_60_e_c', 'DPO_61_e_c', 'DPO_62_e_c',
    'DPO_63_e_c', 'DPO_64_e_c', 'DPO_65_e_c', 'DPO_66_e_c', 'DPO_67_e_c', 'DPO_68_e_c',
    'DPO_69_e_c', 'DPO_70_e_c',

    'DEMA_3_e_c', 'DEMA_4_e_c', 'DEMA_5_e_c', 'DEMA_6_e_c', 'DEMA_7_e_c', 'DEMA_8_e_c',
    'DEMA_9_e_c', 'DEMA_10_e_c', 'DEMA_11_e_c', 'DEMA_12_e_c', 'DEMA_13_e_c', 'DEMA_14_e_c',
    'DEMA_15_e_c', 'DEMA_16_e_c', 'DEMA_17_e_c', 'DEMA_18_e_c', 'DEMA_19_e_c', 'DEMA_20_e_c',
    'DEMA_21_e_c', 'DEMA_22_e_c', 'DEMA_23_e_c', 'DEMA_24_e_c', 'DEMA_25_e_c', 'DEMA_26_e_c',
    'DEMA_27_e_c', 'DEMA_28_e_c', 'DEMA_29_e_c', 'DEMA_30_e_c', 'DEMA_31_e_c', 'DEMA_32_e_c',
    'DEMA_33_e_c', 'DEMA_34_e_c', 'DEMA_35_e_c', 'DEMA_36_e_c', 'DEMA_37_e_c', 'DEMA_38_e_c',
    'DEMA_39_e_c', 'DEMA_40_e_c', 'DEMA_41_e_c', 'DEMA_42_e_c', 'DEMA_43_e_c', 'DEMA_44_e_c',
    'DEMA_45_e_c', 'DEMA_46_e_c', 'DEMA_47_e_c', 'DEMA_48_e_c', 'DEMA_49_e_c', 'DEMA_50_e_c',
    'DEMA_51_e_c', 'DEMA_52_e_c', 'DEMA_53_e_c', 'DEMA_54_e_c', 'DEMA_55_e_c', 'DEMA_56_e_c',
    'DEMA_57_e_c', 'DEMA_58_e_c', 'DEMA_59_e_c', 'DEMA_60_e_c', 'DEMA_61_e_c', 'DEMA_62_e_c',
    'DEMA_63_e_c', 'DEMA_64_e_c', 'DEMA_65_e_c', 'DEMA_66_e_c', 'DEMA_67_e_c', 'DEMA_68_e_c',
    'DEMA_69_e_c', 'DEMA_70_e_c',

    'ATR_4_e_c', 'ATR_5_e_c', 'ATR_6_e_c', 'ATR_7_e_c', 'ATR_8_e_c', 'ATR_9_e_c', 'ATR_10_e_c',
    'ATR_11_e_c', 'ATR_12_e_c', 'ATR_13_e_c', 'ATR_14_e_c', 'ATR_15_e_c', 'ATR_16_e_c',
    'ATR_17_e_c', 'ATR_18_e_c', 'ATR_19_e_c', 'ATR_20_e_c', 'ATR_21_e_c', 'ATR_22_e_c',
    'ATR_23_e_c', 'ATR_24_e_c', 'ATR_25_e_c', 'ATR_26_e_c', 'ATR_27_e_c', 'ATR_28_e_c',
    'ATR_29_e_c', 'ATR_30_e_c', 'ATR_31_e_c', 'ATR_32_e_c', 'ATR_33_e_c', 'ATR_34_e_c',
    'ATR_35_e_c', 'ATR_36_e_c', 'ATR_37_e_c', 'ATR_38_e_c', 'ATR_39_e_c', 'ATR_40_e_c',
    'ATR_41_e_c', 'ATR_42_e_c', 'ATR_43_e_c', 'ATR_44_e_c', 'ATR_45_e_c', 'ATR_46_e_c',
    'ATR_47_e_c', 'ATR_48_e_c', 'ATR_49_e_c', 'ATR_50_e_c', 'ATR_51_e_c', 'ATR_52_e_c',
    'ATR_53_e_c', 'ATR_54_e_c', 'ATR_55_e_c', 'ATR_56_e_c', 'ATR_57_e_c', 'ATR_58_e_c',
    'ATR_59_e_c', 'ATR_60_e_c', 'ATR_61_e_c', 'ATR_62_e_c', 'ATR_63_e_c', 'ATR_64_e_c',
    'ATR_65_e_c', 'ATR_66_e_c', 'ATR_67_e_c', 'ATR_68_e_c', 'ATR_69_e_c', 'ATR_70_e_c',    
    'CCI_10_e_c', 'CCI_13_e_c', 'CHANDE_12_e_c',
    'CHANDE_3_e_c', 'CHANDE_38_e_c', 'CHANDE_4_e_c', 'CHANDE_5_e_c', 'CMF_22_e_c', 'CMF_29_e_c', 'EOM_15_e_c', 
    'EOM_20_e_c',  
    'MFI_14_e_c', 'MFI_21_e_c', 'MFI_28_e_c', 'MFI_48_e_c', 'MFI_5_e_c', 'PB_18_e_c',
    'PB_4_e_c', 'ROC_11_e_c', 'ROC_30_e_c', 'ROC_37_e_c', 'ROC_42_e_c', 'ROC_8_e_c', 'RSI_12_e_c', 'RSI_3_e_c', 
    'RSI_40_e_c', 'RSI_42_e_c', 'RSI_44_e_c', 'RSI_49_e_c', 'SCHAFF_3_e_c', 'SCHAFF_48_e_c', 'SEOM_10_e_c', 'SEOM_11_e_c',
    'SEOM_13_e_c', 'SEOM_23_e_c', 'SEOM_43_e_c', 'VO_3_e_c', 'VO_42_e_c', 'VO_43_e_c', 'VO_46_e_c', 'VO_49_e_c', 
    'VWAP_49_e_c',
   

    'OBV_3_e_c', 'OBV_4_e_c', 'OBV_5_e_c', 'OBV_6_e_c', 'OBV_7_e_c', 'OBV_8_e_c',
    'OBV_9_e_c', 'OBV_10_e_c', 'OBV_11_e_c', 'OBV_12_e_c', 'OBV_13_e_c', 'OBV_14_e_c',
    'OBV_15_e_c', 'OBV_16_e_c', 'OBV_17_e_c', 'OBV_18_e_c', 'OBV_19_e_c', 'OBV_20_e_c',
    'OBV_21_e_c', 'OBV_22_e_c', 'OBV_23_e_c', 'OBV_24_e_c', 'OBV_25_e_c', 'OBV_26_e_c',
    'OBV_27_e_c', 'OBV_28_e_c', 'OBV_29_e_c', 'OBV_30_e_c', 'OBV_31_e_c', 'OBV_32_e_c',
    'OBV_33_e_c', 'OBV_34_e_c', 'OBV_35_e_c', 'OBV_36_e_c', 'OBV_37_e_c', 'OBV_38_e_c',
    'OBV_39_e_c', 'OBV_40_e_c', 'OBV_41_e_c', 'OBV_42_e_c', 'OBV_43_e_c', 'OBV_44_e_c',
    'OBV_45_e_c', 'OBV_46_e_c', 'OBV_47_e_c', 'OBV_48_e_c', 'OBV_49_e_c', 'OBV_50_e_c',
    'OBV_51_e_c', 'OBV_52_e_c', 'OBV_53_e_c', 'OBV_54_e_c', 'OBV_55_e_c', 'OBV_56_e_c',
    'OBV_57_e_c', 'OBV_58_e_c', 'OBV_59_e_c', 'OBV_60_e_c', 'OBV_61_e_c', 'OBV_62_e_c',
    'OBV_63_e_c', 'OBV_64_e_c', 'OBV_65_e_c', 'OBV_66_e_c', 'OBV_67_e_c', 'OBV_68_e_c',
    'OBV_69_e_c', 'OBV_70_e_c'


  
    
]

indicators_dict = {
    'standard_5m_close': standard_indicators,
    'add_dynamism_to_series_5m_close': add_dynamism_to_series,
    'apply_gaussian_smoothing_5m_close': apply_gaussian_smoothing,
    'apply_wavelet_transform_5m_close': apply_wavelet_transform,
    'detrend_series_5m_close': detrend_series
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
        'pyti.aroon',
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
            for indicator in indicators_dict[table_name]:
                indicator_columns.append(f"{indicator} INTEGER")
            
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
    bot_db_filepath = bot_db_path("standard_5m_close")

    indicators_dict = {
        'standard_5m_close': standard_indicators,
        'add_dynamism_to_series_5m_close': add_dynamism_to_series,
        'apply_gaussian_smoothing_5m_close': apply_gaussian_smoothing,
        'apply_wavelet_transform_5m_close': apply_wavelet_transform,
        'detrend_series_5m_close': detrend_series
    }

    async with aiosqlite.connect(bot_db_filepath) as readdb:
        async with readdb.cursor() as cursor:
            for table_name in indicators_dict.keys():  # Using table_name directly
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
                        # Adjust the rank based on twice the maximum window size
                        length_range = determine_length_range(table_name)
                        if length_range:
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



def combined_aroon(high_data, low_data, period):
    """
    Combined Aroon.

    Returns both Aroon Up and Aroon Down as a dictionary.
    """
    a_up = AROON.aroon_up(high_data, period)
    a_down = AROON.aroon_down(low_data, period)
    return {"AROON_UP": a_up, "AROON_DOWN": a_down}

def converted_aroon(high_data, low_data, period):
    """
    Combined Aroon.

    Returns both Aroon Up and Aroon Down as a dictionary.
    """
    # Convert Series to list for processing
    high_data_list = high_data.tolist()
    low_data_list = low_data.tolist()

    a_up_c = AROON.aroon_up(high_data_list, period)
    a_down_c = AROON.aroon_down(low_data_list, period)

    return {"AROON_UP": a_up_c, "AROON_DOWN": a_down_c}


def determine_long_slow_periods(window_size):
    length_range = determine_length_range(window_size)
    if length_range:
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
        column_name = f"{indicator_name}_{window_size}_u_c"
        indicator_data[indicator_name] = pd.Series(indicator_name.lower() + '_values')[column_name].tolist()
    
    standard_df = pd.DataFrame(indicator_data)
    
    return standard_df

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

    print("Starting calculate_standard for calculate_standard: 5m_close")

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
        dpo_values_dict = {period: DPO.detrended_price_oscillator(close_prices, period) for period in dpo_periods}
        for period, dpo_values in dpo_values_dict.items():
            key = f'DPO_{period}_u_c',
            dpo_values = np.where(np.isfinite(dpo_values), dpo_values, 0.0)
            rounded_and_int_values = (pd.Series(dpo_values).round(0) * 1).astype(int)
            indicator_values[key] = rounded_and_int_values.tolist()

        dema_periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]        
        dema_values_dict = {period: DEMA.double_exponential_moving_average(close_prices, period) for period in dema_periods}
        for period, dema_values in dema_values_dict.items():
            key = f'dema_{period}_u_c',
            dema_values = np.where(np.isfinite(dema_values), dema_values, 0.0)
            rounded_and_int_values = (pd.Series(dema_values).round(0) * 1).astype(int)
            indicator_values[key] = rounded_and_int_values.tolist()
            


        mom_periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]        
        mom_values_dict = {period: MOM.momentum(close_prices, period) for period in mom_periods}
        for period, mom_values in mom_values_dict.items():
            key = f'MOM_{period}_u_c',
            mom_values = np.where(np.isfinite(mom_values), mom_values, 0.0)
            rounded_and_int_values = (pd.Series(mom_values).round(0) * 1).astype(int)
            indicator_values[key] = rounded_and_int_values.tolist()

        mae_periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]        
        mae_values_dict = {period: mae_upper_band(close_prices, period, 1) for period in mae_periods}
        for period, mae_values in mae_values_dict.items():
            key = f'MAE_{period}_upper_u_c',
            mae_values = np.where(np.isfinite(mae_values), mae_values, 0.0)
            rounded_and_int_values = (pd.Series(mae_values).round(0) * 1).astype(int)
            indicator_values[key] = rounded_and_int_values.tolist()

      

        mae_periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]        
        mae_values_dict = {period: mae_lower_band(close_prices, period, 1) for period in mae_periods}
        for period, mae_values in mae_values_dict.items():
            key = f'MAE_{period}_lower_u_c',
            mae_values = np.where(np.isfinite(mae_values), mae_values, 0.0)
            rounded_and_int_values = (pd.Series(mae_values).round(0) * 1).astype(int)
            indicator_values[key] = rounded_and_int_values.tolist()

        mae_periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]        
        mae_values_dict = {period: mae_center_band(close_prices, period) for period in mae_periods}
        for period, mae_values in mae_values_dict.items():
            key = f'MAE_{period}_center_u_c',
            mae_values = np.where(np.isfinite(mae_values), mae_values, 0.0)
            rounded_and_int_values = (pd.Series(mae_values).round(0) * 1).astype(int)
            indicator_values[key] = rounded_and_int_values.tolist()

        sd_periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]        
        sd_values_dict = {period: SD.standard_deviation(close_prices, period) for period in sd_periods}
        for period, sd_values in sd_values_dict.items():
            key = f'SD_{period}_u_c',
            sd_values = np.where(np.isfinite(sd_values), sd_values, 0.0)
            rounded_and_int_values = (pd.Series(sd_values).round(0) * 1).astype(int)
            indicator_values[key] = rounded_and_int_values.tolist()

        sv_periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]        
        sv_values_dict = {period: SV.standard_variance(close_prices, period) for period in sv_periods}
        for period, sv_values in sv_values_dict.items():
            key = f'SV_{period}_u_c',
            sv_values = np.where(np.isfinite(sv_values), sv_values, 0.0)
            rounded_and_int_values = (pd.Series(sv_values).round(0) * 1).astype(int)
            indicator_values[key] = rounded_and_int_values.tolist()

        tma_periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]        
        tma_values_dict = {period: TMA.triangular_moving_average(close_prices, period) for period in tma_periods}
        for period, tma_values in tma_values_dict.items():
            key = f'TMA_{period}_u_c',
            tma_values = np.where(np.isfinite(tma_values), tma_values, 0.0)
            rounded_and_int_values = (pd.Series(tma_values).round(0) * 1).astype(int)
            indicator_values[key] = rounded_and_int_values.tolist()

        tema_periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]        
        tema_values_dict = {period: TEMA.triple_exponential_moving_average(close_prices, period) for period in tema_periods}
        for period, tema_values in tema_values_dict.items():
            key = f'TEMA_{period}_u_c',
            tema_values = np.where(np.isfinite(tema_values), tema_values, 0.0)
            rounded_and_int_values = (pd.Series(tema_values).round(0) * 1).astype(int)
            indicator_values[key] = rounded_and_int_values.tolist()
            
        uo_values = UO.ultimate_oscillator(close_prices, low_prices)
        uo_values = np.where(np.isfinite(uo_values), uo_values, 0.0)
        rounded_and_int_values = (pd.Series(uo_values).round(0) * 1).astype(int)
        indicator_values['UO_u_c',] = rounded_and_int_values.tolist()

        vola_periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]        
        vola_values_dict = {period: VOLA.volatility(close_prices, period) for period in vola_periods}
        for period, vola_values in vola_values_dict.items():
            key = f'VOLA_{period}_u_c',
            vola_values = np.where(np.isfinite(vola_values), vola_values, 0.0)
            rounded_and_int_values = (pd.Series(vola_values).round(0) * 1).astype(int)
            indicator_values[key] = rounded_and_int_values.tolist()



        atr_periods = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
        atr_values_dict = {period: ATR.average_true_range(close_prices, period) for period in atr_periods}
        for period, atr_values in atr_values_dict.items():
            key = f'ATR_{period}_u_c',
            atr_values = np.where(np.isfinite(atr_values), atr_values, 0.0)                    
            rounded_and_int_values = (pd.Series(atr_values).round(1) * 10).astype(int)             
            indicator_values[key] = rounded_and_int_values.tolist()  
            atr_values = np.where(np.isfinite(atr_values), atr_values, 0.0)
            rounded_and_int_values = (pd.Series(atr_values).round(0) * 1).astype(int)
            indicator_values[key] = rounded_and_int_values.tolist()                             
            
       
        bb_periods_l = [14, 21, 24, 25, 26, 27, 28, 36, 44, 48, 49, 5]
        bb_values_l_dict = {period: BB.lower_bollinger_band(close_prices, period) for period in bb_periods_l}
        for period, l_bb_values in bb_values_l_dict.items():
            key = f'BB_l_{period}_u_c',
            l_bb_values = np.where(np.isfinite(l_bb_values), l_bb_values, 0.0)
            rounded_and_int_values = (pd.Series(l_bb_values).round(1) * 10).astype(int)
            indicator_values[key] = rounded_and_int_values.tolist()

       

        bb_periods_m = [23, 45, 9]
        bb_values_m_dict = {period: BB.lower_bollinger_band(close_prices, period) for period in bb_periods_m}
        for period, m_bb_values in bb_values_m_dict.items():
            key = f'BB_m_{period}_u_c',
            m_bb_values = np.where(np.isfinite(m_bb_values), m_bb_values, 0.0)
            rounded_and_int_values = (pd.Series(m_bb_values).round(1) * 10).astype(int)
            indicator_values[key] = rounded_and_int_values.tolist()


        bb_periods_u = [12, 3, 37, 38, 39, 44, 45, 46, 49]
        bb_values_u_dict = {period: BB.upper_bollinger_band(close_prices, period) for period in bb_periods_u} 
        for period, u_bb_values in bb_values_u_dict.items():
            key = f'BB_u_{period}_u_c',           
            u_bb_values = np.where(np.isfinite(u_bb_values), u_bb_values, 0.0)            
            rounded_and_int_values = (pd.Series(u_bb_values).round(1) * 10).astype(int)
            indicator_values[key] = rounded_and_int_values.tolist()

            
        
        cci_periods = [7]
        cci_values_dict = {period: CCI.commodity_channel_index(close_prices, high_prices, low_prices, period) for period in cci_periods}
        for period, cci_values in cci_values_dict.items():
            key = f'CCI_{period}_u_c',
            cci_values = np.where(np.isfinite(cci_values), cci_values, 0.0)
            rounded_and_int_values = (pd.Series(cci_values).round(1) * 10).astype(int)
            indicator_values[key] = rounded_and_int_values.tolist()


        chande_periods = [12, 18, 19, 27, 28, 31, 32, 33, 36, 44, 48, 49, 5, 7, 8]
        chande_values_dict = {period: CHANDE.chande_momentum_oscillator(close_prices, period) for period in chande_periods}
        for period, chande_values in chande_values_dict.items():
            key = f'CHANDE_{period}_u_c',
            chande_values = np.where(np.isfinite(chande_values), chande_values, 0.0)
            rounded_and_int_values = (pd.Series(chande_values).round(1) * 10).astype(int)          
            indicator_values[key] = rounded_and_int_values.tolist()



        # MACD
        macd_values_3 = MACD_function(close_prices, 3, 21).tolist()
        macd_values_3 = np.where(np.isfinite(macd_values_3), macd_values_3, 0.0)
        rounded_and_int_values = (pd.Series(macd_values_3).round(2) * 100).astype(int).tolist()
        indicator_values['MACD_3_u_c',] = rounded_and_int_values

        # MFI
        mfi_periods = [19, 49, 6]
        mfi_values_dict = {period: MFI.money_flow_index(close_prices, high_prices, low_prices, volume, period) for period in mfi_periods}
        for period, mfi_values in mfi_values_dict.items():
            key = f'MFI_{period}_u_c',
            mfi_values = np.where(np.isfinite(mfi_values), mfi_values, 0.0)
            rounded_and_int_values = (pd.Series(mfi_values).round(1) * 10).astype(int).tolist()
            indicator_values[key] = rounded_and_int_values

        # OBV
        obv_periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
        obv_values_dict = {period: OBV.on_balance_volume(close_prices, volume) for period in obv_periods}
        for period, obv_values in obv_values_dict.items():
            key = f'OBV_{period}_u_c',
            obv_values = np.where(np.isfinite(obv_values), obv_values, 0.0)
            rounded_and_int_values = (pd.Series(obv_values).round(0) * 1).astype(int)
            indicator_values[key] = rounded_and_int_values.tolist()

        # Percent B
        pb_periods = [13, 3, 4, 5, 8]
        pb_values_dict = {period: BB.percent_b(close_prices, period) for period in pb_periods}
        for period, pb_values in pb_values_dict.items():
            key = f'PB_{period}_u_c',
            pb_values = np.where(np.isfinite(pb_values), pb_values, 0.0)
            rounded_and_int_values = (pd.Series(pb_values).round(1) * 10).astype(int).tolist()
            indicator_values[key] = rounded_and_int_values

        # ROC
        roc_periods = [11, 12, 13, 21, 22, 23, 25, 27, 30, 45, 7]
        roc_values_dict = {period: ROC.rate_of_change(close_prices, period) for period in roc_periods}
        for period, roc_values in roc_values_dict.items():
            key = f'ROC_{period}_u_c',
            roc_values = np.where(np.isfinite(roc_values), roc_values, 0.0)
            rounded_and_int_values = (pd.Series(roc_values).round(2) * 100).astype(int).tolist()
            indicator_values[key] = rounded_and_int_values

      
        # RSI
        rsi_periods = [12, 13, 14, 15, 21, 23, 25, 28, 3, 34, 35, 4, 49, 5, 6, 7, 9]
        rsi_values_dict = {period: RSI.relative_strength_index(close_prices, period) for period in rsi_periods}
        for period, rsi_values in rsi_values_dict.items():
            key = f'RSI_{period}_u_c',
            rsi_values = np.where(np.isfinite(rsi_values), rsi_values, 0.0)
            rounded_and_int_values = (pd.Series(rsi_values).round(1) * 10).astype(int).tolist()
            indicator_values[key] = rounded_and_int_values


        # VO (Assuming VO is a constant value of 3 for now)
        vo_3_values = np.array([3] * len(close_prices))
        vo_3_values = np.where(np.isfinite(vo_3_values), vo_3_values, 0.0)
        rounded_and_int_values = (pd.Series(vo_3_values).round(1) * 10).astype(int).tolist()
        indicator_values['VO_3_u_c',] = rounded_and_int_values

              
      


    except Exception as e:
        tb = traceback.format_exc()
        print(f"Error occurred: {e}\n{tb} 5m_close")
    
    # Create a DataFrame from the indicator values
    standard_df = pd.DataFrame(indicator_values)

    print("Finished calculate_indicators for interval: 5m_close")

    return standard_df




def calculate_dynamism(price_data_df):
    print("Starting calculate_dynamism for interval: 5m_close")

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
        #accum/dist
        acc_values = ACC.accumulation_distribution(close_prices, high_prices, low_prices, volume)
        acc_values_series = pd.Series(acc_values)
        for period in range(3, 71):  # This range covers the periods from 3 to 70 inclusive
            acc_dynamic_values = add_dynamism_to_series(acc_values_series, period)
            acc_dynamic_values = np.where(np.isfinite(acc_dynamic_values), acc_dynamic_values, 0.0)
            rounded_and_int_values = acc_dynamic_values.round(0).astype(int).tolist()
            indicator_values[f'ACC_{period}_d_c',] = rounded_and_int_values



        dpo_periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
        for period in dpo_periods:
            dpo_values = pd.Series(DPO.detrended_price_oscillator(close_prices, period))
            dpo_dynamic_values = add_dynamism_to_series(dpo_values, period)
            key = f'DPO_{period}_d_c'
            dpo_dynamic_values = np.where(np.isfinite(dpo_dynamic_values), dpo_dynamic_values, 0.0)
            rounded_and_int_values = dpo_dynamic_values.round(0).astype(int).tolist()
            indicator_values[key] = rounded_and_int_values

        dema_periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
        dema_values = pd.Series(DEMA.double_exponential_moving_average(close_prices, period))
        dema_values_dict = {period: add_dynamism_to_series(dema_values, period) for period in dema_periods}
        for period, dema_dynamic_values in dema_values_dict.items():
            key = f'dema_{period}_d_c',
            dema_dynamic_values = np.where(np.isfinite(dema_dynamic_values), dema_dynamic_values, 0.0)
            rounded_and_int_values = dema_dynamic_values.round(0).astype(int).tolist()
            indicator_values[key] = rounded_and_int_values
            

        # ATR
        atr_periods = [4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
        atr_values_dict = {period: ATR.average_true_range(close_prices, period) for period in atr_periods}
        for period, atr_values in atr_values_dict.items():
            key = f'ATR_{period}_d_c',
            atr_values = np.where(np.isfinite(atr_values), atr_values, 0.0) 
            dynamic_values = add_dynamism_to_series(pd.Series(atr_values), period)
            rounded_values = (dynamic_values.round(1) * 10).astype(int).tolist()
            indicator_values[key] = rounded_values
            
        # CHANDE
        chande_periods = [16, 17, 19, 26, 38, 6, 7]
        chande_values_dict = {period: CHANDE.chande_momentum_oscillator(close_prices, period) for period in chande_periods}
        for period, chande_values in chande_values_dict.items():
            key = f'CHANDE_{period}_d_c',
            chande_values = np.where(np.isfinite(chande_values), chande_values, 0.0) 
            dynamic_values = add_dynamism_to_series(pd.Series(chande_values), period)
            rounded_values = (dynamic_values.round(1) * 10).astype(int).tolist()
            indicator_values[key] = rounded_values




        # MACD
        macd_values_3 = MACD_function(close_prices, 3, 21)
        macd_values = np.where(np.isfinite(macd_values_3), macd_values_3, 0.0)
        dynamic_values = add_dynamism_to_series(pd.Series(macd_values), 3)
        rounded_and_int_values = (dynamic_values.fillna(0).round(2) * 100).astype(int).tolist()
        indicator_values['MACD_3_d_c',] = rounded_and_int_values

        # MFI
        mfi_periods = [27, 45]
        mfi_values_dict = {period: MFI.money_flow_index(close_prices, high_prices, low_prices, volume, period) for period in mfi_periods}
        for period, mfi_values in mfi_values_dict.items():
            key = f'MFI_{period}_d_c',
            mfi_values = np.where(np.isfinite(mfi_values), mfi_values, 0.0)
            dynamic_values = add_dynamism_to_series(pd.Series(mfi_values), period)
            rounded_and_int_values = (dynamic_values.round(1) * 10).astype(int).tolist()
            indicator_values[key] = rounded_and_int_values

        # OBV
        obv_periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
        obv_values = pd.Series(OBV.on_balance_volume(close_prices, volume))
        obv_values_dict = {period: add_dynamism_to_series(obv_values, period) for period in obv_periods}
        for period, obv_dynamic_values in obv_values_dict.items():
            key = f'OBV_{period}_d_c',
            obv_dynamic_values = np.where(np.isfinite(obv_dynamic_values), obv_dynamic_values, 0.0)
            rounded_and_int_values = obv_dynamic_values.round(0).astype(int).tolist()
            indicator_values[key] = rounded_and_int_values

            
     # ROC
        roc_periods = [14, 18, 20, 24, 36, 4, 40, 7]
        roc_values_dict = {period: ROC.rate_of_change(close_prices, period) for period in roc_periods}
        for period, roc_values in roc_values_dict.items():
            key = f'ROC_{period}_d_c',
            roc_values = np.where(np.isfinite(roc_values), roc_values, 0.0)
            dynamic_values = add_dynamism_to_series(pd.Series(roc_values), period)
            rounded_and_int_values = (dynamic_values.round(2) * 100).astype(int).tolist()
            indicator_values[key] = rounded_and_int_values

        # RSI
        rsi_periods = [15, 24, 26, 3, 31, 33, 39, 4, 47, 48, 49, 5, 9]
        rsi_values_dict = {period: RSI.relative_strength_index(close_prices, period) for period in rsi_periods}
        for period, rsi_values in rsi_values_dict.items():
            key = f'RSI_{period}_d_c',
            rsi_values = np.where(np.isfinite(rsi_values), rsi_values, 0.0)
            dynamic_values = add_dynamism_to_series(pd.Series(rsi_values), period)
            rounded_and_int_values = (dynamic_values.round(1) * 10).astype(int).tolist()
            indicator_values[key] = rounded_and_int_values

    except Exception as e:
        tb = traceback.format_exc()
        print(f"Error occurred: {e}\n{tb} 5m_close")

    # Create a DataFrame from the indicator values
    dynamism_df = pd.DataFrame(indicator_values)

    print("Finished calculate_indicators for interval: 5m_close")

    return dynamism_df



def calculate_gaussian(price_data_df):
    print("Starting calculate_gaussian for 5m_close")
   
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
        dpo_values_dict = {period: DPO.detrended_price_oscillator(close_prices, period) for period in dpo_periods}
        for period, dpo_values in dpo_values_dict.items():
            key = f'DPO_{period}_g_c',
            dpo_values = np.where(np.isfinite(dpo_values), dpo_values, 0.0)
            gaussian_values = apply_gaussian_smoothing(pd.Series(dpo_values), period)
            rounded_and_int_values = gaussian_values.round(0).astype(int).tolist()
            indicator_values[key] = rounded_and_int_values

        dema_periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
        dema_values_dict = {period: DEMA.double_exponential_moving_average(close_prices, period) for period in dema_periods}
        for period, dema_values in dema_values_dict.items():
            key = f'DEMA_{period}_g_c',
            dema_values = np.where(np.isfinite(dema_values), dema_values, 0.0)
            gaussian_values = apply_gaussian_smoothing(pd.Series(dema_values), period)
            rounded_and_int_values = gaussian_values.round(0).astype(int).tolist()
            indicator_values[key] = rounded_and_int_values

        # ATR
        atr_periods = [4, 5, 6, 8, 19, 20, 22, 23, 25, 26, 28, 29, 31, 32, 36, 39, 41, 42, 45, 46, 48, 50,
        51, 53, 54, 57, 60, 61, 62, 63, 65, 67, 68, 69, 70]
        atr_values_dict = {period: ATR.average_true_range(close_prices, period) for period in atr_periods}
        for period, atr_values in atr_values_dict.items():
            key = f'ATR_{period}_g_c',
            atr_values = np.where(np.isfinite(atr_values), atr_values, 0.0)
            atr_smoothed = apply_gaussian_smoothing(pd.Series(atr_values), period)
            rounded_and_int_values = (atr_smoothed.round(1) * 10).astype(int).tolist()
            indicator_values[key] = rounded_and_int_values

           

        cci_periods = [13, 30]
        cci_values_dict = {period: CCI.commodity_channel_index(close_prices, high_prices, low_prices, period) for period in cci_periods}
        for period, cci_values in cci_values_dict.items():
            key = f'CCI_{period}_g_c',
            cci_values = np.where(np.isfinite(cci_values), cci_values, 0.0)
            gaussian_values = apply_gaussian_smoothing(pd.Series(cci_values), period)
            rounded_and_int_values = (gaussian_values.round(1) * 10).astype(int).tolist()
            indicator_values[key] = rounded_and_int_values

        chande_periods = [22, 3, 32, 33, 36, 37, 38, 42, 44, 8]
        chande_values_dict = {period: CHANDE.chande_momentum_oscillator(close_prices, period) for period in chande_periods}
        for period, chande_values in chande_values_dict.items():
            key = f'CHANDE_{period}_g_c',
            chande_values = np.where(np.isfinite(chande_values), chande_values, 0.0)
            gaussian_values= apply_gaussian_smoothing(pd.Series(chande_values), period)
            rounded_and_int_values = (gaussian_values.round(1) * 10).astype(int).tolist()
            indicator_values[key] = rounded_and_int_values
            


        # OBV
        obv_periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
        obv_values_dict = {period: OBV.on_balance_volume(close_prices, volume) for period in obv_periods}
        for period, obv_values in obv_values_dict.items():
            key = f'OBV_{period}_g_c',
            obv_values = np.where(np.isfinite(obv_values), obv_values, 0.0)
            gaussian_values = apply_gaussian_smoothing(pd.Series(obv_values), period)
            rounded_and_int_values = gaussian_values.round(0).astype(int).tolist()
            indicator_values[key] = rounded_and_int_values


        # ROC
        roc_periods = [12, 22, 24, 3, 38, 6, 7, 8]
        roc_values_dict = {period: ROC.rate_of_change(close_prices, period) for period in roc_periods}
        for period, roc_values in roc_values_dict.items():
            key = f'ROC_{period}_g_c',
            roc_values = np.where(np.isfinite(roc_values), roc_values, 0.0)
            gaussian_values = apply_gaussian_smoothing(pd.Series(roc_values), period)
            rounded_and_int_values = (gaussian_values.round(2) * 100).astype(int).tolist()
            indicator_values[key] = rounded_and_int_values

        # RSI
        rsi_periods = [10, 16, 17, 18, 19, 20, 34, 44, 48, 49, 5]
        rsi_values_dict = {period: RSI.relative_strength_index(close_prices, period) for period in rsi_periods}
        for period, rsi_values in rsi_values_dict.items():
            key = f'RSI_{period}_g_c',
            rsi_values = np.where(np.isfinite(rsi_values), rsi_values, 0.0)
            gaussian_values = apply_gaussian_smoothing(pd.Series(rsi_values), period)
            rounded_and_int_values = (gaussian_values.round(1) * 10).astype(int).tolist()
            indicator_values[key] = rounded_and_int_values

    except Exception as e:
        tb = traceback.format_exc()
        print(f"Error occurred: {e}\n{tb} 5m_close")

    # Create a DataFrame from the indicator values
    gaussian_df = pd.DataFrame(indicator_values)


    print("Finished calculate_gaussian for interval: 5m_close")

    return gaussian_df

        
def calculate_wavelet(price_data_df):
    print("Starting calculate_wavelet for interval: 5m_close")
    
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
        dpo_values_dict = {period:  DPO.detrended_price_oscillator(close_prices, period) for period in dpo_periods}
        for period, dpo_values in dpo_values_dict.items():
            key = f'DPO_{period}_w_c',
            dpo_values = np.where(np.isfinite(dpo_values), dpo_values, 0.0)
            transformed_values = apply_wavelet_transform(pd.Series(dpo_values), period)
            rounded_and_int_values = (transformed_values.round(1) * 10).astype(int).tolist()
            indicator_values[key] = rounded_and_int_values

        dema_periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
        dema_values_dict = {period: DEMA.double_exponential_moving_average(close_prices, period) for period in dema_periods}
        for period, dema_values in dema_values_dict.items():
            key = f'DEMA_{period}_w_c',
            dema_values = np.where(np.isfinite(dema_values), dema_values, 0.0)
            transformed_values = apply_wavelet_transform(pd.Series(dema_values), period)
            rounded_and_int_values = (transformed_values.round(1) * 10).astype(int).tolist()
            indicator_values[key] = rounded_and_int_values

        # ATR
        atr_periods = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
        atr_values_dict = {period: ATR.average_true_range(close_prices, period) for period in atr_periods}
        for period, atr_values in atr_values_dict.items():
            key = f'ATR_{period}_w_c',
            atr_values = np.where(np.isfinite(atr_values), atr_values, 0.0)
            transformed_values = apply_wavelet_transform(pd.Series(atr_values), period)
            rounded_and_int_values = (transformed_values.round(1) * 10).astype(int).tolist()
            indicator_values[key] = rounded_and_int_values
                
        # CHANDE
        chande_periods = [14, 20, 22, 25, 29, 37, 4, 7]
        chande_values_dict = {period: CHANDE.chande_momentum_oscillator(close_prices, period) for period in chande_periods}
        for period, chande_values in chande_values_dict.items():
            key = f'CHANDE_{period}_w_c',
            chande_values = np.where(np.isfinite(chande_values), chande_values, 0.0)
            transformed_values = apply_wavelet_transform(pd.Series(chande_values), period)
            rounded_and_int_values = (transformed_values.round(1) * 10).astype(int).tolist()
            indicator_values[key] = rounded_and_int_values






        # OBV
        obv_periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
        obv_values_dict = {period: OBV.on_balance_volume(close_prices, volume) for period in obv_periods}
        for period, obv_values in obv_values_dict.items():
            key = f'OBV_{period}_w_c',
            obv_values = np.where(np.isfinite(obv_values), obv_values, 0.0)
            transformed_values = apply_wavelet_transform(pd.Series(obv_values), period)
            rounded_and_int_values = transformed_values.round(0).astype(int).tolist()
            indicator_values[key] = rounded_and_int_values

        # ROC
        roc_periods = [28, 29]
        roc_values_dict = {period: ROC.rate_of_change(close_prices, period) for period in roc_periods}
        for period, roc_values in roc_values_dict.items():
            key = f'ROC_{period}_w_c',
            roc_values = np.where(np.isfinite(roc_values), roc_values, 0.0)
            transformed_values = apply_wavelet_transform(pd.Series(roc_values), period)
            rounded_and_int_values = (transformed_values.round(2) * 100).astype(int).tolist()
            indicator_values[key] = rounded_and_int_values

        # RSI
        rsi_periods = [44]
        rsi_values_dict = {period: RSI.relative_strength_index(close_prices, period) for period in rsi_periods}
        for period, rsi_values in rsi_values_dict.items():
            key = f'RSI_{period}_w_c',
            rsi_values = np.where(np.isfinite(rsi_values), rsi_values, 0.0)
            transformed_values = apply_wavelet_transform(pd.Series(rsi_values), period)
            rounded_and_int_values = (transformed_values.round(2) * 100).astype(int).tolist()
            indicator_values[key] = rounded_and_int_values
        
     
    
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Error occurred: {e}\n{tb} 5m_close")

    # Create a DataFrame from the indicator values
    wavelet_df = pd.DataFrame(indicator_values)    

    print("Finished calculate_wavelet for interval: 5m_close")

    return wavelet_df

def calculate_detrend(price_data_df):
    print("Starting calculate_detrend for interval: 5m_close")

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
        atr_values_dict = {period: ATR.average_true_range(close_prices, period) for period in atr_periods}
        for period, atr_values in atr_values_dict.items():
            key = f'ATR_{period}_e_c',
            atr_values = np.where(np.isfinite(atr_values), atr_values, 0.0)
            detrended_values = detrend_series(pd.Series(atr_values), period)
            rounded_and_int_values = (detrended_values.round(1) * 10).astype(int).tolist()
            indicator_values[key] = rounded_and_int_values


        acc_values_dict = {period: ACC.accumulation_distribution(close_prices, high_prices, low_prices, volume) for period in range(3, 71)}
        for period in range(3, 71): 
            acc_values = acc_values_dict[period]
            acc_values = np.where(np.isfinite(acc_values), acc_values, 0.0)    
            detrended_values = detrend_series(pd.Series(acc_values), period)
            rounded_and_int_values = detrended_values.round(0).astype(int).tolist()    
            indicator_values[f'ACC_{period}_e_c',] = rounded_and_int_values

            

        dpo_periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
        dpo_values_dict = {period: DPO.detrended_price_oscillator(close_prices, period) for period in dpo_periods}
        for period, dpo_dynamic_values in dpo_values_dict.items():
            key = f'dpo_{period}_e_c',
            dpo_values = np.where(np.isfinite(dpo_dynamic_values), dpo_dynamic_values, 0.0)
            detrended_values = detrend_series(pd.Series(dpo_values), period)
            rounded_and_int_values = (detrended_values.round(0) * 1).astype(int).tolist()
            indicator_values[key] = rounded_and_int_values

        dema_periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
        dema_values_dict = {period: DEMA.double_exponential_moving_average(close_prices, period) for period in dema_periods}
        for period, dema_dynamic_values in dema_values_dict.items():
            key = f'DEMA_{period}_e_c',
            dema_values = np.where(np.isfinite(dema_dynamic_values), dema_dynamic_values, 0.0)
            detrended_values = detrend_series(pd.Series(dema_values), period)
            rounded_and_int_values = (detrended_values.round(0) * 1).astype(int).tolist()
            indicator_values[key] = rounded_and_int_values
              


        # CCI
        cci_periods = [10, 13]
        cci_values_dict = {period: CCI.commodity_channel_index(close_prices, high_prices, low_prices, period) for period in cci_periods}
        for period, cci_values in cci_values_dict.items():
            key = f'CCI_{period}_e_c',
            cci_values = np.where(np.isfinite(cci_values), cci_values, 0.0)
            detrended_values = detrend_series(pd.Series(cci_values), period)
            rounded_and_int_values = (detrended_values.round(1) * 10).astype(int).tolist()
            indicator_values[key] = rounded_and_int_values

                # CHANDE
        chande_periods = [12, 3, 38, 4, 5]
        chande_values_dict = {period: CHANDE.chande_momentum_oscillator(close_prices, period) for period in chande_periods}
        for period, chande_values in chande_values_dict.items():
            key = f'CHANDE_{period}_e_c',
            chande_values = np.where(np.isfinite(chande_values), chande_values, 0.0)
            detrended_values = detrend_series(pd.Series(chande_values), period)
            rounded_and_int_values = (detrended_values.round(1) * 10).astype(int).tolist()
            indicator_values[key] = rounded_and_int_values


        # CMF
        cmf_periods = [22, 29]
        cmf_values_dict = {period: CMF.chaikin_money_flow(close_prices, high_prices, low_prices, volume, period) for period in cmf_periods}
        for period, cmf_values in cmf_values_dict.items():
            key = f'CMF_{period}_e_c',
            cmf_values = np.where(np.isfinite(cmf_values), cmf_values, 0.0)
            detrended_values = detrend_series(pd.Series(cmf_values), period)
            rounded_and_int_values = (detrended_values.round(2) * 100).astype(int).tolist()  
            indicator_values[key] = rounded_and_int_values


        # MFI
        mfi_periods = [14, 21, 28, 48, 5]
        mfi_values_dict = {period: pd.Series(MFI.money_flow_index(close_prices, high_prices, low_prices, volume, period)).fillna(0) for period in mfi_periods}
        for period, mfi_values in mfi_values_dict.items():
            key = f'MFI_{period}_e_c',
            detrended_values = detrend_series(mfi_values, period)
            rounded_and_int_values = (detrended_values.round(1) * 10).astype(int).tolist()
            indicator_values[key] = rounded_and_int_values

        # OBV
        obv_periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
        obv_values_dict = {period: OBV.on_balance_volume(close_prices, volume) for period in obv_periods}
        for period, obv_values in obv_values_dict.items():
            key = f'OBV_{period}_e_c',
            obv_values = np.where(np.isfinite(obv_values), obv_values, 0.0)
            detrended_values = detrend_series(pd.Series(obv_values), period)
            rounded_and_int_values = (detrended_values.round(1) * 10).astype(int).tolist()
            indicator_values[key] = rounded_and_int_values


        # ROC
        roc_periods = [11, 30, 37, 42, 8]
        roc_values_dict = {period: ROC.rate_of_change(close_prices, period) for period in roc_periods}
        for period, roc_values in roc_values_dict.items():
            key = f'ROC_{period}_e_c',
            roc_values = np.where(np.isfinite(roc_values), roc_values, 0.0)
            detrended_values = detrend_series(pd.Series(roc_values), period)
            rounded_and_int_values = (detrended_values.round(1) * 10).astype(int).tolist()
            indicator_values[key] = rounded_and_int_values


        # RSI
        rsi_periods = [12, 3, 40, 42, 44, 49]
        rsi_values_dict = {period: RSI.relative_strength_index(close_prices, period) for period in rsi_periods}
        for period, rsi_values in rsi_values_dict.items():
            key = f'RSI_{period}_e_c',
            rsi_values = np.where(np.isfinite(rsi_values), rsi_values, 0.0)
            detrended_values = detrend_series(pd.Series(rsi_values), period)
            rounded_and_int_values = (detrended_values.round(1) * 10).astype(int).tolist()
            indicator_values[key] = rounded_and_int_values


        # Percent B
        pb_periods = [18,4]
        pb_values_dict = {period: BB.percent_b(close_prices, period) for period in pb_periods}
        for period, pb_values in pb_values_dict.items():
            key = f'PB_{period}_e_c',
            pb_values = np.where(np.isfinite(pb_values), pb_values, 0.0)
            detrended_values = detrend_series(pd.Series(pb_values), period)
            rounded_and_int_values = (pd.Series(pb_values).round(1) * 10).astype(int).tolist()
            indicator_values[key] = rounded_and_int_values
            

      

    except Exception as e:
        tb = traceback.format_exc()
        print(f"Error occurred: {e}\n{tb} 5m_close")

    # Create a DataFrame from the indicator values
    detrend_df = pd.DataFrame(indicator_values)


    print("Finished calculate_detrend for interval: 5m_close")

    return detrend_df


async def fetch_data_from_db(global_start_timestamps):
    
    print("Starting fetch_data_from_db for timeframe: 5m_close")
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
    print("Finished fetch_data_from_db for timeframe: 5m_close ", len(data_df))
    
    return data_df

def manage_process(func_name, function, args):
    print(f"Starting {func_name} on a new thread 5m_close...")
    function(*args)
    print(f"Completed {func_name} 5m_close.")

def run_parallel_calculations(data_df):
    results = {}
    
    with ProcessPoolExecutor() as executor:
        tasks = {
           "standard_5m_close": executor.submit(calculate_standard, data_df),
           "add_dynamism_to_series_5m_close": executor.submit(calculate_dynamism, data_df),
           "apply_gaussian_smoothing_5m_close": executor.submit(calculate_gaussian, data_df),
           "apply_wavelet_transform_5m_close": executor.submit(calculate_wavelet, data_df),
           "detrend_series_5m_close": executor.submit(calculate_detrend, data_df)
        }
        
        for key, future in tasks.items():
            results[key] = future.result()
        
    print("returned parallel calc_5m_close")
    return results



async def process_indicator_series(botdb_queue, global_start_timestamps): 
    print(f"Fetching data from database for indicator_5m_close")
    data_df = await fetch_data_from_db(global_start_timestamps)
    print(f"Fetched {len(data_df)} 5m_close")
    
    dfs_dict = run_parallel_calculations(data_df)

    for key, dfs in dfs_dict.items():
        dfs = dfs.fillna(0).astype(int)
        print(f"Finished processing {key} 5m_close")

        # Check if DataFrame is empty and skip the rest if it is
        if dfs.empty:
            print(f"No results for {key}. Skipping_5m_close.")
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

    print(f"Sentinel Indicatorvalues_5m, close")
    botdb_queue.put(None)


async def indicatorvalues_async_main(botdb_queue, table_created_event_5m_c, table_created_event_15m_c, table_created_event_1h_c):
    df_indicators = {}  # This is now a dictionary
    print("Starting indicatorvalues_async_main_5m_close")
    await create_indicator_tables(botdb_queue)
    print("awaiting create table_5m_close")
    table_created_event_5m_c.wait()
    await asyncio.sleep(10)
    print("awaiting await event_5m_close")
    global_start_timestamps, *_ = await get_start_timestamp_for_indicators()


    print("getting timestamp_5m_close")
    await process_indicator_series(botdb_queue, global_start_timestamps)
    #manage_process_thread = threading.Thread(target=manage_process, args=("indicatorvalues_main_15m_close", indicatorvalues_main_15m_c, (botdb_queue, table_created_event_15m_c)))
    #manage_process_thread.start()
    #manage_process_thread.join()

def indicatorvalues_main_5m_c(botdb_queue, table_created_event_5m_c, table_created_event_15m_c, table_created_event_1h_c):
    print("Indicator values running_5m_close")
    #setup_logging()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(indicatorvalues_async_main(botdb_queue, table_created_event_5m_c, table_created_event_15m_c, table_created_event_1h_c))
    botdb_queue.put(None) 
    loop.close()
    time.sleep(3)    
    print("Indicator values aren't what they used to be_5m_close")