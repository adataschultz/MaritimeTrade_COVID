# -*- coding: utf-8 -*-
"""
@author: aschu
"""
###############################################################################
###############   Create Dataset for Company Exploration ######################
###############################################################################
import os
import random
import numpy as np
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

seed_value = 42
os.environ['MaritimeTrade'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

path = r'D:\MaritimeTrade\Data'
os.chdir(path)

# Read data
df = pd.read_csv('combined_trade.csv', low_memory=False)
df = df.drop_duplicates()

df = df.drop(['Foreign_Country', 'Foreign_Port', 'US_Port', 'Teus', 
              'Container_Type_Refrigerated', 'HS_Mixed', 
              'Foreign_Company_Country', 'carrier_size', 'US_Port_Clustered', 
              'Foreign_Company_Country_Continent', 'Foreign_Country_Continent',
              'Foreign_Company_Country_Region','Free_Trade_Agreement_with_US',
              'European_Union', 'Currency', 'Price', 'Time0_StateCase', 
              'cases_state_firstweek', 'Time0_StateDeath',
              'deaths_state_firstweek', 'DateTime_YearMonth'], axis=1)
df = df.drop_duplicates()
print(df.shape)

df.to_csv('combined_trade_companyExploration.csv', index=False)

###############################################################################
######################## Create sample data set  ##############################
###############################################################################
df_sample = df.sample(n=300000)
df_sample.to_csv('combined_trade_companyExploration_sample_3e5.csv', index=False)

###############################################################################
