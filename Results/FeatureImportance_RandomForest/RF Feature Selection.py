# -*- coding: utf-8 -*-
"""
@author: aschu
"""

print('\nMaritime Trade - Feature Importance using Random Forest') 
print("==========================================================")

import random
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from joblib import parallel_backend
from matplotlib import pyplot as plt
import pickle

##############################################################################
# Input Path
path = r'D:\MaritimeTrade\Data'
os.chdir(path)

df = pd.read_csv('combined_trade.csv', low_memory=False)
df = df.drop_duplicates()

# Output Path
path = r'D:\Maritime_Trade\Results\FeatureImportance_RandomForest'
os.chdir(path)

# =============================================================================
# Research Question 1: How has the composition and/or volume of maritime imports and exports from the US changed from 2010-2015
# 
# 1. Filter to 2010-2015
# 2. Drop variables not needed for question or missingnes
#     a. Time vars
#     b. Associated with COVID-19
#     c.'US_Port_State' and 'US_Port_Clustered' - too many levels
#     d. 'Foreign_Company_Country_Region' since 'Foreign_Country_Region' is more useful
#     e. 'Price' and 'Currency' not used for composition
#     f. Foreign_Country_Continent' has missing so will use Foreign_Company_Country_Continent instead
#     g. 'Free_Trade_Agreement_with_US', 'European_Union' as well
# 
# 3. Procees with selecting non missing & create dummy variables
# 4. Create sets for all included vars and all included without Teus and TCVUSD since associated with metric tonnage
# 5. Fit random forest regression models to examine feature importance on metric tonnage
# =============================================================================

# Subset years for question
df1 = df.loc[df['Year'] < 2016]

print('\nDimensions of Question 1 EDA:', df1.shape) # (13760331, 33)
print('======================================')

# Drop time vars
df1 = df1.drop(['DateTime', 'Year', 'DateTime_YearWeek', 'Date_Weekly_COVID', 
                'Date_Announced',
              'Effective_Date'], axis=1)

# Drop vars not useful for question
df1 = df1.drop(['Trade_Direction', 'Free_Trade_Agreement_with_US', 
                'European_Union', 'Price', 'Currency', 
                'Foreign_Company_Country_Region', 'US_Port_State', 
                'US_Port_Clustered', 'cases_weekly', 'deaths_weekly'], axis=1)

##############################################################################
# Remove rows with any column having NA/null for some important variables
df1 = df1[df1.Foreign_Country_Continent.notna() & 
          df1.Foreign_Country_Region.notna() & df1.Average_Tariff.notna()]

# Create dummy variables for categorical variables
df1 = pd.get_dummies(df1, drop_first=True)

##############################################################################
# Q1 all included vars
X = df1.drop('Metric_Tons', axis=1)
y = df1.Metric_Tons

##############################################################################
# Remove 2 vars: Teus & TCVUSD
df11 = df1.drop(['Teus', 'TCVUSD'], axis=1)

X1 = df11.drop('Metric_Tons', axis=1)
y1 = df11.Metric_Tons

##############################################################################
# Create train test split 
# Q1 - All vars
X_train, X_test, 
y_train, y_test = train_test_split(X, y, test_size = 0.25, 
                                   train_size = 0.75, random_state=42)

#Q1 - Remove 2 vars: Teus & TCVUSD
X1_train, X1_test, 
y1_train, y1_test = train_test_split(X, y1, test_size = 0.25, 
                                     train_size = 0.75, random_state=42)

##############################################################################
# Q1: Random Forest Regressors

# RF - Q1 
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the regressor
with parallel_backend('threading', n_jobs=-1):
    rf.fit(X_train, y_train)

# Print the name and gini importance of each feature
df_rf = []
for feature in zip(X_train, rf.feature_importances_):
    df_rf.append(feature)
    
df_rf = pd.DataFrame(df_rf,columns=['Variable', 'Feature_Importance'])
df_rf = df_rf.sort_values('Feature_Importance', ascending = False)

df_rf.to_csv('q1_rf_featureimportance.csv', index=False, 
             encoding='utf-8-sig')

# Save model
Pkl_Filename = "RF_Model_Q1.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(rf, file)

##############################################################################
# Change picture resolution
my_dpi=96
plt.rcParams.update({'font.size': 7})

# Identify And Select Most Important Features
plt.figure(figsize=(27.841,10.195), dpi=100)
sorted_idx = rf.feature_importances_.argsort()
plt.barh(X.columns[sorted_idx], 
rf.feature_importances_[sorted_idx])
plt.xlabel("Random Forest Feature Importance")
plt.savefig('q1_rf_featureimportance.png', dpi=my_dpi * 10)

##############################################################################
##############################################################################
# RF - Q1 - Remove Teus & TCVUSD 
rf1 = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the regressor
with parallel_backend('threading', n_jobs=-1):
    rf1.fit(X1_train, y1_train)

# Print the name and gini importance of each feature
df_rf1 = []
for feature in zip(X1_train, rf1.feature_importances_):
    df_rf1.append(feature)
    
df_rf1 = pd.DataFrame(df_rf1,columns=['Variable', 'Feature_Importance'])
df_rf1 = df_rf1.sort_values('Feature_Importance', ascending = False)

df_rf1.to_csv('q1_rf_featureimportance_drop2vars.csv', index=False, 
              encoding='utf-8-sig')

# Save model
Pkl_Filename = "RF_Model_Q1_Remove2Vars.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(rf1, file)
    
##############################################################################
# Change picture resolution
my_dpi=96
plt.rcParams.update({'font.size': 7})

# Identify And Select Most Important Features
# Remove 2 vars: Teus & TCVUSD
plt.figure(figsize=(27.841,10.195), dpi=100)
sorted_idx = rf1.feature_importances_.argsort()
plt.barh(X1.columns[sorted_idx], 
rf1.feature_importances_[sorted_idx])
plt.xlabel("Random Forest Feature Importance")
plt.savefig('q1_rf featureimportance_drop2vars.png', dpi=my_dpi * 10)

##############################################################################
##############################################################################
##############################################################################
# =============================================================================
# Research Question 2: How did COVID-19 impact the volume and composition of international maritime trade?
# 
# 1. Filter to 2020 due to COVID-19
# 2. Drop variables not needed for question or missingnes
#     a. Time vars
#     b.'US_Port_State' and 'US_Port_Clustered' - too many levels
#     c. 'Foreign_Company_Country_Region' since 'Foreign_Country_Region' is more useful
#     d. Foreign_Country_Continent' has missing so will use Foreign_Company_Country_Continent instead
# 
# 3. Procees with selecting non missing & create dummy variables
# 4. Create sets for all included vars and all included without Teus and TCVUSD since associated with metric tonnage
# 5. Fit random forest regression models to examine feature importance on metric tonnage
# =============================================================================

# Subset year for question
df2 = df.loc[df['Year'] == 2020]
 
print('\nDimensions of Question 2 EDA:', df2.shape) #(3637621, 33)
print('======================================')

# Drop time vars
df2 = df2.drop(['DateTime', 'Year', 'DateTime_YearWeek', 'Date_Weekly_COVID', 
                'Date_Announced','Effective_Date'], axis=1)

# Drop vars not useful for question
df2 = df2.drop(['Foreign_Company_Country_Region', 'US_Port_State', 
                'US_Port_Clustered'], axis=1)

# Remove rows with any column having NA/null for some important variables
df2 = df2[df2.Foreign_Country_Continent.notna() & 
          df2.Foreign_Country_Region.notna() & df2.Average_Tariff.notna() & 
          df2.Price.notna() & df2.Currency.notna()]

# Dummy variables
df2 = pd.get_dummies(df2, drop_first=True)

##############################################################################
# Q1 all included vars
X2 = df2.drop('Metric_Tons', axis=1)
y2 = df2.Metric_Tons

##############################################################################
# Remove 2 vars: Teus & TCVUSD
df12 = df2.drop(['Teus', 'TCVUSD'], axis=1)

X3 = df12.drop('Metric_Tons', axis=1)
y3 = df12.Metric_Tons

##############################################################################
# Create train test split 
X2_train, X2_test, 
y2_train, y2_test = train_test_split(X2, y2, test_size = 0.25, 
                                     train_size = 0.75, random_state=42)

# Remove 2 vars: Teus & TCVUSD
X3_train, X3_test, 
y3_train, y3_test = train_test_split(X3, y3, test_size = 0.25, 
                                      train_size = 0.75, random_state=42)

##############################################################################
# Q2: Random Forest regressors

# RF - Q2
rf2 = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the regressor
with parallel_backend('threading', n_jobs=-1):
    rf2.fit(X2_train, y2_train)

# Print the name and gini importance of each feature
df_rf2 = []
for feature in zip(x2_train, rf2.feature_importances_):
    df_rf2.append(feature)
    
df_rf2 = pd.DataFrame(df_rf2,columns=['Variable', 'Feature_Importance'])
df_rf2 = df_rf2.sort_values('Feature_Importance', ascending = False)

df_rf2.to_csv('q2_rf_featureimportance.csv', index=False, 
              encoding='utf-8-sig')

# Save model
Pkl_Filename = "RF_Model_Q2.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(rf2, file)
    
##############################################################################
# Change picture resolution
my_dpi=96
plt.rcParams.update({'font.size': 7})

# Identify And Select Most Important Features
plt.figure(figsize=(27.841,10.195))
sorted_idx = rf2.feature_importances_.argsort()
plt.barh(X2.columns[sorted_idx], 
rf2.feature_importances_[sorted_idx])
plt.xlabel("Random Forest Feature Importance")
plt.savefig('q2_rf_featureimportance.png', dpi=my_dpi * 10 )

##############################################################################
##############################################################################
# RF - Q2 - Remove Teus & TCVUSD  
rf3 = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the regressor
with parallel_backend('threading', n_jobs=-1):
    rf3.fit(X3_train, y3_train)

# Print the name and gini importance of each feature
df_rf3 = []
for feature in zip(X3_train, rf3.feature_importances_):
    df_rf3.append(feature)
    
df_rf3 = pd.DataFrame(df_rf3,columns=['Variable', 'Feature_Importance'])
df_rf3 = df_rf3.sort_values('Feature_Importance', ascending = False)

df_rf3.to_csv('q2_rf_featureimportance_drop2vars.csv', index=False,
              encoding='utf-8-sig')

# Save model
Pkl_Filename = "RF_Model_Q2_Remove2Vars.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(rf3, file)
    
##############################################################################
# Change picture resolution
my_dpi=96
plt.rcParams.update({'font.size': 7})

# Identify And Select Most Important Features
# Remove 2 vars: Teus & TCVUSD
plt.figure(figsize=(27.841,10.195))
sorted_idx = rf3.feature_importances_.argsort()
plt.barh(X3.columns[sorted_idx], 
rf3.feature_importances_[sorted_idx])
plt.xlabel("Random Forest Feature Importance")
plt.savefig('q2_rf_featureimportance_drop2vars_.png', dpi=my_dpi * 10 )

##############################################################################
##############################################################################
##############################################################################
# =============================================================================
# Research Question 3: Are there any confounding effects impacting the volume of imports and exports of the targeted commodities?
# 
# 
# 1. Drop variables not needed for question or missingnes
#     a. Time vars
#     b.'US_Port_State' and 'US_Port_Clustered' - too many levels
#     c. 'Foreign_Company_Country_Region' since 'Foreign_Country_Region' is more useful
#     d. Foreign_Country_Continent' has missing so will use Foreign_Company_Country_Continent instead
# 
# 2. Procees with selecting non missing & create dummy variables
# 3. Create sets for all included vars and all included without Teus and TCVUSD since associated with metric tonnage
# 4. Fit random forest regression models to examine feature importance on metric tonnage
# =============================================================================

# Drop time vars
df3 = df.drop(['DateTime', 'Year', 'DateTime_YearWeek', 'Date_Weekly_COVID', 
               'Date_Announced', 'Effective_Date'], axis=1)

# Drop vars not useful for question
df3 = df3.drop(['Foreign_Company_Country_Region', 'US_Port_State', 
                'US_Port_Clustered'], axis=1)

# Remove rows with any column having NA/null for some important variables
df3 = df3[df3.Foreign_Country_Continent.notna() & 
          df3.Foreign_Country_Region.notna() & df3.Average_Tariff.notna() &
          df3.Price.notna() & df3.Currency.notna()]

print('\nDimensions of Question 3 EDA:', df3.shape)  #(27817027, 24)
print('===============================================')

# Dummy variables
df3 = pd.get_dummies(df3, drop_first=True)

##############################################################################
# Q3 all included vars
X4 = df3.drop('Metric_Tons', axis=1)
y4 = df3.Metric_Tons

##############################################################################
# Remove 2 vars: Teus & TCVUSD
df13 = df3.drop(['Teus', 'TCVUSD'], axis=1)

X5 = df13.drop('Metric_Tons', axis=1)
y5 = df13.Metric_Tons

##############################################################################
# Create train test split 
X4_train, X4_test, 
y4_train, y4_test = train_test_split(X4, y4, test_size = 0.25, 
                                     train_size = 0.75, random_state=42)

# Remove 2 vars: Teus & TCVUSD
X5_train, X5_test, 
y5_train, y5_test = train_test_split(X5, y5, test_size = 0.25, 
                                      train_size = 0.75, random_state=42)

##############################################################################
# Q3: Random Forest regressors

# RF - Q3
rf4 = RandomForestRegressor(n_estimators=50, random_state=42)

# Train the regressor
with parallel_backend('threading', n_jobs=8):
    rf4.fit(X4_train, y4_train)

# Print the name and gini importance of each feature
df_rf4 = []
for feature in zip(X4_train, rf4.feature_importances_):
    df_rf4.append(feature)
    
df_rf4 = pd.DataFrame(df_rf4,columns=['Variable', 'Feature_Importance'])
df_rf4 = df_rf4.sort_values('Feature_Importance', ascending = False)

df_rf4.to_csv('q3_rf_featureimportance.csv', index=False, 
              encoding='utf-8-sig')

# Save model
Pkl_Filename = "RF_Model_Q3.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(rf4, file)
    
##############################################################################
# Change picture resolution
my_dpi=96
plt.rcParams.update({'font.size': 7})

# Identify And Select Most Important Features
# Drop 2 vars and rerun
from matplotlib import pyplot as plt
plt.figure(figsize=(27.841,10.195), dpi=100)
sorted_idx = rf4.feature_importances_.argsort()
plt.barh(X4.columns[sorted_idx], 
rf4.feature_importances_[sorted_idx])
plt.xlabel("Random Forest Feature Importance")
plt.savefig('q3_rf_featureimportance.png', dpi=my_dpi * 10)

##############################################################################
##############################################################################
# RF - Q3 - Remove Teus & TCVUSD 
rf5 = RandomForestRegressor(n_estimators=50, random_state=42)

# Train the regressor
with parallel_backend('threading', n_jobs=8):
    rf5.fit(X5_train, y5_train)

# Print the name and gini importance of each feature
df_rf5 = []
for feature in zip(X5_train, rf5.feature_importances_):
    df_rf5.append(feature)
    
df_rf5 = pd.DataFrame(df_rf5,columns=['Variable', 'Feature_Importance'])
df_rf5 = df_rf5.sort_values('Feature_Importance', ascending = False)

df_rf5.to_csv('q3_rf_featureimportance_drop2vars.csv', index=False,
              encoding='utf-8-sig')

# Save model
Pkl_Filename = "RF_Model_Q3_Remove2Vars.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(rf5, file)
    
##############################################################################
# Change picture resolution
my_dpi=96
plt.rcParams.update({'font.size': 7})

# Identify And Select Most Important Features
# Drop 2 vars and rerun
from matplotlib import pyplot as plt
plt.figure(figsize=(27.841,10.195), dpi=100)
sorted_idx = rf5.feature_importances_.argsort()
plt.barh(X5.columns[sorted_idx], 
rf5.feature_importances_[sorted_idx])
plt.xlabel("Random Forest Feature Importance")
plt.savefig('q3_rf_featureimportance_drop2vars.png', dpi=my_dpi * 10)

##############################################################################


