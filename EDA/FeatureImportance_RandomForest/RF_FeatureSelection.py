# -*- coding: utf-8 -*-
"""
@author: aschu
"""
###############################################################################
############################## Random Forest ##################################
########################### Feature Importance ################################
###############################################################################
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from joblib import parallel_backend
from matplotlib import pyplot as plt
import pickle

# Set resolution of saved graphs
my_dpi = 96

seed_value = 42
os.environ['RF_featureSelection'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

# Input Path
path = r'D:\MaritimeTrade\Data'
os.chdir(path)

df = pd.read_csv('combined_trade_final.csv', low_memory = False)
df = df.drop_duplicates()

# =============================================================================
# Research Question 1: How has the composition and/or volume of maritime imports and exports from the US changed from 2010-2015
# =============================================================================

# Output Path
path = r'D:\Maritime_Trade\EDA\FeatureImportance_RandomForest\CompositionVolume_2010_2015'
os.chdir(path)

# Subset years for question
df1 = df.loc[df['Year'] < 2016]

print('\nDimensions of Question 1 EDA:', df1.shape) # (13760331, 33)
print('======================================================================')

# Drop time vars and not related to question
df1 = df1.drop(['DateTime', 'Year', 'DateTime_YearWeek', 'Date_Weekly_COVID', 
                'Date_Announced', 'Effective_Date', 'Trade_Direction', 
                'Free_Trade_Agreement_with_US', 'European_Union', 'Price', 
                'Currency', 'Foreign_Company_Country_Region', 'US_Port_State', 
                'US_Port_Clustered', 'cases_weekly', 'deaths_weekly'], axis=1)

# Remove rows with any column having NA/null for some important variables
df1 = df1[df1.Foreign_Country_Continent.notna() & 
          df1.Foreign_Country_Region.notna() & df1.Average_Tariff.notna()]

# Create dummy variables for categorical variables
df1 = pd.get_dummies(df1, drop_first=True)

###############################################################################
# Q1 all included vars
X = df1.drop('Metric_Tons', axis=1)
y = df1.Metric_Tons

# Create train test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, 
                                                    random_state=seed_value)

# RF Regressors
rf = RandomForestRegressor(n_estimators=100, random_state=seed_value)

# Train the regressor
with parallel_backend('threading', n_jobs=-1):
    rf.fit(X_train, y_train)

# Print the name and gini importance of each feature
df_rf = []
for feature in zip(X_train, rf.feature_importances_):
    df_rf.append(feature)
    
df_rf = pd.DataFrame(df_rf,columns=['Variable', 'Feature_Importance'])
df_rf = df_rf.sort_values('Feature_Importance', ascending = False)

df_rf.to_csv('q1_rf_featureimportance.csv', index=False, encoding='utf-8-sig')

# Save model
Pkl_Filename = 'CompositionVolume_2010_2015_RF.pkl'  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(rf, file)

# Identify and select most important features
plt.rcParams.update({'font.size': 7})
plt.figure(figsize=(27.841,10.195))
sorted_idx = rf.feature_importances_.argsort()
plt.barh(X.columns[sorted_idx], 
rf.feature_importances_[sorted_idx])
plt.xlabel('Random Forest Feature Importance')
plt.savefig('q1_rf_featureimportance.png', dpi=my_dpi*10)

###############################################################################
# Q1 - Remove Teus & TCVUSD 
df1 = df1.drop(['Teus', 'TCVUSD'], axis=1)

X = df1.drop('Metric_Tons', axis=1)
y = df1.Metric_Tons

# Create train test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, 
                                                    random_state=seed_value)

# Train the regressor
with parallel_backend('threading', n_jobs=-1):
    rf.fit(X_train, y_train)

# Print the name and gini importance of each feature
df_rf = []
for feature in zip(X_train, rf.feature_importances_):
    df_rf.append(feature)
    
df_rf = pd.DataFrame(df_rf,columns=['Variable', 'Feature_Importance'])
df_rf = df_rf.sort_values('Feature_Importance', ascending = False)

df_rf.to_csv('q1_rf_featureimportance_drop2vars.csv',
             index=False, encoding='utf-8-sig')

# Save model
Pkl_Filename = 'CompositionVolume_2010_2015_RF_Remove2Vars.pkl'

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(rf, file)
    
# Identify And Select Most Important Features
sorted_idx = rf.feature_importances_.argsort()
plt.barh(X.columns[sorted_idx], 
rf.feature_importances_[sorted_idx])
plt.xlabel('Random Forest Feature Importance')
plt.savefig('q1_rf_featureimportance_drop2vars.png', dpi=my_dpi*10)

del df1, X, y, X_train, X_test, y_train, y_test, rf, df_rf

###############################################################################
# =============================================================================
# Research Question 2: How did COVID-19 impact the volume and composition of international maritime trade?
# =============================================================================

# Output Path
path = r'D:\Maritime_Trade\EDA\FeatureImportance_RandomForest\COVID2020'
os.chdir(path)

# Subset year for question
df2 = df.loc[df['Year'] == 2020]
 
print('\nDimensions of Question 2 EDA:', df2.shape) #(3637621, 33)
print('======================================================================')

# Drop time vars and not related to question
df2 = df2.drop(['DateTime', 'Year', 'DateTime_YearWeek', 'Date_Weekly_COVID', 
                'Date_Announced','Effective_Date', 
                'Foreign_Company_Country_Region', 'US_Port_State', 
                                'US_Port_Clustered'], axis=1)

# Remove rows with any column having NA/null for some important variables
df2 = df2[df2.Foreign_Country_Continent.notna() & 
          df2.Foreign_Country_Region.notna() & df2.Average_Tariff.notna() & 
          df2.Price.notna() & df2.Currency.notna()]

# Dummy variables
df2 = pd.get_dummies(df2, drop_first=True)

###############################################################################
# Q2 all included vars
X = df2.drop('Metric_Tons', axis=1)
y = df2.Metric_Tons

# Train the regressor
with parallel_backend('threading', n_jobs=-1):
    rf.fit(X_train, y_train)

# Print the name and gini importance of each feature
df_rf = []
for feature in zip(X_train, rf.feature_importances_):
    df_rf.append(feature)
    
df_rf = pd.DataFrame(df_rf,columns=['Variable', 'Feature_Importance'])
df_rf = df_rf.sort_values('Feature_Importance', ascending = False)

df_rf.to_csv('q2_rf_featureimportance.csv', index=False, 
              encoding='utf-8-sig')

# Save model
Pkl_Filename = 'COVID2020_RF.pkl'  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(rf, file)
    
# Identify And Select Most Important Features
sorted_idx = rf.feature_importances_.argsort()
plt.barh(X.columns[sorted_idx], 
rf.feature_importances_[sorted_idx])
plt.xlabel('Random Forest Feature Importance')
plt.savefig('q2_rf_featureimportance.png', dpi=my_dpi*10)

###############################################################################
# Q2 - Remove Teus & TCVUSD  
df2 = df2.drop(['Teus', 'TCVUSD'], axis=1)

X = df2.drop('Metric_Tons', axis=1)
y = df2.Metric_Tons

# Train the regressor
with parallel_backend('threading', n_jobs=-1):
    rf.fit(X_train, y_train)

# Print the name and gini importance of each feature
df_rf = []
for feature in zip(X_train, rf.feature_importances_):
    df_rf.append(feature)
    
df_rf = pd.DataFrame(df_rf,columns=['Variable', 'Feature_Importance'])
df_rf = df_rf.sort_values('Feature_Importance', ascending = False)

df_rf.to_csv('q2_rf_featureimportance_drop2vars.csv', index=False,
              encoding='utf-8-sig')

# Save model
Pkl_Filename = 'COVID2020_RF_Remove2Vars.pkl' 

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(rf, file)
    
# Identify And Select Most Important Features
sorted_idx = rf.feature_importances_.argsort()
plt.barh(X.columns[sorted_idx], 
rf.feature_importances_[sorted_idx])
plt.xlabel('Random Forest Feature Importance')
plt.savefig('q2_rf_featureimportance_drop2vars_.png', dpi=my_dpi*10)

del df2, X, y, X_train, X_test, y_train, y_test, rf, df_rf

###############################################################################
# =============================================================================
# Research Question 3: Are there any confounding effects impacting the volume of imports and exports of the targeted commodities?
# =============================================================================

# Output Path
path = r'D:\Maritime_Trade\EDA\FeatureImportance_RandomForest\Confounding'
os.chdir(path)

# Drop time vars and not related to question
df3 = df.drop(['DateTime', 'Year', 'DateTime_YearWeek', 'Date_Weekly_COVID', 
               'Date_Announced', 'Effective_Date', 
               'Foreign_Company_Country_Region', 'US_Port_State', 
               'US_Port_Clustered'], axis=1)

# Remove rows with any column having NA/null for some important variables
df3 = df3[df3.Foreign_Country_Continent.notna() & 
          df3.Foreign_Country_Region.notna() & df3.Average_Tariff.notna() &
          df3.Price.notna() & df3.Currency.notna()]

print('\nDimensions of Question 3 EDA:', df3.shape)  #(27817027, 24)
print('======================================================================')

# Dummy variables
df3 = pd.get_dummies(df3, drop_first=True)

###############################################################################
# Q3 all included vars
X = df3.drop('Metric_Tons', axis=1)
y = df3.Metric_Tons

# Train the regressor
with parallel_backend('threading', n_jobs=-1):
    rf.fit(X_train, y_train)

# Print the name and gini importance of each feature
df_rf = []
for feature in zip(X_train, rf.feature_importances_):
    df_rf.append(feature)
    
df_rf = pd.DataFrame(df_rf,columns=['Variable', 'Feature_Importance'])
df_rf = df_rf.sort_values('Feature_Importance', ascending = False)

df_rf.to_csv('q3_rf_featureimportance.csv', index=False, 
              encoding='utf-8-sig')

# Save model
Pkl_Filename = 'Confounding_RF.pkl'  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(rf, file)
    
# Identify And Select Most Important Features
sorted_idx = rf.feature_importances_.argsort()
plt.barh(X.columns[sorted_idx], 
rf.feature_importances_[sorted_idx])
plt.xlabel('Random Forest Feature Importance')
plt.savefig('q3_rf_featureimportance.png', dpi=my_dpi*10)

###############################################################################
# Q3 - Remove Teus & TCVUSD 
df3 = df3.drop(['Teus', 'TCVUSD'], axis=1)

X = df3.drop('Metric_Tons', axis=1)
y = df3.Metric_Tons

# Train the regressor
with parallel_backend('threading', n_jobs=-1):
    rf.fit(X_train, y_train)

# Print the name and gini importance of each feature
df_rf = []
for feature in zip(X_train, rf.feature_importances_):
    df_rf.append(feature)
    
df_rf = pd.DataFrame(df_rf,columns=['Variable', 'Feature_Importance'])
df_rf = df_rf.sort_values('Feature_Importance', ascending = False)

df_rf.to_csv('q3_rf_featureimportance_drop2vars.csv', index=False,
              encoding='utf-8-sig')

# Save model
Pkl_Filename = 'Confounding_RF_Remove2Vars.pkl'  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(rf, file)
    
# Identify And Select Most Important Features
sorted_idx = rf.feature_importances_.argsort()
plt.barh(X.columns[sorted_idx], 
rf.feature_importances_[sorted_idx])
plt.xlabel('Random Forest Feature Importance')
plt.savefig('q3_rf_featureimportance_drop2vars.png', dpi=my_dpi*10)

###############################################################################