"""
@author: aschu
"""
###############################################################################
########################## Regression - Linear ################################
########################## Lasso & Elastic Net ################################
###############################################################################
import os
import random
import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import joblib
from joblib import parallel_backend
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
#import itertools
import time
import pickle
import sklearn
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
#from sklearn.metrics import r2_score
import matplotlib as plt
from matplotlib import pyplot
import eli5 as eli
from eli5.sklearn import PermutationImportance 
from eli5 import show_weights
import webbrowser
from eli5.sklearn import explain_weights_sklearn
from eli5.formatters import format_as_dataframe, format_as_dataframes
from eli5 import show_prediction
import lime
from lime import lime_tabular

# Set resolution of saved graphs
my_dpi = 96

seed_value = 42
os.environ['linear_train19test20'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

# Input Path
path = r'D:\MaritimeTrade\Data'
os.chdir(path)    

# Read in data
df = pd.read_csv('combined_trade_final.csv', low_memory = False)
df = df.drop_duplicates()

# Output Path
path = r'D:\MaritimeTrade\Results\Linear'
os.chdir(path)

# Filter the df to 2019 & 2020
df = df[df['Year'] == 2019]
df1 = df[df['Year'] == 2020]

# Drop time and COVID-19 vars
df = df.drop(['DateTime', 'DateTime_YearWeek', 'Date_Weekly_COVID', 
              'Year', 'State'], axis=1)
# Remove missing data - COVID
df = df[df.Foreign_Country_Region.notna() & df.Average_Tariff.notna() 
        & df.State_Closure_EA_Diff.notna()]

# Reformat for X, y
df2 = df.drop(['Metric_Tons'], axis=1)
df3 = df.loc[:, ['Metric_Tons']]
df = pd.concat([df3, df2], axis=1)

del df2, df3

# Set up for partitioning data
X = df.iloc[:,:-1]
y = df.iloc[:,:1]

# Encode variables using ranking - ordinal 
ce_ord = ce.OrdinalEncoder(cols = ['foreign_company_size', 'US_company_size'])
X = ce_ord.fit_transform(X, y['Metric_Tons'])

###############################################################################
########################## COVID: Train 2019 - Test 2020 ######################
###############################################################################
# Set up training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, 
                                                    random_state=seed_value)

# Design pipeline
numeric_features = list(X.columns[X.dtypes != 'object'])
numeric_transformer = Pipeline(steps=[
    ('mn', MinMaxScaler())])

categorical_features = X.select_dtypes(include=['object', 'bool']).columns

t = [('cat', OneHotEncoder(sparse=False,  drop='first'),
      categorical_features), ('num', MinMaxScaler(), numeric_features)]

preprocessor = ColumnTransformer(transformers=t)

pipe_lasso = Pipeline([('preprocessor', preprocessor),
                       ('clf', Lasso(random_state=seed_value))])

grid_params_lasso = [{'clf__max_iter': [10000, 50000, 100000], 
                      'clf__alpha': [0.000001, 0.00001, 0.0001, 0.001, 0.01,
                                     0.1, 1, 10, 100]}] 

LR = GridSearchCV(estimator=pipe_lasso, param_grid=grid_params_lasso,
            scoring='neg_mean_absolute_error', cv=3, n_jobs=-1) 

pipe_elnet = Pipeline([('preprocessor', preprocessor),
                       ('clf1', ElasticNet(random_state=seed_value))])

grid_params_elnet = [{'clf1__max_iter': [100000, 500000, 10000000], 
                      'clf1__alpha': [0.000001, 0.00001, 0.0001, 0.001, 0.01,
                                      0.1, 1, 10, 100]}] 

ELNETR = GridSearchCV(estimator=pipe_elnet,
            param_grid=grid_params_elnet,
            scoring='neg_mean_absolute_error',
            cv=3, n_jobs=-1) 

grids = [LR, ELNETR]

grid_dict = {0: 'Lasso', 1: 'Elastic Net'}

# Fit the parameters for the grid search 
print('Performing Performing Linear HPO for 2019...')
search_time_start = time.time()
best_mae = 1000000
best_reg = 0
best_gs = ''
for idx, gs in enumerate(grids):
    print('\nEstimator: %s' % grid_dict[idx])
    with parallel_backend('threading', n_jobs=-1):
        gs.fit(X_train, y_train)
        print('Best params are : %s' % gs.best_params_)
    # Best training data mae
        print('Best training mae: %.3f' % gs.best_score_)
    # Predict on test data with best params
        y_pred = gs.predict(X_test)
    # Test data mae of model with best params
        print('Test set mae for best params: %.3f ' % mean_absolute_error(y_test,
                                                                          y_pred))
    # Track best (lowest test mae) model
        if mean_absolute_error(y_test, y_pred) < best_mae:
            best_mae = mean_absolute_error(y_test, y_pred)
            best_gs = gs
            best_reg = idx
print('\nRegressor with best test set mae: %s' % grid_dict[best_reg])
print('Finished fit the best hyperparameters from Lasso/Elastic Net grid search to the data:',
      time.time() - search_time_start)
print('======================================================================')

# Fit model with best MAE
Trade_Linear_HPO = Lasso(alpha=0.001, max_iter=10000, random_state=seed_value)

preprocessor = ColumnTransformer(transformers=t)

Trade_Linear_HPO = Pipeline([('preprocessor', preprocessor),
                       ('clf', Lasso(alpha=0.001, max_iter=10000, 
                                     random_state=seed_value))])

grid_params_lasso = [{'clf__max_iter': [10000], 
                      'clf__alpha': [0.000001]}] 

LR = GridSearchCV(estimator=pipe_lasso,
            param_grid=grid_params_lasso,
            scoring='neg_mean_absolute_error',
            cv=3, n_jobs=-1) 

print('Start fit the best hyperparameters from Lasso/Elastic Net grid search to the data..')
search_time_start = time.time()
with parallel_backend('threading', n_jobs=-1):
    Trade_Linear_HPO.fit(X_train, y_train)
print('Finished fit the best hyperparameters from Lasso/Elastic Net grid search to the data:',
      time.time() - search_time_start)
print('======================================================================')

# Save model
Pkl_Filename = 'MaritimeTrade_Linear_2019gridSearch.pkl'  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(Trade_Linear_HPO, file)

# =============================================================================
# # To load saved model
# Trade_Linear_HPO = joblib.load('MaritimeTrade_Linear_2019gridSearch.pkl')
# print(Trade_Linear_HPO)
# =============================================================================

###############################################################################
# Fit best parameters on model
# Create dummy variables for categorical variables
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# MinMax Scaling
mn = MinMaxScaler()
X_train = pd.DataFrame(mn.fit_transform(X_train))
X_test = pd.DataFrame(mn.fit_transform(X_test))

# Best model parameters for HPO for 2019 data
Trade_Linear_HPO = Lasso(alpha=0.001, max_iter=10000, random_state=seed_value)

print('Start fit the best hyperparameters from Lasso/Elastic Net grid search to 2019 data..')
search_time_start = time.time()
with parallel_backend('threading', n_jobs=-1):
    Trade_Linear_HPO.fit(X_train, y_train)
print('Finished fit the best hyperparameters from Lasso/Elastic Net grid search to 2019 data:',
      time.time() - search_time_start)
print('======================================================================')

# Save model
Pkl_Filename = 'MaritimeTrade_Linear_HPO_2019.pkl'  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(Trade_Linear_HPO, file)

# =============================================================================
# # To load saved model
# Trade_Linear_HPO = joblib.load('MaritimeTrade_Linear_HPO_2019.pkl')
# print(Trade_Linear_HPO)
# =============================================================================

###############################################################################
# Predict based on training 
y_pred_Trade_Linear_HPO = Trade_Linear_HPO.predict(X_test)

print('Results from LASSO using 2019 HPO:')
print('\n')

# Calculate MAE/MSE
print('Mean Absolute Error (MAE): %3f' % sklearn.metrics.mean_absolute_error(y_pred_Trade_Linear_HPO,
                                                                             y_test))
print('Mean Square Error (MSE): %3f' % sklearn.metrics.mean_squared_error(y_pred_Trade_Linear_HPO,
                                                                          y_test))

# Plot actual vs predicted metric tonnage
plt.rcParams['agg.path.chunksize'] = 10000
pyplot.plot(y_test, label = '2019')
pyplot.plot(y_pred_Trade_Linear_HPO, label = '2019 - predicted')
pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
pyplot.savefig('Train19_HPO_Test2019_Predict2019.png', dpi=my_dpi, 
               bbox_inches='tight')
pyplot.show()

# Model metrics with Eli5
# Compute permutation feature importance
perm_importance = PermutationImportance(Trade_Linear_HPO,
                                        random_state=seed_value).fit(X_test,
                                                                     y_test)

# Create dummy variables for feature names 
X = pd.get_dummies(X, drop_first=True)

# Store feature weights in an object
html_obj = eli.show_weights(perm_importance,
                            feature_names = X.columns.tolist())

# Write feature weights html object to a file 
with open('D:\MaritimeTrade\Results\Linear\Trade_Linear_HPO_2019_WeightsFeatures.htm',
          'wb') as f:
    f.write(html_obj.data.encode("UTF-8"))

# Open the stored feature weights HTML file
url = r'D:\MaritimeTrade\Results\Linear\Trade_Linear_HPO_2019_WeightsFeatures.htm'
webbrowser.open(url, new=2)

# Explain weights
explanation = eli.explain_weights_sklearn(perm_importance,
                            feature_names = X.columns.tolist())
exp = format_as_dataframe(explanation)

# Write processed data to csv
exp.to_csv('MaritimeTrade_Linear_HPO_2019_WeightsExplain.csv',
           index=False, encoding='utf-8-sig')

# Show prediction
html_obj2 = show_prediction(Trade_Linear_HPO, X.iloc[1],
                            show_feature_values=True)

# Write show prediction html object to a file 
with open('D:\MaritimeTrade\Results\Linear\Trade_Linear_HPO_2019_Prediction.htm',
          'wb') as f:
    f.write(html_obj2.data.encode("UTF-8"))

# Open the show prediction stored HTML file
url2 = r'D:\MaritimeTrade\Results\Linear\Trade_Linear_HPO_2019_Prediction.htm'
webbrowser.open(url2, new=2)

# Explain prediction
#explanation_pred = eli.explain_prediction(Trade_Linear_HPO, np.array(X_test)[1])
#explanation_pred

###############################################################################
# LIME for model explanation
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X.columns,
    class_names=['Metric_Tons'],
    mode='regression')

exp = explainer.explain_instance(
    data_row=X.iloc[1], 
    predict_fn=Trade_Linear_HPO.predict)

exp.save_to_file('Trade_Linear_HPO_2019_LIME.html')

###############################################################################
# Test 2019 HPO model on 2020 data
# Drop time and COVID-19 vars
df1 = df1.drop(['DateTime', 'DateTime_YearWeek', 'Date_Weekly_COVID', 'Year',
                'State'], axis=1)

# Remove missing data 
df1 = df1[df1.Foreign_Country_Region.notna() & df1.Average_Tariff.notna() 
          & df1.State_Closure_EA_Diff.notna()]

# Reformat for X, y
df2 = df1.drop(['Metric_Tons'], axis=1)
df3 = df1.loc[:, ['Metric_Tons']]
df1 = pd.concat([df3, df2], axis=1)

del df2, df3

# Set up for partitioning data
X1 = df1.iloc[:,:-1]
y1 = df1.iloc[:,:1]

# Encode variables using ranking - ordinal 
ce_ord = ce.OrdinalEncoder(cols = ['foreign_company_size', 'US_company_size'])
X1 = ce_ord.fit_transform(X1, y1['Metric_Tons'])

# Create dummy variables for categorical variables
X1 = pd.get_dummies(X1, drop_first=True)

# MinMax Scaling
X1 = pd.DataFrame(mn.fit_transform(X1))

print('Start fit the best hyperparameters from 2019 Lasso/Elastic Net grid search to 2020 data..')
search_time_start = time.time()
with parallel_backend('threading', n_jobs=-1):
    Trade_Linear_HPO.fit(X1, y1)
print('Finished fit the best hyperparameters from 2019 Lasso/Elastic Net grid search to 2020 data:',
      time.time() - search_time_start)
print('======================================================================')

# Save model
Pkl_Filename = 'MaritimeTrade_Linear_HPO_2019_2020data.pkl'  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(Trade_Linear_HPO, file)

# =============================================================================
# # To load saved model
# Trade_Linear_HPO = joblib.load('MaritimeTrade_Linear_HPO_2019_2020data.pkl')
# print(Trade_Linear_HPO)
# =============================================================================

###############################################################################
# Predict based on training 
y_pred_Trade_Linear_HPO1 = Trade_Linear_HPO.predict(X1)

print('Results from LASSO using HPO on 2020 data:')
print('\n')

# Calculate MAE/MSE
print('Mean Absolute Error (MAE): %3f' % sklearn.metrics.mean_absolute_error(y_pred_Trade_Linear_HPO1,
                                                                             y1))
print('Mean Square Error (MSE): %3f' % sklearn.metrics.mean_squared_error(y_pred_Trade_Linear_HPO1, 
                                                                          y1))

# Plot actual vs predicted metric tonnage
pyplot.plot(y, label = '2019')
pyplot.plot(y_pred_Trade_Linear_HPO1, label = '2020 - predicted')
pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
pyplot.savefig('Train19_HPO_Predict2020.png', dpi=my_dpi, bbox_inches='tight')
pyplot.show()

# Plot actual vs predicted metric tonnage
pyplot.plot(y1, label = '2020')
pyplot.plot(y_pred_Trade_Linear_HPO1, label = '2020 - predicted')
pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
pyplot.savefig('Train19_HPO_Test2020_Predict2020.png', dpi=my_dpi, 
               bbox_inches='tight')
pyplot.show()

# Model metrics with Eli5
# Compute permutation feature importance
perm_importance = PermutationImportance(Trade_Linear_HPO,
                                        random_state=seed_value).fit(X1, y1)

# Store feature weights in an object
html_obj = eli.show_weights(perm_importance,
                            feature_names = X.columns.tolist())

# Write feature weights html object to a file 
with open('D:\MaritimeTrade\Results\Linear\Trade_Linear_HPO_2019_2020data_WeightsFeatures.htm',
          'wb') as f:
    f.write(html_obj.data.encode("UTF-8"))

# Open the stored feature weights HTML file
url = r'D:\MaritimeTrade\Results\Linear\Trade_Linear_HPO_2019_2020data_WeightsFeatures.htm'
webbrowser.open(url, new=2)

# Explain weights
explanation = eli.explain_weights_sklearn(perm_importance,
                            feature_names = X.columns.tolist())
exp = format_as_dataframe(explanation)

# Write processed data to csv
exp.to_csv('MaritimeTrade_Linear_HPO_2019_2020data_WeightsExplain.csv',
           index=False, encoding='utf-8-sig')

# Show prediction
html_obj2 = show_prediction(Trade_Linear_HPO, X.iloc[1],
                            show_feature_values=True)

# Write show prediction html object to a file 
with open('D:\MaritimeTrade\Results\Linear\Trade_Linear_HPO_2019_2020data_Prediction.htm',
          'wb') as f:
    f.write(html_obj2.data.encode("UTF-8"))

# Open the show prediction stored HTML file
url2 = r'D:\MaritimeTrade\Results\Linear\Trade_Linear_HPO_2019_2020data_Prediction.htm'
webbrowser.open(url2, new=2)

# Explain prediction
#explanation_pred = eli.explain_prediction(Trade_Linear_HPO, np.array(X1)[1])
#explanation_pred

###############################################################################
# LIME for model explanation
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X1),
    feature_names=X.columns,
    class_names=['Metric_Tons'],
    mode='regression')

exp = explainer.explain_instance(
    data_row=X1.iloc[1], 
    predict_fn=Trade_Linear_HPO.predict)

exp.save_to_file('Trade_Linear_HPO_2019_2020data_LIME.html')

###############################################################################
########################## COVID: Train 2020 - Test 2019 ######################
###############################################################################
# Set up for partitioning data
X = df1.iloc[:,:-1]
y = df1.iloc[:,:1]

# Encode variables using ranking - ordinal 
ce_ord = ce.OrdinalEncoder(cols = ['foreign_company_size', 'US_company_size'])
X = ce_ord.fit_transform(X, y['Metric_Tons'])

###############################################################################
# Set up training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, 
                                                    random_state=seed_value)

# Design pipeline
numeric_features = list(X.columns[X.dtypes != 'object'])
numeric_transformer = Pipeline(steps=[
    ('mn', MinMaxScaler())])

categorical_features = X.select_dtypes(include=['object', 'bool']).columns

t = [('cat', OneHotEncoder(sparse=False,  drop='first'),
      categorical_features), ('num', MinMaxScaler(), numeric_features)]

preprocessor = ColumnTransformer(transformers=t)

pipe_lasso = Pipeline([('preprocessor', preprocessor),
                       ('clf', Lasso(random_state=seed_value))])

grid_params_lasso = [{'clf__max_iter': [10000, 50000, 100000], 
                      'clf__alpha': [0.000001, 0.00001, 0.0001, 0.001, 0.01,
                                     0.1, 1, 10, 100]}] 

LR = GridSearchCV(estimator=pipe_lasso, param_grid=grid_params_lasso,
            scoring='neg_mean_absolute_error', cv=3, n_jobs=-1) 

pipe_elnet = Pipeline([('preprocessor', preprocessor),
                       ('clf1', ElasticNet(random_state=seed_value))])

grid_params_elnet = [{'clf1__max_iter': [100000, 500000, 10000000], 
                      'clf1__alpha': [0.000001, 0.00001, 0.0001, 0.001, 0.01,
                                      0.1, 1, 10, 100]}] 

ELNETR = GridSearchCV(estimator=pipe_elnet, param_grid=grid_params_elnet,
            scoring='neg_mean_absolute_error', cv=3, n_jobs=-1) 

grids = [LR, ELNETR]

grid_dict = {0: 'Lasso', 1: 'Elastic Net'}

# Fit the parameters for the grid search 
print('Performing Linear HPO for 2020...')
search_time_start = time.time()
best_mae = 1000000
best_reg = 0
best_gs = ''
for idx, gs in enumerate(grids):
    print('\nEstimator: %s' % grid_dict[idx])
    with parallel_backend('threading', n_jobs=-1):
        gs.fit(X_train, y_train)
        print('Best params are : %s' % gs.best_params_)
    # Best training data mae
        print('Best training mae: %.3f' % gs.best_score_)
    # Predict on test data with best params
        y_pred = gs.predict(X_test)
    # Test data mae of model with best params
        print('Test set mae for best params: %.3f ' % mean_absolute_error(y_test,
                                                                          y_pred))
    # Track best (lowest test mae) model
        if mean_absolute_error(y_test, y_pred) < best_mae:
            best_mae = mean_absolute_error(y_test, y_pred)
            best_gs = gs
            best_reg = idx
print('\nRegressor with best test set mae: %s' % grid_dict[best_reg])
print('Finished fit the best hyperparameters from Lasso/Elastic Net grid search to 2020 data:',
      time.time() - search_time_start)
print('======================================================================')

# Fit model with best MAE
Trade_Linear_HPO = Lasso(alpha=0.001, max_iter=10000, random_state=seed_value)

preprocessor = ColumnTransformer(transformers=t)

Trade_Linear_HPO = Pipeline([('preprocessor', preprocessor),
                       ('clf', Lasso(alpha=0.001, max_iter=10000, 
                                     random_state=seed_value))])

grid_params_lasso = [{'clf__max_iter': [10000], 
                      'clf__alpha': [0.000001]}] 

LR = GridSearchCV(estimator=pipe_lasso,
            param_grid=grid_params_lasso,
            scoring='neg_mean_absolute_error',
            cv=3, n_jobs=-1) 

print('Start fit the best hyperparameters from Lasso/Elastic Net grid search to 2020 data..')
search_time_start = time.time()
with parallel_backend('threading', n_jobs=-1):
    Trade_Linear_HPO.fit(X_train, y_train)
print('Finished fit the best hyperparameters from Lasso/Elastic Net grid search to 2020 data:',
      time.time() - search_time_start)
print('======================================================================')

# Save model
Pkl_Filename = 'MaritimeTrade_Linear_2020gridSearch.pkl'  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(Trade_Linear_HPO, file)

# =============================================================================
# # To load saved model
# Trade_Linear_HPO = joblib.load('MaritimeTrade_Linear_2020gridSearch.pkl')
# print(Trade_Linear_HPO)
# =============================================================================

###############################################################################
# Fit best parameters on model
# Create dummy variables for categorical variables
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# MinMax Scaling
X_train = pd.DataFrame(mn.fit_transform(X_train))
X_test = pd.DataFrame(mn.fit_transform(X_test))

Trade_Linear_HPO = Lasso(alpha=0.001, max_iter=10000, random_state=seed_value)

print('Start fit the best hyperparameters from Lasso/Elastic Net grid search to 2020 data..')
search_time_start = time.time()
with parallel_backend('threading', n_jobs=-1):
    Trade_Linear_HPO.fit(X_train, y_train)
print('Finished fit the best hyperparameters from Lasso/Elastic Net grid search to 2020 data:',
      time.time() - search_time_start)
print('======================================================================')

# Save model
Pkl_Filename = 'MaritimeTrade_Linear_HPO_2020.pkl'  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(Trade_Linear_HPO, file)

# =============================================================================
# # To load saved model
# Trade_Linear_HPO = joblib.load('MaritimeTrade_Linear_HPO_2020.pkl')
# print(Trade_Linear_HPO)
# =============================================================================

###############################################################################
# Predict based on training 
y_pred_Trade_Linear_HPO = Trade_Linear_HPO.predict(X_test)

print('Results from LASSO using 2020 HPO:')
print('\n')

# Calculate MAE/MSE
print('Mean Absolute Error (MAE): %3f' % sklearn.metrics.mean_absolute_error(y_pred_Trade_Linear_HPO,
                                                                             y_test))
print('Mean Square Error (MSE): %3f' % sklearn.metrics.mean_squared_error(y_pred_Trade_Linear_HPO,
                                                                          y_test))

# Plot actual vs predicted metric tonnage
pyplot.plot(y_test, label = '2020')
pyplot.plot(y_pred_Trade_Linear_HPO, label = '2020 - predicted')
pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
pyplot.savefig('Train2020_HPO_Test2020_Predict2020.png', dpi=my_dpi, 
               bbox_inches='tight')
pyplot.show()

# Model metrics with Eli5
# Compute permutation feature importance
perm_importance = PermutationImportance(Trade_Linear_HPO,
                                        random_state=seed_value).fit(X_test,
                                                                     y_test)

# Create dummy variables for feature names 
X = pd.get_dummies(X, drop_first=True)

# Store feature weights in an object
html_obj = eli.show_weights(perm_importance,
                            feature_names = X.columns.tolist())

# Write feature weights html object to a file 
with open('D:\MaritimeTrade\Results\Linear\Trade_Linear_HPO_2020_WeightsFeatures.htm',
          'wb') as f:
    f.write(html_obj.data.encode("UTF-8"))

# Open the stored feature weights HTML file
url = r'D:\MaritimeTrade\Results\Linear\Trade_Linear_HPO_2020_WeightsFeatures.htm'
webbrowser.open(url, new=2)

# Explain weights
explanation = eli.explain_weights_sklearn(perm_importance,
                            feature_names = X.columns.tolist())
exp = format_as_dataframe(explanation)

# Write processed data to csv
exp.to_csv('MaritimeTrade_Linear_HPO_2020_WeightsExplain.csv',
           index=False, encoding='utf-8-sig')

# Show prediction
html_obj2 = show_prediction(Trade_Linear_HPO, X.iloc[1],
                            show_feature_values=True)

# Write show prediction html object to a file 
with open('D:\MaritimeTrade\Results\Linear\Trade_Linear_HPO_2020_Prediction.htm',
          'wb') as f:
    f.write(html_obj2.data.encode("UTF-8"))

# Open the show prediction stored HTML file
url2 = r'D:\MaritimeTrade\Results\Linear\Trade_Linear_HPO_2020_Prediction.htm'
webbrowser.open(url2, new=2)

# Explain prediction
#explanation_pred = eli.explain_prediction(Trade_Linear_HPO, np.array(X_test)[1])
#explanation_pred

###############################################################################
# LIME for model explanation
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X.columns,
    class_names=['Metric_Tons'],
    mode='regression')

exp = explainer.explain_instance(
    data_row=X.iloc[1], 
    predict_fn=Trade_Linear_HPO.predict)

exp.save_to_file('Trade_Linear_HPO_2020_LIME.html')

###############################################################################
# Test 2020 HPO model on 2019 data
# Set up for partitioning data
X1 = df.iloc[:,:-1]
y1 = df.iloc[:,:1]

# Encode variables using ranking - ordinal 
ce_ord = ce.OrdinalEncoder(cols = ['foreign_company_size', 'US_company_size'])
X1 = ce_ord.fit_transform(X1, y1['Metric_Tons'])

# Create dummy variables for categorical variables
X1 = pd.get_dummies(X1, drop_first=True)

# MinMax Scaling
X1 = pd.DataFrame(mn.fit_transform(X1))

print('Start fit the best hyperparameters from 2020 Lasso/Elastic Net grid search to 2019 data..')
search_time_start = time.time()
with parallel_backend('threading', n_jobs=-1):
    Trade_Linear_HPO.fit(X1, y1)
print('Finished fit the best hyperparameters from 2020 Lasso/Elastic Net grid search to 2019 data:',
      time.time() - search_time_start)
print('======================================================================')

# Save model
Pkl_Filename = 'MaritimeTrade_Linear_HPO_2020_2019data.pkl'  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(Trade_Linear_HPO, file)

# =============================================================================
# # To load saved model
# Trade_Linear_HPO = joblib.load('MaritimeTrade_Linear_HPO_2020_2019data.pkl')
# print(Trade_Linear_HPO)
# =============================================================================

###############################################################################
# Predict based on training 
y_pred_Trade_Linear_HPO1 = Trade_Linear_HPO.predict(X1)

print('Results from LASSO using 2020 HPO on 2019 data:')
print('\n')

# Calculate MAE/MSE
print('Mean Absolute Error (MAE): %3f' % sklearn.metrics.mean_absolute_error(y_pred_Trade_Linear_HPO1, y1))
print('Mean Square Error (MSE): %3f' % sklearn.metrics.mean_squared_error(y_pred_Trade_Linear_HPO1, y1))

# Plot actual vs predicted metric tonnage
pyplot.plot(y, label = '2020')
pyplot.plot(y_pred_Trade_Linear_HPO1, label = '2019 - predicted')
pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
pyplot.savefig('Train20_HPO_Predict2019.png', dpi=my_dpi, bbox_inches='tight')
pyplot.show()

# Plot actual vs predicted metric tonnage
pyplot.plot(y1, label = '2019')
pyplot.plot(y_pred_Trade_Linear_HPO1, label = '2019 - predicted')
pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
pyplot.savefig('Train20_HPO_Test2019_Predict2019.png', dpi=my_dpi, 
               bbox_inches='tight')
pyplot.show()

# Model metrics with Eli5
# Compute permutation feature importance
perm_importance = PermutationImportance(Trade_Linear_HPO,
                                        random_state=seed_value).fit(X1, y1)

# Store feature weights in an object
html_obj = eli.show_weights(perm_importance,
                            feature_names = X.columns.tolist())

# Write feature weights html object to a file 
with open('D:\MaritimeTrade\Results\Linear\Trade_Linear_HPO_2020_2019data_WeightsFeatures.htm',
          'wb') as f:
    f.write(html_obj.data.encode("UTF-8"))

# Open the stored feature weights HTML file
url = r'D:\MaritimeTrade\Results\Linear\Trade_Linear_HPO_2020_2019data_WeightsFeatures.htm'
webbrowser.open(url, new=2)

# Explain weights
explanation = eli.explain_weights_sklearn(perm_importance,
                            feature_names = X.columns.tolist())
exp = format_as_dataframe(explanation)

# Write processed data to csv
exp.to_csv('MaritimeTrade_Linear_HPO_2020_2019data_WeightsExplain.csv',
           index=False, encoding='utf-8-sig')

# Show prediction
html_obj2 = show_prediction(Trade_Linear_HPO, X.iloc[1],
                            show_feature_values=True)

# Write show prediction html object to a file 
with open('D:\MaritimeTrade\Results\Linear\Trade_Linear_HPO_2020_2019data_Prediction.htm',
          'wb') as f:
    f.write(html_obj2.data.encode("UTF-8"))

# Open the show prediction stored HTML file
url2 = r'D:\MaritimeTrade\Results\Linear\Trade_Linear_HPO_2020_2019data_Prediction.htm'
webbrowser.open(url2, new=2)

# Explain prediction
#explanation_pred = eli.explain_prediction(Trade_Linear_HPO, np.array(X1)[1])
#explanation_pred

###############################################################################
# LIME for model explanation
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X1),
    feature_names=X.columns,
    class_names=['Metric_Tons'],
    mode='regression')

exp = explainer.explain_instance(
    data_row=X1.iloc[1], 
    predict_fn=Trade_Linear_HPO.predict)

exp.save_to_file('Trade_Linear_HPO_2020_2019data_LIME.html')

###############################################################################