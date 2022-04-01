# -*- coding: utf-8 -*-
"""
@author: aschu
"""
###############################################################################
######################## light GBM HPO for 2019 - 2020 ########################
################ Train 2019 Test 2020 & Train 2020 Test 2019 ##################
###############################################################################
import os
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler
import joblib
from hyperopt import STATUS_OK
import lightgbm as lgb
from hyperopt import hp
from hyperopt import tpe
from hyperopt import fmin, tpe, Trials
import csv
from timeit import default_timer as timer
import ast
import pickle
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from eli5.sklearn import PermutationImportance 
from eli5 import show_weights
import webbrowser
import eli5 as eli
from eli5.sklearn import explain_weights_sklearn
from eli5.formatters import format_as_dataframe, format_as_dataframes
from eli5 import show_prediction
import lime
from lime import lime_tabular

# Set seed
seed_value = 42
os.environ['lightGBM_train19test20'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

# Input Path
path = r'D:\MaritimeTrade\Data'
os.chdir(path)    

# Read in the df
df = pd.read_csv('combined_trade_final_LSTM.csv', low_memory = False)
df = df.drop_duplicates()

# Drop time and COVID-19 vars
df = df.drop(['State', 'DateTime', 'Date_Weekly_COVID'], axis=1)

# Convert dtypes
df = df.copy()
df['State_Closure_EA_Diff'] = df['State_Closure_EA_Diff'].astype('float64')
df['Container_Type_Dry'] = df['Container_Type_Dry'].astype('object')

# Filter missing data
df = df[df.Foreign_Country_Region.notna() & df.Average_Tariff.notna()]

# Filter df to 2019
df1 = df[df['Year'] == 2019]

# Filter df to 2020
df2 = df[df['Year'] == 2020]

# Drop year variable
df1 = df1.drop(['Year'], axis=1)
df2 = df2.drop(['Year'], axis=1)

# Prepare 2019 for partitioning data
X = df1.drop(['Metric_Tons'],axis=1)
y = df1['Metric_Tons']

# Set up train/test split with stratified by 'DateTime_YearWeek'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                    stratify= X.DateTime_YearWeek,
                                                    random_state = seed_value)

# Drop time variable
X_train = X_train.drop(['DateTime_YearWeek'], axis=1)
X_test = X_test.drop(['DateTime_YearWeek'], axis=1)

# Training set: Encode variables using ranking - ordinal               
ce_ord = ce.OrdinalEncoder(cols = ['foreign_company_size', 'US_company_size'])
X_train = ce_ord.fit_transform(X_train)

# Testing set: Encode variables using ranking - ordinal               
ce_ord = ce.OrdinalEncoder(cols = ['foreign_company_size', 'US_company_size'])
X_test = ce_ord.fit_transform(X_test)

# Train: Create dummy variables for categorical variables
X_train = pd.get_dummies(X_train, drop_first=True)

# Test: Create dummy variables for categorical variables
X_test = pd.get_dummies(X_test, drop_first=True)

# MinMax Scaling
mn = MinMaxScaler()
X_train = pd.DataFrame(mn.fit_transform(X_train), columns = X_train.columns)
X_test = pd.DataFrame(mn.fit_transform(X_test), columns = X_test.columns)

# Set path for ML results
path = r'D:\MaritimeImportsExports\ML\lightGBM\trialOptions'
os.chdir(path)

###############################################################################
############################# light GBM HPO  ##################################
############################# GBDT, DART, GOSS ################################
############################### 300 trials ####################################
########################## Train 2019 Test 2020 ###############################
###############################################################################
# Create a lgb dataset
params = {'verbose': -1}
train_set = lgb.Dataset(X_train, label = y_train, params=params)

# Define an objective function
NUM_EVAL = 300
N_FOLDS = 3

def objective(params, n_folds = N_FOLDS):
    """lightGBM HPO"""
    
    # Keep track of evals
    global ITERATION
    
    ITERATION += 1
    
    # Retrieve the subsample if present otherwise set to 1.0
    subsample = params['boosting_type'].get('subsample', 1.0)
    
    # Extract the boosting type
    params['boosting_type'] = params['boosting_type']['boosting_type']
    params['subsample'] = subsample
    
    # Make sure parameters that need to be integers are integers
    for param_name in ['max_depth', 'num_leaves']:
        params[param_name] = int(params[param_name])
        
    start = timer()
    
    # Perform n_folds cross validation
    cv_results = lgb.cv(params, train_set, num_boost_round = 1000, nfold = N_FOLDS, 
                        stratified=False, early_stopping_rounds = 10, 
                        metrics = 'rmse', seed = seed_value)
    
    run_time = timer() - start
    
    loss = cv_results['rmse-mean'][-1]
        
    # Boosting rounds that returned the lowest cv score
    n_estimators = int(np.argmin(cv_results['rmse-mean']) + 1)
    
    # Write to the csv file ('a' means append)
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, params, ITERATION, n_estimators, run_time])

    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'iteration': ITERATION,
            'estimators': n_estimators, 'train_time': run_time,
            'status': STATUS_OK}

# Optimization algorithm
tpe_algorithm = tpe.suggest

# Define the parameter grid
param_grid = {
    'force_col_wise': hp.choice('force_col_wise', "+"),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
    'max_depth': hp.choice('max_depth', np.arange(5, 6, dtype=int)),
    'num_leaves': hp.choice('num_leaves', np.arange(30, 100, dtype=int)),
    'boosting_type': hp.choice('boosting_type', [{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)}, 
                                                 {'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1)},
                                                 {'boosting_type': 'goss', 'subsample': 1.0}]),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
}

# File to save results
out_file = 'lightGBM_HPO_train19test19_trials.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration', 'estimators', 'train_time'])
of_connection.close()

# Set global variable and HPO is run with fmin
global ITERATION
ITERATION = 0
bayesOpt_trials = Trials()

best_param = fmin(objective, param_grid, algo=tpe.suggest,
                  max_evals=NUM_EVAL, trials=bayesOpt_trials,
                  rstate=np.random.RandomState(42))

# Sort the trials with lowest loss (highest AUC) first
bayesOpt_trials_results = sorted(bayesOpt_trials.results, key = lambda x: x['loss'])
print('Top two trials with the lowest loss (lowest MAE)')
print(bayesOpt_trials_results[:2])

# Access results
results = pd.read_csv('lightGBM_HPO_train19test19_trials.csv')

# Sort with best scores on top and reset index for slicing
results.sort_values('loss', ascending = True, inplace = True)
results.reset_index(inplace = True, drop = True)

# Convert from a string to a dictionary for later use
ast.literal_eval(results.loc[0, 'params'])

# Evaluate Best Results
# Extract the ideal number of estimators and hyperparameters
best_bayes_estimators = int(results.loc[0, 'estimators'])
best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()

# Set path for ML results
path = r'D:\MaritimeImportsExports\ML\lightGBM\Model_PKL'
os.chdir(path)

# Re-create the best model and train on the training data
best_bayes_model = lgb.LGBMRegressor(n_estimators=best_bayes_estimators,
                                      n_jobs = -1, objective = 'regression',
                                      random_state = seed_value, **best_bayes_params)

# Fit the model
best_bayes_model.fit(X_train, y_train)

# Save model
Pkl_Filename = 'lightGBM_HPO_train19test19.pkl'  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(best_bayes_model, file)

# =============================================================================
# # To load saved model
# model = joblib.load('lightGBM_HPO_train19test19.pkl')
# print(model)
# =============================================================================
    
# Model Metrics
y_train_pred = best_bayes_model.predict(X_train)
y_test_pred = best_bayes_model.predict(X_test)

print('\nModel Metrics for lightGBM HPO Train 2019 Test 2019')
print('MAE train: %.3f, test: %.3f' % (
        mean_absolute_error(y_train, y_train_pred),
        mean_absolute_error(y_test, y_test_pred)))
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('RMSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred, squared=False),
        mean_squared_error(y_test, y_test_pred, squared=False)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

# Evaluate on the testing data 
print('The best model from Bayes optimization scores {:.5f} MSE on the test set.'.format(mean_squared_error(y_test, y_test_pred)))
print('This was achieved after {} search iterations'.format(results.loc[0, 'iteration']))

# Create a new dataframe for storing parameters
bayes_params = pd.DataFrame(columns = list(ast.literal_eval(results.loc[0, 'params']).keys()),
                            index = list(range(len(results))))

# Convert data types for graphing
bayes_params['colsample_bytree'] = bayes_params['colsample_bytree'].astype('float64')
bayes_params['learning_rate'] = bayes_params['learning_rate'].astype('float64')
bayes_params['num_leaves'] = bayes_params['num_leaves'].astype('float64')
bayes_params['reg_alpha'] = bayes_params['reg_alpha'].astype('float64')
bayes_params['reg_lambda'] = bayes_params['reg_lambda'].astype('float64')
bayes_params['subsample'] = bayes_params['subsample'].astype('float64')

# Add the results with each parameter a different column
for i, params in enumerate(results['params']):
    bayes_params.loc[i, :] = list(ast.literal_eval(params).values())
    
bayes_params['loss'] = results['loss']
bayes_params['iteration'] = results['iteration']

# Set path for ML results
path = r'D:\MaritimeImportsExports\ML\lightGBM\bayesParams'
os.chdir(path)

# Save dataframes of parameters
bayes_params.to_csv('bayes_params_lightGBM_HPO_train19test19_300.csv',
                    index = False)

# Visualize results from different boosting methods
bayes_params['boosting_type'].value_counts().plot.bar(figsize = (10, 5),
                                                      color = 'green',
                                                      title = 'Bayes Optimization Boosting Type')

print('Bayes Optimization 300 trials boosting type percentages')
100 * bayes_params['boosting_type'].value_counts() / len(bayes_params)

# Density plots of the learning rate distributions 
plt.figure(figsize = (20, 8))
plt.rcParams['font.size'] = 18
sns.kdeplot(bayes_params['learning_rate'], label = 'Bayes Optimization', linewidth = 2)
plt.legend(loc = 1)
plt.xlabel('Learning Rate'); plt.ylabel('Density'); plt.title('Learning Rate Distribution');
plt.show()

# Create plots of Hyperparameters that are numeric
for i, hpo in enumerate(bayes_params.columns):
    if hpo not in ['boosting_type', 'iteration', 'subsample', 'force_col_wise',
                     'max_depth']:
        plt.figure(figsize = (14, 6))
        # Plot the random search distribution and the bayes search distribution
        if hpo != 'loss':
            sns.kdeplot(bayes_params[hpo], label = 'Bayes Optimization')
            plt.legend(loc = 0)
            plt.title('{} Distribution'.format(hpo))
            plt.xlabel('{}'.format(hpo)); plt.ylabel('Density')
            plt.tight_layout()
            plt.show()

# Map boosting type to integer (essentially label encoding)
bayes_params['boosting_int'] = bayes_params['boosting_type'].replace({'goss': 0, 'dart': 1, 'gbdt': 2})

# Plot the boosting type over the search
plt.plot(bayes_params['iteration'], bayes_params['boosting_int'], 'ro')
plt.yticks([0, 1, 2], ['goss', 'dart', 'gbdt']);
plt.xlabel('Iteration'); plt.title('Boosting Type over trials')
plt.show()

# Plot quantitative hyperparameters
fig, axs = plt.subplots(1, 4, figsize = (20, 5))
i = 0
for i, hpo in enumerate(['colsample_bytree', 'learning_rate', 'num_leaves']):
    
        # Scatterplot
        sns.regplot('iteration', hpo, data = bayes_params, ax = axs[i])
        axs[i].set(xlabel = 'Iteration', ylabel = '{}'.format(hpo), title = '{} over Trials'.format(hpo))
plt.tight_layout()
plt.show()

# Scatterplot of regularization hyperparameters
fig, axs = plt.subplots(1, 2, figsize = (14, 6))
i = 0
for i, hpo in enumerate(['reg_alpha', 'reg_lambda']):
        sns.regplot('iteration', hpo, data = bayes_params, ax = axs[i])
        axs[i].set(xlabel = 'Iteration', ylabel = '{}'.format(hpo), title = '{} over Trials'.format(hpo))
plt.tight_layout()
plt.show()

###############################################################################
# Set path for ML results
path = r'D:\MaritimeImportsExports\ML\lightGBM\Model_Explanations'
os.chdir(path)

# Model metrics with Eli5
# Compute permutation feature importance
perm_importance = PermutationImportance(best_bayes_model,
                                        random_state=seed_value).fit(X_test,
                                                                     y_test)
                                                                     
# Store feature weights in an object
html_obj = eli.show_weights(perm_importance,
                            feature_names = X_train.columns.tolist())

# Write feature weights html object to a file 
with open(r'D:\MaritimeTrade\ML\lightGBM\Model_Explanations\best_bayes_train19test19_HPO_300_WeightsFeatures.htm',
          'wb') as f:
    f.write(html_obj.data.encode("UTF-8"))

# Open the stored feature weights HTML file
url = r'D:\MaritimeTrade\ML\lightGBM\Model_Explanations\best_bayes_train19test19_HPO_300_WeightsFeatures.htm'
webbrowser.open(url, new=2)

# Show prediction
html_obj2 = show_prediction(best_bayes_model, X_train.iloc[1],
                            show_feature_values=True)

# Write show prediction html object to a file 
with open(r'D:\MaritimeTrade\ML\lightGBM\Model_Explanations\best_bayes_train19test19_HPO_300_Prediction.htm',
          'wb') as f:
    f.write(html_obj2.data.encode("UTF-8"))

# Open the show prediction stored HTML file
url2 = r'D:\MaritimeTrade\ML\lightGBM\Model_Explanations\best_bayes_train19test19_HPO_300_Prediction.htm'
webbrowser.open(url2, new=2)

# Explain prediction
#explanation_pred = eli.explain_prediction(best_bayes_model, np.array(X_test)[1])
#explanation_pred

###############################################################################
# Set path for ML results
path = r'D:\MaritimeImportsExports\ML\lightGBM\bestBayes_WeightsExplain'
os.chdir(path)

# Explain weights
explanation = eli.explain_weights_sklearn(perm_importance,
                            feature_names = X_train.columns.tolist())
exp = format_as_dataframe(explanation)

# Write processed data to csv
exp.to_csv('best_bayes_train19test19_HPO_300_WeightsExplain.csv',
           index=False, encoding='utf-8-sig')

###############################################################################
##################### Test trained model on 2019 on 2020 ######################
###############################################################################
# Prepare 2020 to fit model train on 2019
X_test1 = df2.drop(['Metric_Tons'],axis=1)
y_test1 = df2['Metric_Tons']

# Drop year variable
X_test1 = X_test1.drop(['DateTime_YearWeek'], axis=1)

# Testing set: Encode variables using ranking - ordinal               
ce_ord = ce.OrdinalEncoder(cols = ['foreign_company_size', 'US_company_size'])
X_test1 = ce_ord.fit_transform(X_test1)

# Test: Create dummy variables for categorical variables
X_test1 = pd.get_dummies(X_test1, drop_first=True)

# MinMax Scaling
X_test1 = pd.DataFrame(mn.fit_transform(X_test1), columns = X_test1.columns)

# Model Metrics
y_test_pred = best_bayes_model.predict(X_test1)

print('\nModel Metrics for lightGBM HPO Train 2019 Test 2020')
print('MAE train: %.3f, test: %.3f' % (
        mean_absolute_error(y_train, y_train_pred),
        mean_absolute_error(y_test1, y_test_pred)))
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test1, y_test_pred)))
print('RMSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred, squared=False),
        mean_squared_error(y_test1, y_test_pred, squared=False)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test1, y_test_pred)))

###############################################################################
# Set path for ML results
path = r'D:\MaritimeImportsExports\ML\lightGBM\Model_Explanations'
os.chdir(path)

# Model metrics with Eli5
# Compute permutation feature importance
perm_importance = PermutationImportance(best_bayes_model,
                                        random_state=seed_value).fit(X_test1,
                                                                     y_test1)
                                                                     
# Store feature weights in an object
html_obj = eli.show_weights(perm_importance,
                            feature_names = X_train.columns.tolist())

# Write feature weights html object to a file 
with open(r'D:\MaritimeTrade\ML\lightGBM\Model_Explanations\best_bayes_train19test20_HPO_300_WeightsFeatures.htm',
          'wb') as f:
    f.write(html_obj.data.encode("UTF-8"))

# Open the stored feature weights HTML file
url = r'D:\MaritimeTrade\ML\lightGBM\Model_Explanations\best_bayes_train19test20_HPO_300_WeightsFeatures.htm'
webbrowser.open(url, new=2)

# Show prediction
html_obj2 = show_prediction(best_bayes_model, X_train.iloc[1],
                            show_feature_values=True)

# Write show prediction html object to a file 
with open(r'D:\MaritimeTrade\ML\lightGBM\Model_Explanations\best_bayes_train19test20_HPO_300_Prediction.htm',
          'wb') as f:
    f.write(html_obj2.data.encode("UTF-8"))

# Open the show prediction stored HTML file
url2 = r'D:\MaritimeTrade\ML\lightGBM\Model_Explanations\best_bayes_train19test20_HPO_300_Prediction.htm'
webbrowser.open(url2, new=2)

# Explain prediction
#explanation_pred = eli.explain_prediction(best_bayes_model, np.array(X_test)[1])
#explanation_pred

###############################################################################
# Set path for ML results
path = r'D:\MaritimeImportsExports\ML\lightGBM\bestBayes_WeightsExplain'
os.chdir(path)

# Explain weights
explanation = eli.explain_weights_sklearn(perm_importance,
                            feature_names = X_train.columns.tolist())
exp = format_as_dataframe(explanation)

# Write processed data to csv
exp.to_csv('best_bayes_train19test20_HPO_300_WeightsExplain.csv',
           index=False, encoding='utf-8-sig')

del X, y, X_train, X_test, y_train, y_test, X_test1, y_test1, best_bayes_model
del y_train_pred, y_test_pred

###############################################################################
############################# light GBM HPO  ##################################
########################## Train 2020 Test 2019 ###############################
###############################################################################
# Set seed
seed_value = 42
os.environ['lightGBM_train20test19'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

# Prepare 2020 for partitioning data
X = df2.drop(['Metric_Tons'],axis=1)
y = df2['Metric_Tons']

# Set up train/test split with stratified by 'DateTime_YearWeek'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                    stratify= X.DateTime_YearWeek,
                                                    random_state = seed_value)

# Drop DateTime_YearWeek variable
X_train = X_train.drop(['DateTime_YearWeek'], axis=1)
X_test = X_test.drop(['DateTime_YearWeek'], axis=1)

# Training set: Encode variables using ranking - ordinal               
ce_ord = ce.OrdinalEncoder(cols = ['foreign_company_size', 'US_company_size'])
X_train = ce_ord.fit_transform(X_train)

# Testing set: Encode variables using ranking - ordinal               
ce_ord = ce.OrdinalEncoder(cols = ['foreign_company_size', 'US_company_size'])
X_test = ce_ord.fit_transform(X_test)

# Train: Create dummy variables for categorical variables
X_train = pd.get_dummies(X_train, drop_first=True)

# Test: Create dummy variables for categorical variables
X_test = pd.get_dummies(X_test, drop_first=True)

# MinMax Scaling
mn = MinMaxScaler()
X_train = pd.DataFrame(mn.fit_transform(X_train), columns = X_train.columns)
X_test = pd.DataFrame(mn.fit_transform(X_test), columns = X_test.columns)

###############################################################################
# Set path for ML results
path = r'D:\MaritimeImportsExports\ML\lightGBM\trialOptions'
os.chdir(path)

# Create a lgb dataset
params = {'verbose': -1}
train_set = lgb.Dataset(X_train, label = y_train, params=params)

# File to save results
out_file = 'lightGBM_HPO_train20test20_trials.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration', 'estimators', 'train_time'])
of_connection.close()

# Set global variable and HPO is run with fmin
global ITERATION
ITERATION = 0
bayesOpt_trials = Trials()

best_param = fmin(objective, param_grid, algo=tpe.suggest,
                  max_evals=NUM_EVAL, trials=bayesOpt_trials,
                  rstate=np.random.RandomState(42))

# Sort the trials with lowest loss (highest AUC) first
bayesOpt_trials_results = sorted(bayesOpt_trials.results, key = lambda x: x['loss'])
print('Top two trials with the lowest loss (lowest MAE)')
print(bayesOpt_trials_results[:2])

# Access results
results = pd.read_csv('lightGBM_HPO_train20test20_trials.csv')

# Sort with best scores on top and reset index for slicing
results.sort_values('loss', ascending = True, inplace = True)
results.reset_index(inplace = True, drop = True)

# Convert from a string to a dictionary for later use
ast.literal_eval(results.loc[0, 'params'])

# Evaluate Best Results
# Extract the ideal number of estimators and hyperparameters
best_bayes_estimators = int(results.loc[0, 'estimators'])
best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()

# Set path for ML results
path = r'D:\MaritimeImportsExports\ML\lightGBM\Model_PKL'
os.chdir(path)

# Re-create the best model and train on the training data
best_bayes_model = lgb.LGBMRegressor(n_estimators=best_bayes_estimators,
                                      n_jobs = -1, objective = 'regression',
                                      random_state = seed_value, **best_bayes_params)

# Fit the model
best_bayes_model.fit(X_train, y_train)

# Save model
Pkl_Filename = 'lightGBM_HPO_train20test20.pkl'  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(best_bayes_model, file)

# Model Metrics
y_train_pred = best_bayes_model.predict(X_train)
y_test_pred = best_bayes_model.predict(X_test)

print('\nModel Metrics for lightGBM HPO Train 2020 Test 2020')
print('MAE train: %.3f, test: %.3f' % (
        mean_absolute_error(y_train, y_train_pred),
        mean_absolute_error(y_test, y_test_pred)))
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('RMSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred, squared=False),
        mean_squared_error(y_test, y_test_pred, squared=False)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

# Evaluate on the testing data 
print('The best model from Bayes optimization scores {:.5f} MSE on the test set.'.format(mean_squared_error(y_test, y_test_pred)))
print('This was achieved after {} search iterations'.format(results.loc[0, 'iteration']))

# Create a new dataframe for storing parameters
bayes_params = pd.DataFrame(columns = list(ast.literal_eval(results.loc[0, 'params']).keys()),
                            index = list(range(len(results))))

# Convert data types for graphing
bayes_params['colsample_bytree'] = bayes_params['colsample_bytree'].astype('float64')
bayes_params['learning_rate'] = bayes_params['learning_rate'].astype('float64')
bayes_params['num_leaves'] = bayes_params['num_leaves'].astype('float64')
bayes_params['reg_alpha'] = bayes_params['reg_alpha'].astype('float64')
bayes_params['reg_lambda'] = bayes_params['reg_lambda'].astype('float64')
bayes_params['subsample'] = bayes_params['subsample'].astype('float64')

# Add the results with each parameter a different column
for i, params in enumerate(results['params']):
    bayes_params.loc[i, :] = list(ast.literal_eval(params).values())
    
bayes_params['loss'] = results['loss']
bayes_params['iteration'] = results['iteration']

# Set path for ML results
path = r'D:\MaritimeImportsExports\ML\lightGBM\bayesParams'
os.chdir(path)

# Save dataframes of parameters
bayes_params.to_csv('bayes_params_lightGBM_HPO_train20test20_300.csv',
                    index = False)

# Visualize results from different boosting methods
bayes_params['boosting_type'].value_counts().plot.bar(figsize = (10, 5),
                                                      color = 'green',
                                                      title = 'Bayes Optimization Boosting Type')

print('Bayes Optimization 300 trials boosting type percentages')
100 * bayes_params['boosting_type'].value_counts() / len(bayes_params)

# Density plots of the learning rate distributions 
plt.figure(figsize = (20, 8))
plt.rcParams['font.size'] = 18
sns.kdeplot(bayes_params['learning_rate'], label = 'Bayes Optimization', linewidth = 2)
plt.legend(loc = 1)
plt.xlabel('Learning Rate'); plt.ylabel('Density'); plt.title('Learning Rate Distribution');
plt.show()

# Create plots of Hyperparameters that are numeric
for i, hpo in enumerate(bayes_params.columns):
    if hpo not in ['boosting_type', 'iteration', 'subsample', 'force_col_wise',
                     'max_depth']:
        plt.figure(figsize = (14, 6))
        # Plot the random search distribution and the bayes search distribution
        if hpo != 'loss':
            sns.kdeplot(bayes_params[hpo], label = 'Bayes Optimization')
            plt.legend(loc = 0)
            plt.title('{} Distribution'.format(hpo))
            plt.xlabel('{}'.format(hpo)); plt.ylabel('Density')
            plt.tight_layout()
            plt.show()

# Map boosting type to integer (essentially label encoding)
bayes_params['boosting_int'] = bayes_params['boosting_type'].replace({'goss': 0, 'dart': 1, 'gbdt': 2})

# Plot the boosting type over the search
plt.plot(bayes_params['iteration'], bayes_params['boosting_int'], 'ro')
plt.yticks([0, 1, 2], ['goss', 'dart', 'gbdt']);
plt.xlabel('Iteration'); plt.title('Boosting Type over trials')
plt.show()

# Plot quantitative hyperparameters
fig, axs = plt.subplots(1, 4, figsize = (20, 5))
i = 0
for i, hpo in enumerate(['colsample_bytree', 'learning_rate', 'num_leaves']):
    
        # Scatterplot
        sns.regplot('iteration', hpo, data = bayes_params, ax = axs[i])
        axs[i].set(xlabel = 'Iteration', ylabel = '{}'.format(hpo), title = '{} over Trials'.format(hpo))
plt.tight_layout()
plt.show()

# Scatterplot of regularization hyperparameters
fig, axs = plt.subplots(1, 2, figsize = (14, 6))
i = 0
for i, hpo in enumerate(['reg_alpha', 'reg_lambda']):
        sns.regplot('iteration', hpo, data = bayes_params, ax = axs[i])
        axs[i].set(xlabel = 'Iteration', ylabel = '{}'.format(hpo), title = '{} over Trials'.format(hpo))
plt.tight_layout()
plt.show()

###############################################################################
# Set path for ML results
path = r'D:\MaritimeImportsExports\ML\lightGBM\Model_Explanations'
os.chdir(path)

# Model metrics with Eli5
# Compute permutation feature importance
perm_importance = PermutationImportance(best_bayes_model,
                                        random_state=seed_value).fit(X_test,
                                                                     y_test)
                                                                     
# Store feature weights in an object
html_obj = eli.show_weights(perm_importance,
                            feature_names = X_train.columns.tolist())

# Write feature weights html object to a file 
with open(r'D:\MaritimeTrade\ML\lightGBM\Model_Explanations\best_bayes_train20test20_HPO_300_WeightsFeatures.htm',
          'wb') as f:
    f.write(html_obj.data.encode("UTF-8"))

# Open the stored feature weights HTML file
url = r'D:\MaritimeTrade\ML\lightGBM\Model_Explanations\best_bayes_train20test20_HPO_300_WeightsFeatures.htm'
webbrowser.open(url, new=2)

# Show prediction
html_obj2 = show_prediction(best_bayes_model, X_train.iloc[1],
                            show_feature_values=True)

# Write show prediction html object to a file 
with open(r'D:\MaritimeTrade\ML\lightGBM\Model_Explanations\best_bayes_train20test20_HPO_300_Prediction.htm',
          'wb') as f:
    f.write(html_obj2.data.encode("UTF-8"))

# Open the show prediction stored HTML file
url2 = r'D:\MaritimeTrade\ML\lightGBM\Model_Explanations\best_bayes_train20test20_HPO_300_Prediction.htm'
webbrowser.open(url2, new=2)

# Explain prediction
#explanation_pred = eli.explain_prediction(best_bayes_model, np.array(X_test)[1])
#explanation_pred

###############################################################################
# Set path for ML results
path = r'D:\MaritimeImportsExports\ML\lightGBM\bestBayes_WeightsExplain'
os.chdir(path)

# Explain weights
explanation = eli.explain_weights_sklearn(perm_importance,
                            feature_names = X_train.columns.tolist())
exp = format_as_dataframe(explanation)

# Write processed data to csv
exp.to_csv('best_bayes_train20test20_HPO_300_WeightsExplain.csv',
           index=False, encoding='utf-8-sig')

###############################################################################
##################### Test trained model on 2020 on 2019 ######################
###############################################################################
# Prepare 2019 to fit model train on 2020
X_test1 = df1.drop(['Metric_Tons'], axis=1)
y_test1 = df1['Metric_Tons']

# Drop year variable
X_test1 = X_test1.drop(['DateTime_YearWeek'], axis=1)

# Testing set: Encode variables using ranking - ordinal               
ce_ord = ce.OrdinalEncoder(cols = ['foreign_company_size', 'US_company_size'])
X_test1 = ce_ord.fit_transform(X_test1)

# Test: Create dummy variables for categorical variables
X_test1 = pd.get_dummies(X_test1, drop_first=True)

# MinMax Scaling
X_test1 = pd.DataFrame(mn.fit_transform(X_test1), columns = X_test1.columns)

# Model Metrics
y_test_pred = best_bayes_model.predict(X_test1)

print('\nModel Metrics for lightGBM HPO Train 2020 Test 2019')
print('MAE train: %.3f, test: %.3f' % (
        mean_absolute_error(y_train, y_train_pred),
        mean_absolute_error(y_test1, y_test_pred)))
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test1, y_test_pred)))
print('RMSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred, squared=False),
        mean_squared_error(y_test1, y_test_pred, squared=False)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test1, y_test_pred)))

###############################################################################
# Set path for ML results
path = r'D:\MaritimeImportsExports\ML\lightGBM\Model_Explanations'
os.chdir(path)

# Model metrics with Eli5
# Compute permutation feature importance
perm_importance = PermutationImportance(best_bayes_model,
                                        random_state=seed_value).fit(X_test1,
                                                                     y_test1)
                                                                     
# Store feature weights in an object
html_obj = eli.show_weights(perm_importance,
                            feature_names = X_train.columns.tolist())

# Write feature weights html object to a file 
with open(r'D:\MaritimeTrade\ML\lightGBM\Model_Explanations\best_bayes_train20test19_HPO_300_WeightsFeatures.htm',
          'wb') as f:
    f.write(html_obj.data.encode("UTF-8"))

# Open the stored feature weights HTML file
url = r'D:\MaritimeTrade\ML\lightGBM\Model_Explanations\best_bayes_train20test19_HPO_300_WeightsFeatures.htm'
webbrowser.open(url, new=2)

# Show prediction
html_obj2 = show_prediction(best_bayes_model, X_train.iloc[1],
                            show_feature_values=True)

# Write show prediction html object to a file 
with open(r'D:\MaritimeTrade\ML\lightGBM\Model_Explanations\best_bayes_train20test19_HPO_300_Prediction.htm',
          'wb') as f:
    f.write(html_obj2.data.encode("UTF-8"))

# Open the show prediction stored HTML file
url2 = r'D:\MaritimeTrade\ML\lightGBM\Model_Explanations\best_bayes_train20test19_HPO_300_Prediction.htm'
webbrowser.open(url2, new=2)

# Explain prediction
#explanation_pred = eli.explain_prediction(best_bayes_model, np.array(X_test)[1])
#explanation_pred

###############################################################################
# Set path for ML results
path = r'D:\MaritimeImportsExports\ML\lightGBM\bestBayes_WeightsExplain'
os.chdir(path)

# Explain weights
explanation = eli.explain_weights_sklearn(perm_importance,
                            feature_names = X_train.columns.tolist())
exp = format_as_dataframe(explanation)

# Write processed data to csv
exp.to_csv('best_bayes_train20test19_HPO_300_WeightsExplain.csv',
           index=False, encoding='utf-8-sig')

###############################################################################