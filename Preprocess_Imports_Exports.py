# -*- coding: utf-8 -*-
"""
@author: aschu
"""
import os
import random
import numpy as np
import sys
import glob
import pandas as pd
import seaborn as sns
import datapackage
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

# Set seed
seed_value = 42
os.environ['MaritimeTrade_Preprocessing'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

# Write results to log file
stdoutOrigin=sys.stdout 
sys.stdout = open('MaritimeTrade_preprocess_log.txt', 'w')

print('\nMaritime Trade Preprocess into Data Warehouse') 
print('======================================================================')

# Concat all files in directory
# Imports
path = r'D:\Maritime Trade\Data\Imports'
os.chdir(path)

extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

# Combine all files in the list
imports = pd.concat([pd.read_csv(f) for f in all_filenames ])
imports = imports.drop(['Unnamed: 18','In bond entry type', 'Vessel Country'], 
                       axis=1)

# Rename columns
imports.rename(columns={'Consignee (Unified)':'US_Company',
                        'HS Description':'HS_Description',
                        'Country of Origin':'Foreign_Country',
                        'Port of Departure':'Foreign_Port', 
                        'Port of Arrival':'US_Port',
                        'Shipper (Unified)':'Foreign_Company',  
                        'Container LCL/FCL':'Container_LCL/FCL',
                        'Metric Tons':'Metric_Tons',  
                        'Container Type Refrigerated':'Container_Type_Refrigerated', 
                        'Container Type Dry': 'Container_Type_Dry',
                        'VIN Quantity':'VIN_Quantity'}, inplace=True)

print('\nDimensions of Imports:', imports.shape) ##38049723, 16)
print('======================================================================')

##############################################################################
# Exports
path = r'D:\Maritime Trade\Data\Exports'
os.chdir(path)

extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

# Combine all files in the list
exports = pd.concat([pd.read_csv(f) for f in all_filenames ])
exports = exports.drop(['Unnamed: 16', 'Vessel Country'], 
                       axis=1)

# Rename columns
exports.rename(columns={'Exporter (Unified)':'US_Company',
                        'HS Description':'HS_Description', 
                        'Country of Final destination':'Foreign_Country',
                        'Foreign Port':'Foreign_Port', 'US Port':'US_Port', 
                        'Container LCL/FCL':'Container_LCL/FCL',
                        'Metric Tons':'Metric_Tons', 
                        'VIN Quantity':'VIN_Quantity'}, inplace=True)

print('\nDimensions of Exports:', exports.shape) #(24377605, 16)
print('======================================================================')

##############################################################################
##############################################################################
##############################################################################

# =============================================================================
# Prepreocess Exports Data
# 
# 1. Create a “Trade Direction” attribute, assigning the value of 'Export' to all records.
# 
# 2. Create Foreign_Company attribute, assigning the value of 'Not Applicable to Exports' to all records.
# 
# 3. Use the “Short Container Description” to create binary values.  
#     a. If “*REEF*” and similar words exists in “Short Container Description”, create a new variable “Container Type Refrigerated” and assign the value a 1. 
#     
#     b. If “Short Container Description” has a value, but does not included “*REEF*”, create a variable called “Container Type Dry” 
#     and assign it a value of 1.  Then drop the “Short Container Description” from Exports.
# =============================================================================

df = exports

# Select non missing Short Container Description 
df = df[df['Short Container Description'].notna()]

df = df.copy()
df.loc[:,'Trade_Direction'] = 'Export'
df.loc[:,'Foreign_Company'] = 'Not Applicable to Exports'

# Subset containerized = 1
dat = df.loc[df['Containerized'] == 1]

# Select rows by list of values to create new feature
search_values = ['REEF','Cold', 'Deg']
dat3 = dat[dat['Short Container Description'].str.contains(
    '|'.join(search_values))]

dat3 = dat3.copy()
dat3.loc[:,'Container_Type_Refrigerated'] = 'True'
dat3.loc[:,'Container_Type_Dry'] = 'False'

# Select Non REEF data
dat4 = dat[~dat['Short Container Description'].str.contains('|'.join(search_values))]

dat4 = dat4.copy()
dat4.loc[:,'Container_Type_Refrigerated'] = 'False'
dat4.loc[:,'Container_Type_Dry'] = 'True'

#Concatenate data with new features
dat = pd.concat([dat3, dat4])

##############################################################################
# Subset containerized = 0
dat2 = df.loc[df['Containerized'] == 0]

# Feature engineer containerized into new vars 
dat2 = dat2.copy()
dat2.loc[:,'Container_Type_Refrigerated'] = 'False'
dat2.loc[:,'Container_Type_Dry'] = 'False'

##############################################################################
# Concatenate data with new features
df = pd.concat([dat, dat2])

# Remove short container description since not needed
df =  df.drop(['Short Container Description', 'Containerized'], axis=1)

exports = df

# Remove data not needed
del df, dat, dat2, dat3, dat4

##############################################################################
# =============================================================================
# Prepreocess Imports Data
# 
# 1. Create a “Trade Direction” attribute, assigning the value of 'Import' to all records.
# 
# 2. Use  “Container_Type_Refrigerated” and "Container_Type_Dry" to create a binary feature by grouping all the observations with 1 or more into one group.
# =============================================================================

df = imports

df = df.copy()
df.loc[:,'Trade_Direction'] = 'Import'

mask = df['Container_Type_Refrigerated'] >= 1
df.loc[mask, 'Container_Type_Refrigerated'] = 1

mask = df['Container_Type_Dry'] >= 1
df.loc[mask, 'Container_Type_Dry'] = 1

imports = df

# Remove data not needed
del df

##############################################################################
# Concatenate Imports and Exports
df = pd.concat([imports, exports])
print('\nDimensions of Concatenated Imports & Exports:', df.shape) #(62279097, 16))
print('======================================================================')

# Remove data not needed
del imports, exports

# Convert variables to T/F
df = df.copy()
df['Container_Type_Refrigerated'] = df['Container_Type_Refrigerated'].astype('str')
df['Container_Type_Refrigerated'] = df['Container_Type_Refrigerated'].replace('0','False')
df['Container_Type_Refrigerated'] = df['Container_Type_Refrigerated'].replace('1','True')
# 
df['Container_Type_Dry'] = df['Container_Type_Dry'].astype('str')
df['Container_Type_Dry'] = df['Container_Type_Dry'].replace('0','False')
df['Container_Type_Dry'] = df['Container_Type_Dry'].replace('1','True')

###############################################################################
# =============================================================================
# Preprocess Concatenated Imports/Exports Data
# 
# 1. Create a new variable, “HS_Class”, pulling the first two digits from “HS Description” (NOT “HS”!).  
#   The HS_Code_Dictionary entity has cleaner, more homogenized labels for the HS family classes.
# 
# 2. Create a new variable called HS_Mixed” as a Boolean value, TRUE if there are more than 6 digits in “HS” and FALSE if there are 6 or less digits in “HS”.
# 
# 3. Separate US_Company into two attributes, at the comma furthest to the right.  There should be two digits after that comma that indicate the company’s state.
#   The new variable is “US_Company_State. Aggregrate the same compaies together by removing state info
#
# 4.  Foreign_Company split into new var by Country, at the comma furthest to the right. Parentheses are needed to be remove first. 
# There should be two digits after that comma that indicate the company’s country. The new variable is Foreign_Company_Country”.
# =============================================================================

# Remove missing date observations due to the download from Descartes
df = df[df['Date'].notna()]

##############################################################################
# Extract first 2 characters from a string to create HS_Class
df['HS_Class'] = [x[:2] for x in df['HS_Description']]

#Drop description since feature engineered HS_Class
df = df.drop(['HS_Description'], axis=1)

##############################################################################
# Filter HS_Class for 23 selected to examine
df = df.copy()
df['HS_Class'] = pd.to_numeric(df['HS_Class'], errors='coerce')

df = df[df['HS_Class'].notna()]

# Subset HS Classes 
list = [2,7,8,9,10,11,14,15,16,17,18,19,20,22,24,30,40,47,72,73,87,94,95]

df = df[df['HS_Class'].isin(list)]

print('\nNumber of Unique :', df[['HS_Class']].nunique()) #23
print('======================================================================')

# Convert HS_Class to string
df['HS_Class'] = df['HS_Class'].astype('str')

##############################################################################
# Create a new variable HS_Mixed as a Boolean value, TRUE if there are more than 6 digits in “HS” and FALSE if there are 6 or less digits in “HS”.
# Length 6 is actually 7
mask = df['HS'].astype(str).str.len() <= 7
df.loc[mask, 'HS_Mixed'] = 'False'

mask = df['HS'].astype(str).str.len() > 7
df.loc[mask, 'HS_Mixed'] = 'True'

df = df.drop(['HS'], axis=1)

##############################################################################
# Create HS_Groups linking grouping similar HS_class items together to reduce dimensionality
path = r'D:\Maritime Trade\Data\Warehouse_Construction'
os.chdir(path)

# Merge HS Groups
HS_Groups = pd.read_csv('HS_Groups.csv', sep=',', index_col=False)
HS_Groups = HS_Groups.drop(['Description'], axis = 1)
HS_Groups['HS_Class'] = HS_Groups['HS_Class'].astype(float)

df['HS_Class'] = df['HS_Class'].astype(float)

df = pd.merge(df, HS_Groups, how = 'left', left_on = ['HS_Class'], 
              right_on = ['HS_Class'])
df = df.drop_duplicates()

# Remove data not needed
del HS_Groups

##############################################################################
# Filter US_Company into different dfs because of missing US_Company
dat = df[df['US_Company'].notna()]

dat2 = df[df['US_Company'].isna()]

# Missing data is converted from NaN to below
dat2 = dat2.copy()
dat2.loc[:,'US_Company'] = 'Not Provided'
dat2.loc[:,'US_Company_State'] = 'Not Provided'
dat2.loc[:,'comma_count_US_Company'] = 'Not Applicable'

##############################################################################
# Separate US_Company into two attributes, at the comma furthest to the right creating “US_Company_State”
dat = dat.copy()
dat['US_Company_State'] = [x.rsplit(",", 1)[-1] for x in dat['US_Company']]

print('\nNumber of Unique :', dat[['US_Company_State']].nunique()) #75
print('======================================================================')

##############################################################################
# Concatenate data
df = pd.concat([dat, dat2])

# Remove data not needed
del dat, dat2

df = df.drop(['comma_count_US_Company'], axis=1)

##############################################################################
# Remove everything after last comma to aggregrate companies
df = df.copy()
df.loc[:,'US_Company_Agg'] = [x.rsplit(",", 1)[:-1] for x in df['US_Company']]
df['US_Company_Agg'] = df['US_Company_Agg'].str[0]

print('\nNumber of Unique of US_Company_Aggregrated is:', 
      df[['US_Company_Agg']].nunique())
print('======================================================================')

##############################################################################
# Filter Foreign_Company into different dfs because of missing Foreign_Company
dat = df[df['Foreign_Company'].notna()]

dat2 = df[df['Foreign_Company'].isna()]
dat2 = dat2.copy()

# Missing data is converted from NaN to below
dat2.loc[:,'Foreign_Company'] = 'Not Provided'
dat2.loc[:,'Foreign_Company_Country'] = 'Not Provided'

##############################################################################
# Remove all parentheses
dat = dat.copy()
dat.loc[:,'Foreign_Company_Country'] =  dat['Foreign_Company'].str.replace(r'[\(\)\d]+', '')

# Extract first 2 characters from a string 
dat['Foreign_Company_Country'] = [x[-2:] for x in dat['Foreign_Company_Country']]
     
print('\nNumber of Unique of Foreign_Company_Country is:', 
      dat[['Foreign_Company_Country']].nunique()) 
print('======================================================================')

#Concat data with foreign company present and missing
df = pd.concat([dat, dat2])

# Remove data not needed
del dat, dat2

##############################################################################
# Extract names for clustering using OpenRefine and binning from original data
path = r'D:\Maritime Trade\OpenRefine_ClusteringNames'
os.chdir(path)

#the columns we want distinct values from
us_companies = pd.DataFrame({'US_Company': df['US_Company'].unique()})
us_companies.to_csv('us_companies.csv', index=False, encoding='utf-8-sig')

foreign_country = pd.DataFrame({'Foreign_Country': df['Foreign_Country'].unique()})
foreign_country.to_csv('foreign_country.csv', index=False, encoding='utf-8-sig')

foreign_port = pd.DataFrame({'Foreign_Port': df['Foreign_Port'].unique()})
foreign_port.to_csv('foreign_port.csv', index=False, encoding='utf-8-sig')

US_port = pd.DataFrame({'US_Port': df['US_Port'].unique()})
US_port.to_csv('us_port.csv', index=False, encoding='utf-8-sig')

carrier = pd.DataFrame({'Carrier': df['Carrier'].unique()})
carrier.to_csv('carrier.csv', index=False, encoding='utf-8-sig')

foreign_company = pd.DataFrame({'Foreign_Company': df['Foreign_Company'].unique()})
foreign_company.to_csv('foreign_company.csv', index=False, encoding='utf-8-sig')

foreign_company_country = pd.DataFrame({'Foreign_Company_Country': df['Foreign_Company_Country'].unique()})
foreign_company_country.to_csv('foreign_company_country.csv', index=False, encoding='utf-8-sig')

##############################################################################
# Generate a map for ISO standard country names and abbreviations
data_url = 'https://datahub.io/core/country-list/datapackage.json'

# Load Data Package into storage
package = datapackage.Package(data_url)

# Load only tabular data
resources = package.resources
for resource in resources:
    if resource.tabular:
        data = pd.read_csv(resource.descriptor['path'])
        print (data)
        
data.to_csv('country_map.csv', index=False, encoding='utf-8-sig')

###############################################################################
###################### Perform clustering using OpenRefine ####################
###############################################################################
# Merge data with the clustered names generated by OpenRefine
path = r'D:\Maritime Trade\Data\Keys_and_Dictionaries'
os.chdir(path)

# Merge us_companies_clustered
us_company_clustered = pd.read_csv('us_companies_clustered.csv')

df = pd.merge(df, us_company_clustered, left_on='US_Company', 
              right_on='US_Company')
df = df.drop(['US_Company'], axis=1)
df = df.drop_duplicates()

# Merge foreign_company_clustered
foreign_company_clustered = pd.read_csv('foreign_company_clustered.csv')

df = pd.merge(df, foreign_company_clustered, left_on='Foreign_Company', 
              right_on='Foreign_Company')
df = df.drop(['Foreign_Company'], axis=1)
df = df.drop_duplicates()

##############################################################################
# Merging carrier binned data
carrier = pd.read_csv('carrier_clustered_binned.csv', index_col=False)

df = pd.merge(df, carrier, left_on = 'Carrier', right_on = 'Carrier')
df = df.drop(['Carrier_Clustered', 'Carrier'], axis=1)
df = df.drop_duplicates()

# Merge us_port_and_state_code_clustered_binned
us_port_and_state_code_clustered_binned = pd.read_csv('us_port_and_state_code_clustered_binned.csv')

df = pd.merge(df, us_port_and_state_code_clustered_binned, 
              left_on='US_Port',right_on='US_Port')
df = df.drop_duplicates()
df.rename(columns={'STATE_CODE':'US_Company_State'}, inplace=True)

# Merge foreign_company_country_clustered
foreign_company_country_clustered = pd.read_csv('foreign_company_country_clustered.csv')

df = pd.merge(df,foreign_company_country_clustered, 
              left_on='Foreign_Company_Country',
              right_on='Foreign_Company_Country')
df = df.drop(['Code'], axis=1)
df = df.drop_duplicates()
df.rename(columns={'Name':'Foreign_Company_Country_Clustered',
                   'Continent':'Foreign_Company_Country_Continent'}, 
          inplace=True)

# Merge foreign_country_clustered
foreign_country_clustered = pd.read_csv("foreign_country_clustered.csv")

df = pd.merge(df, foreign_country_clustered, left_on='Foreign_Country', 
              right_on='Foreign_Country')
df = df.drop(['Code'], axis=1)
df = df.drop_duplicates()
df.rename(columns={'Name':'Foreign_Country_Name_Clustered',
                   'Continent':'Foreign_Country_Continent'}, inplace=True)

# Merge to be applied AFTER the foreign country clustered as this takes only the standard names as a key
country_continent_region_key = pd.read_csv('country_continent_region_key.csv')

# Grab the regions for both foreign country fields one at a time
df = pd.merge(df, country_continent_region_key, 
              left_on = ['Foreign_Country_Name_Clustered'], 
              right_on = ['Name'])
df = df.drop([ 'Name', 'Country Code', 'Continent'], axis=1)
df = df.drop_duplicates()
df.rename(columns={'Region':'Foreign_Country_Region'}, inplace=True)

# Left merge to get Foreign_Company_Country_Region
country_continent_region_key = country_continent_region_key.drop(
    ['Country Code', 'Continent'], axis=1)

df = pd.merge(df, country_continent_region_key, how='left', 
              left_on=['Foreign_Company_Country_Clustered'],right_on=['Name'])
df = df.drop([ 'Name'], axis=1)
df = df.drop_duplicates()
df.rename(columns={'Region':'Foreign_Company_Country_Region'}, inplace=True)

# Remove data not needed
del us_company_clustered, foreign_company_clustered, carrier
del us_port_and_state_code_clustered_binned, foreign_company_country_clustered 
del foreign_country_clustered, country_continent_region_key

##############################################################################
# =============================================================================
# # Postprocessing after merging results from clustering in OpenRefine
# 
# 1. Filter US companies with at least 100 metric tons total over the time period
# 
# 2. Conditional Binning of US Company by Metric Tons for Mapping
# 
# 3. Filter foreign companies with at least 100 metric tons total over the time period
#
# 4. Conditional Binning Foreign Company by Metric Tons for Mapping
# 
# 5. Clean variables with inconsistencies
# =============================================================================

# All US companies that trade less than 100 metric tons are omitted from the dataset
df = df.copy()
df1 = df.loc[:,['US_company_Clustered', 'Metric_Tons']]

df1 = df1.groupby('US_company_Clustered')['Metric_Tons'].sum().reset_index()

df1 = df1.loc[df1['Metric_Tons'] > 100]
print('\nDimensions of US Company with Metric Tons over 100:', df1.shape) 
print('======================================================================')

# Rename total metric tons for US companies
df1.rename(columns = {'Metric_Tons': 'Metric_Tons_Totals'}, inplace=True)

##############################################################################
# =============================================================================
# Conditional Binning of US and Foreign Company by Metric Tons for Mapping
#
# Creating the bins for the mappings:
#     a. micro =  100-1000
#     b. small = 1001-10000
#     c. medium = 10001-100000 
#     d. large = 100001-1000000 
#     e. huge = 1000000+ 
# =============================================================================

us_company_data = df1

# Function to bin metric tonnage into specified groups
def tonnage_binner(df):
#create a list of our conditions
    conditions = [
        (df['Metric_Tons_Totals'] <= 1000),
        (df['Metric_Tons_Totals'] > 1000) & (df['Metric_Tons_Totals'] <= 10000),
        (df['Metric_Tons_Totals'] > 10000) & (df['Metric_Tons_Totals'] <= 100000),
        (df['Metric_Tons_Totals'] > 100000) & (df['Metric_Tons_Totals'] <= 1000000),
        (df['Metric_Tons_Totals'] > 1000000),
        (df['Company'].str.strip() == 'NOT AVAILABLE')  
        ]

    # Create a list of the values to assign for each condition
    values = ['micro','small', 'medium', 'large', 'huge','unknown']

    # Create a new column and use np.select to assign values to it using lists as arguments
    df['company_size'] = np.select(conditions, values)
    return(df)

# Rename the columns to work in binner function
us_company_data = us_company_data.rename(columns={'US_company_Clustered':'Company'})

# Making the bins
us_company_data = tonnage_binner(us_company_data)

# Attach the bins to the mapping files to both drop all companies below threshold and add the bins
us_company_data = us_company_data.drop(['Metric_Tons_Totals'], axis=1)
us_company_data = us_company_data.rename(columns={'Company':'US_company_Clustered'})

# Merge binned data with main
df = pd.merge(df, us_company_data, how='right', 
              left_on=['US_company_Clustered'], 
              right_on=['US_company_Clustered'])
df = df.drop_duplicates()
df.rename(columns={'company_size':'US_company_size'}, inplace=True)

# Remove data not needed
del df1, us_company_data

##############################################################################
# All Foreign companies that trade less than 100 metric tons are omitted from the dataset
df = df.copy()
df1 = df.loc[:,['Foreign_Company_Clustered', 'Metric_Tons']]

df1 = df1.groupby('Foreign_Company_Clustered')['Metric_Tons'].sum().reset_index()

df1 = df1.loc[df1['Metric_Tons'] > 100]
print('\nDimensions of Foreign Company with Metric Tons over 100:', df1.shape) 
print('======================================================================')

# Rename total metric tons for foreign companies
df1.rename(columns = {'Metric_Tons': 'Metric_Tons_Totals'}, inplace=True)

foreign_company_data = df1

# Rename the columns to work in binner function
foreign_company_data = foreign_company_data.rename(columns={'Foreign_Company_Clustered':'Company'})

# Making the bins
foreign_company_data = tonnage_binner(foreign_company_data)

# Attach the bins to the mapping files to both drop all companies below threshold and add the bins
foreign_company_data = foreign_company_data.drop(['Metric_Tons_Totals'], 
                                                 xis=1)
foreign_company_data = foreign_company_data.rename(columns={'Company':'Foreign_Company_Clustered'})

# Merge binned data with main
df = pd.merge(df, foreign_company_data, how='right', 
              left_on=['Foreign_Company_Clustered'], 
              right_on=['Foreign_Company_Clustered'])
df = df.drop_duplicates()
df.rename(columns={'company_size':'foreign_company_size'}, inplace=True)

# Remove data not needed
del df1, foreign_company_data

##############################################################################
# Isolate US port state to use state abbreviation as a key to match with the full state names
df = df.copy()
df['US_Port_State'] = df['US_Port_Clustered'].str.rsplit(',').str[-1] 

# Standardized state and territory codes
df['US_Port_State'] = df['US_Port_State'].replace(['NEW YORK',' NY',' FL',
                                                   ' TX','OH (DHL COURIER)',
                                                   ' VA', ' NJ', ' UT',' HI', 
                                                   ' MD', ' DC', ' AK',' WA', 
                                                   ' VI',' MI',' MT', ' FL)', 
                                                   ' ND', ' NM', ' PA', 
                                                   'VIRGIN ISLANDS', ' CO', 
                                                   ' NE', ' CA'],['NY', 'NY', 
                                                                  'FL', 'TX', 
                                                                  'OH', 'VA', 
                                                                  'NJ', 'UT', 
                                                                  'HI', 'MD', 
                                                                  'DC', 'AK', 
                                                                  'WA', 'VI', 
                                                                  'MI', 'MI', 
                                                                  'FL','ND',
                                                                  'NM', 'PA', 
                                                                  'VI', 'CO',
                                                                  'NE','CA'])

##############################################################################
# Errors with WI, MS)
df['US_Port_State'].mask(df['US_Port_State'] == ' WI', 'WI', inplace=True)
df['US_Port_State'].mask(df['US_Port_State'] == ' MS)', 'MS', inplace=True)
df['US_Port_State'].mask(df['US_Port_State'] == ' IA', 'IA', inplace=True)
df['US_Port_State'].mask(df['US_Port_State'] == ' INC.', 'ND', inplace=True)
df['US_Port_State'].mask(df['US_Port_State'] == 'ND', 'NOT DECLARED', 
                         inplace=True)

# Change names from the trade dataset keys to match changes made to names in the tariff data
df = df.copy()
df['Foreign_Country_Name_Clustered'] = df['Foreign_Country_Name_Clustered'].replace('Viet Nam', 'Vietnam')
df['Foreign_Country_Name_Clustered'] = df['Foreign_Country_Name_Clustered'].replace('Taiwan, Province of China', 'Taiwan')
df['Foreign_Country_Name_Clustered'] = df['Foreign_Country_Name_Clustered'].replace('Russian Federation', 'Russia')
df['Foreign_Country_Name_Clustered'] = df['Foreign_Country_Name_Clustered'].replace('Tanzania, United Republic of', 'Tanzania')
df['Foreign_Country_Name_Clustered'] = df['Foreign_Country_Name_Clustered'].replace('Bolivia, Plurinational State of', 'Bolivia')
df['Foreign_Company_Country_Clustered'] = df['Foreign_Company_Country_Clustered'].replace('Viet Nam', 'Vietnam')
df['Foreign_Company_Country_Clustered'] = df['Foreign_Company_Country_Clustered'].replace('Taiwan, Province of China', 'Taiwan')
df['Foreign_Company_Country_Clustered'] = df['Foreign_Company_Country_Clustered'].replace('Russian Federation', 'Russia')
df['Foreign_Company_Country_Clustered'] = df['Foreign_Company_Country_Clustered'].replace('Tanzania, United Republic of', 'Tanzania')
df['Foreign_Company_Country_Clustered'] = df['Foreign_Company_Country_Clustered'].replace('Bolivia, Plurinational State of', 'Bolivia')

# Change ND and ED to not declared
df['US_Company_State'].mask(df['US_Company_State'] == 'ND', 'NOT DECLARED', 
                            inplace=True)
df['US_Company_State'].mask(df['US_Company_State'] == 'ED', 'NOT DECLARED', 
                            inplace=True)

##############################################################################
# =============================================================================
# Outlier testing of quantitative variables
# 
# Filter the data where Metric_Tons, Teus and TCBUSD are within three standard deviations
# =============================================================================
# Metric_Tons
# To filter the DataFrame where Metric_Tons is within three standard deviations
df = df[((df.Metric_Tons - df.Metric_Tons.mean()) / 
         df.Metric_Tons.std()).abs() < 3.5]
print('\nDimensions after removing outliers in Metric Tons:', df.shape) 
print('======================================================================')

sns.boxplot(x=df['Metric_Tons']).set_title('Distribution of Metric Tons')
plt.savefig('Distribution_Metric_Tons_Afteroutliertesting.png', 
            bbox_inches='tight')

# Retain data that is in long right tail
df = df.loc[df['Metric_Tons'] < 250]
print('\nDimensions after removing data in right tail of Metric Tons:', 
      df.shape) 
print('======================================================================')

sns.boxplot(x=df['Metric_Tons']).set_title('Distribution of Metric Tons')
plt.savefig('Distribution_Metric_Tons_Afterrighttail.png', bbox_inches='tight')

##############################################################################
# Teus
# To filter the DataFrame where Teus is within three standard deviations
df = df[((df.Teus - df.Teus.mean()) / df.Teus.std()).abs() < 3.5]
print('\nDimensions after removing outliers in Teus:', df.shape) 
print('======================================================================')

sns.boxplot(x=df['Teus']).set_title('Distribution of Teus')
plt.savefig('Distribution_Teus_Afteroutliertesting.png', bbox_inches='tight')

##############################################################################
# Total calculated value (US$): rename to TCVUSD
# To filter the DataFrame where TCVUSD is within three standard deviations
df = df.rename(columns = {"Total calculated value (US$)": "TCVUSD"}, 
               inplace = True) 
df = df[((df['TCVUSD'] - df['TCVUSD'].mean()) / df['TCVUSD'].std()).abs() < 3.5]
print('\nDimensions after removing outliers in TCVUSD:', df.shape) 
print('======================================================================')

sns.boxplot(x=df['TCVUSD']).set_title('Distribution of Total calculated value (US$)')
plt.savefig('Distribution_TCVUSD_Afteroutliertesting.png', bbox_inches='tight')

df = df.drop_duplicates()

##############################################################################
##############################################################################
##############################################################################
# =============================================================================
# Merge Trade Data with Other Data Sources
# 
# 1. Unemployment
# 2. Tariffs
# 3. Exchange Rates
# 4. State mandated closure dates
# 5. COVID-19 cases and deaths in US states
# =============================================================================
path = r'D:\Maritime Trade\Data\Keys_and_Dictionaries'
os.chdir(path)

country_continent_region_key = pd.read_csv('country_continent_region_key.csv', 
                                           sep=',', index_col=False)

# Joining state abbreviation key table to trade dataset
df = pd.merge(df, country_continent_region_key, how='left', 
              left_on=['Foreign_Country_Code'], right_on=['Country Code'])
df = df.drop_duplicates()

# Remove data not needed
del country_continent_region_key

##############################################################################
# Joining unemployment table to trade dataset
path = r'D:\Maritime Trade\Data\Warehouse_Construction'
os.chdir(path)

unemployment = pd.read_csv('Unemployment Data 2010-2020.csv', sep=',', 
                           index_col=False)
unemployment['Year-Month'] = pd.to_datetime(unemployment['Year-Month'].astype(
    str), format='%Y-%m')
unemployment['Year-Month'] = unemployment['Year-Month'].dt.to_period('M')

df = pd.merge(df, unemployment, left_on='DateTime_YearMonth', 
              right_on='Year-Month')

df = df.drop(['Month', 'Year-Month', 'Total in Thousands'], axis=1)
df = df.drop_duplicates()

# Rename the Unemployment Rate Total to specify unemployment rate in "US".
df = df.copy()
df.rename(columns={'Unemployment Rate Total': 'US_Unemployment_Rate'}, 
               inplace=True)

# Remove data not needed
del unemployment

##############################################################################
# Merging tariff data
tariffs = pd.read_csv('Tariffs.csv')

df = pd.merge(df, tariffs, how='left', 
              left_on=['Foreign_Country_Name_Clustered','HS_Class','Year'], 
              right_on = ['Country','HS_Family','Tariff_Year'])
df = df.drop(['Country','Tariff_Year','HS_Family'], axis=1)
df = df.drop_duplicates()

# Reformat some vars
df = df.copy()
df['European_Union'] = df['European_Union'].astype('str')
df['European_Union'] = df['European_Union'].replace('No','False')
df['European_Union'] = df['European_Union'].replace('Yes','True')
df['Free_Trade_Agreement_with_US'] = df['Free_Trade_Agreement_with_US'].astype('str')
df['Free_Trade_Agreement_with_US'] = df['Free_Trade_Agreement_with_US'].replace('No','False')
df['Free_Trade_Agreement_with_US'] = df['Free_Trade_Agreement_with_US'].replace('Yes','True')

df['Average_Tariff'] = df['Average_Tariff'].str.replace(',','')
df['Average_Tariff'] = df['Average_Tariff'].astype('float64')

# Remove data not needed
del tariffs

##############################################################################
# Merging exchange data
currencies = pd.read_csv('2010 - 2021 Exchange Rates.csv')
currencies = currencies.drop(['Month','Exchange_Year','Open','High','Low',],
                             axis=1)
currencies['Month_Year'] = pd.to_datetime(currencies['Month_Year'].astype(
    str), format='%Y-%m')
currencies['Month_Year'] = currencies['Month_Year'].dt.to_period('M')

df = pd.merge(df, currencies, how='left', 
              left_on=['DateTime_YearMonth','Foreign_Country_Name_Clustered'], 
              right_on=['Month_Year','Country'])

df = df.drop(['Month_Year','Country','European_Union_Member',
              'Foreign_Country_Name_Clustered'],axis=1)
df = df.drop_duplicates()

# Remove data not needed
del currencies

##############################################################################
# Prepare to merge data with KFF state closure data
# The full state names exist in the COVID-19 counts and KFF-stay-at-home-order datasets
path = r'D:\Maritime Trade\Data\Keys_and_Dictionaries'
os.chdir(path)

state_abbreviation_key = pd.read_csv('State, Abbrev, Region key.csv', 
                                     sep=',', index_col=False)
state_abbreviation_key.rename(columns={'Region': 'State_Region'}, inplace=True)

# Joining state abbreviation key table to trade dataset
df = pd.merge(df, state_abbreviation_key, how='left', 
              left_on=['US_Port_State'], right_on=['State Code'])
df = df.drop(['State Code', 'Region'], axis=1)
df = df.drop_duplicates()

##############################################################################
# Merge KFF data
# Not an ideal merge, because variable is only relevant in 2020 
path = r'D:\Maritime Trade\Data\Warehouse_Construction'
os.chdir(path)

KFF_Statewide_Stay_at_Home_Orders = pd.read_csv('KFF_Statewide-Stay-at-Home-Orders.csv', 
                                                sep=',', index_col=False)

df = pd.merge(df, KFF_Statewide_Stay_at_Home_Orders, how='right', 
              left_on='State', right_on='State')
df = df.drop_duplicates()
df.rename(columns={'Date Announced': 'Date_Announced', 
                  'Effective Date': 'Effective_Date'}, inplace=True)

# Remove data not needed
del KFF_Statewide_Stay_at_Home_Orders

##############################################################################
# Merge COVID Weekly Data
COVID_cases = pd.read_csv('us-counties - NYTimes.csv', sep=',', 
                          index_col=False)

COVID_cases['Date'] = pd.to_datetime(COVID_cases['date']) - pd.to_timedelta(7, unit='d')

# Aggregrate cases and deaths due to COVID to weekly in each state
df1 = (COVID_cases.groupby(['state', pd.Grouper(key='Date', freq='W-MON')])
          .agg({'cases':'sum', 'deaths':'sum'}).reset_index())

df1.rename(columns={'Date':'Date_Weekly_COVID', 'cases':'cases_weekly', 
                    'deaths':'deaths_weekly'}, inplace=True)

df1['Year'] = df1['Date_Weekly_COVID'].dt.year

df1 = df1[df1.Year < 2021] # Remove 2021
df1 = df1.drop(['Year'], axis=1)

##############################################################################
# Joining state abbreviation key table to trade dataset.
df1 = pd.merge(df1, state_abbreviation_key, 
               how='left', left_on='US_Port_State', right_on='State')

df1 = df1.drop(['State','State Code', 'State_Region'], axis=1)
df1 = df1.drop_duplicates()
df1 = df1[df1['state'].notna()]

df1['DateTime_YearWeek'] = pd.to_datetime(df1['Date_Weekly_COVID'])
df1['DateTime_YearWeek'] = df1['DateTime_YearWeek'].dt.strftime('%Y-w%U')

# Convert Date to Time Variables
df = df.copy()
df['DateTime'] = pd.to_datetime(df['Date'].astype(int), format='%Y%m%d')
df['Year'] = df.DateTime.dt.year
df['DateTime_YearWeek'] = df['DateTime'].dt.strftime('%Y-w%U')
df = df.drop(['Date'], axis=1)

# Outer merge of main data with aggregrated weekly COVID-19 cases and deaths in US States
df = pd.merge(df, df1, how='outer', left_on=['State', 'DateTime_YearWeek'], 
              right_on=['US_Port_State','DateTime_YearWeek'])
df = df.drop(['Foreign_Country_Code', 'Name', 'Country Code', 
              'Continent'], axis=1)
df = df.drop_duplicates()
print('\nDimensions after merging to a data warehouse:', df.shape) 
print('======================================================================')

# Remove data not needed
del state_abbreviation_key, COVID_cases, df1

##############################################################################
# Outer merge resulted in time points where there were not trade in specific states so fill gaps with data
df1 = df[df['HS_Mixed'].notna()]

# Fill missing values for COVID-19 data with 0s
df1 = df1.copy()
df1['cases_weekly'] = df1['cases_weekly'].fillna(0)
df1['deaths_weekly'] = df1['deaths_weekly'].fillna(0)

##############################################################################
df2 = df[df['HS_Mixed'].isna()]
print('\nNumber of observations where no trade occurred :', df2.shape) 
print('======================================================================')

# Fill variables with like variables to retain grain
df2 = df2.copy()
df2['Year'] = df2['Date_Weekly_COVID'].dt.year
df2['Year'] = df2['Year'].astype('Int64')

df2['US_Port_State'] = df2['State Code']
df2['State'] = df2['state']

df2['DateTime_YearMonth'] = df2['Date_Weekly_COVID'].dt.to_period('M')

df2 = df2.copy()
df2['Year'] = df2['Year'].astype('object')
df2['DateTime'] = df2['DateTime'].astype('object')

# Fill missing data with ''
df2 = df2.fillna('')

quant = ['Teus', 'Metric_Tons', 'VIN_Quantity', 'Total calculated value (US$)',
       'Metric_Tons_Totals', 'US_Unemployment_Rate', 'Average_Tariff', 'Price']

# Fill quant vars with 0
df2 = df2.copy()
df2.loc[:, quant] = df2.loc[:, quant].replace('', 0)

# Concatenate main data with filled data due to missing trade time points
df = pd.concat([df1, df2])
df = df.drop(['State Code'], axis=1)

# Remove data not needed
del df1, df2

##############################################################################
# =============================================================================
# Feature Engineering COVID-19 associated vars
# 
# 1. Calculate the number of days between a state mandated closure was announced and made effective.
# 
# 2. Calculate percent change in weekly COVID-19 cases and deaths from the first time point in each state. 
# 
# =============================================================================
# Prepare efective and announced variables for creating the difference variable
df = df.copy()
df['Date_Announced'] = pd.to_datetime(df['Date_Announced'].astype(object), 
                                       format='%m/%d/%Y')
df[['US_Port_State', 'Date_Announced']].value_counts()

df['Effective_Date'] = pd.to_datetime(df['Effective_Date'].astype(object), 
                                       format='%m/%d/%Y')

# Create feature for the number of days between state closure announced vs effective
df['State_Closure_EA_Diff'] = (df.Effective_Date - df.Date_Announced).dt.days

# Drop variables not using
df = df.drop(['Date_Announced', 'Effective_Date'], axis = 1)

##############################################################################
# Set up for finding date where first cases occurred in each state
df1 = df[df['Date_Weekly_COVID'].isna()]

df2 = df[df['Date_Weekly_COVID'].notna()]

print('\nNumber of unique US states is:', 
      df2[['State']].nunique())
print('======================================================================')

df2 = df2.copy()
df2['Date_Weekly_COVID'] = pd.to_datetime(df2.Date_Weekly_COVID)

# Create pivot table of cases_weekly for the States with weekly COVID as index
df3 = df2.pivot_table(index='Date_Weekly_COVID', columns='State', 
                      values='cases_weekly', aggfunc=np.min)

# Fill missing case data with 0
df3.fillna(df3.fillna(0), inplace=True)

# Find first nonzero in each column = first date with cases
df3 = df3.ne(0).idxmax()
df3 = df3.to_frame()
df3 = df3.reset_index()
df3.rename(columns={0: 'Time0_StateCase'}, inplace=True)

# Merge so data has first occurrence of COVID cases in US states
df3 = pd.merge(df3, df2, how='left', left_on=['US_Port_State', 'Time0_StateCase'], 
               right_on=['US_Port_State', 'Date_Weekly_COVID'])
df3 = df3.drop_duplicates()

df3 = df3.copy()
df3['cases_state_firstweek'] = df3['cases_weekly']

df3 = df3.loc[:, ['US_Port_State', 'Time0_StateCase', 'Date_Weekly_COVID', 
                  'cases_state_firstweek']]
df3 = df3.drop_duplicates()

# Create key to join on
df3['Year_M'] = df3['Date_Weekly_COVID'].dt.year
df2['Year_M'] = df2['Date_Weekly_COVID'].dt.year

df3 = df3.drop(['Date_Weekly_COVID'], axis=1)
df3 = df3.drop_duplicates()

df2 = pd.merge(df2, df3, how='left', left_on=['US_Port_State','Year_M'], 
                right_on=['US_Port_State','Year_M'])

df2 = df2.drop(['Year_M'], axis=1)
df2 = df2.drop_duplicates()

# Calculate percent change from number of cases in each week compared to the first week in state
df2 = df2.copy()
df2['cases_pctdelta'] = df2.apply(lambda x: (x['cases_weekly'] 
                                             - x['cases_state_firstweek']
                                             / x['cases_state_firstweek'])*100, 
                                  axis=1)

# Add the missing covid data to concat nicely
df1 = df1.copy()
df1.loc[:,'Time0_StateCase'] = 'Not Applicable'
df1.loc[:,'cases_state_firstweek'] = 0
df1.loc[:,'cases_pctdelta'] = 0

df = pd.concat([df2, df1])

# Remove data not needed
del df1, df2, df3

##############################################################################
# Set up for finding date where first deaths occurred in each state
df1 = df[df['Date_Weekly_COVID'].isna()]

df2 = df[df['Date_Weekly_COVID'].notna()]

df2= df2.copy()
df2['Date_Weekly_COVID'] = pd.to_datetime(df2.Date_Weekly_COVID)

# Create pivot table of deaths_weekly for the States with weekly COVID as index
df3 = df2.pivot_table(index='Date_Weekly_COVID', columns='State', 
                      values='deaths_weekly', aggfunc=np.min)

# Fill missing case data with 0
df3.fillna(df3.fillna(0), inplace=True)

# Find first nonzero in each column = first date with deaths
df3 = df3.ne(0).idxmax()
df3 = df3.to_frame()
df3 = df3.reset_index()
df3.rename(columns={0: 'Time0_StateDeath'}, inplace=True)

# Merge so data has first occurrence of COVID deaths in US states
df3 = pd.merge(df3, df2, how='left', left_on=['US_Port_State', 'Time0_StateDeath'],
               right_on=['US_Port_State',' Date_Weekly_COVID'])
df3 = df3.drop_duplicates()

df3 = df3.copy()
df3['deaths_state_firstweek'] = df3['deaths_weekly']
df3 = df3.loc[:, ['US_Port_State', 'Time0_StateDeath', 'Date_Weekly_COVID',
                  'deaths_state_firstweek']]

# Create key to join on
df3['Year_M'] = df3['Date_Weekly_COVID'].dt.year
df2['Year_M'] = df2['Date_Weekly_COVID'].dt.year

df3 = df3.drop(['Date_Weekly_COVID'], axis=1)
df3 = df3.drop_duplicates()

df2 = pd.merge(df2, df3, how='left', left_on=['US_Port_State', 'Year_M'], 
                right_on=['US_Port_State', 'Year_M'])

df2 = df2.drop_duplicates()
df2 = df2.drop(['Year_M'], axis=1)

# Calculate percent change from number of deaths in each week compared to the first week in state
df2 = df2.copy()
df2['deaths_pctdelta'] = df2.apply(lambda x: (x['deaths_weekly'] 
                                              - x['deaths_state_firstweek'] 
                                              / x['deaths_state_firstweek'])*100, 
                                   axis=1)

# Add the missing COVID data to concat nicely
df1 = df1.copy()
df1.loc[:,'Time0_StateDeath'] = 'Not Applicable'
df1.loc[:,'deaths_state_firstweek'] = 0
df1.loc[:,'deaths_pctdelta'] = 0

df = pd.concat([df2, df1])

# Remove data not needed
del df1, df2, df3

##############################################################################
# Aggregrate metric_tons to weekly in each state
df['Date_Weekly_Agg'] = pd.to_datetime(df['DateTime']) 

new_df1 = (df.groupby(['State', pd.Grouper(key='Date_Weekly_Agg',
                                           freq='W-MON')])
          .agg({'Metric_Tons':'sum'}).reset_index())

new_df1.rename(columns={'Metric_Tons':'Metric_Tons_Weekly'}, inplace=True)
new_df1 = new_df1.drop_duplicates()

df = pd.merge(df, new_df1, how='left', left_on=['State', 'Date_Weekly_Agg'],
              right_on=['State', 'Date_Weekly_Agg'])
df = df.drop_duplicates()

# Remove data not needed
del new_df1

##############################################################################
# Drop variables not using
df = df.drop(['HS_Class', 'VIN_Quantity', 'US_company_Clustered', 
              'Foreign_Company_Clustered','Foreign_Company_Country_Clustered', 
              'US_Company_State', 'US_Port_State', 'state', 'State_Region'], 
             axis=1)
df = df.drop_duplicates()

print('\nDimensions of Data Warehouse for EDA:', df.shape)  
print('======================================================================')

print('\nMissing Data in Warehouse :')
print(df.isna().sum())  
print('======================================================================')

# Close to create log file
sys.stdout.close()
sys.stdout=stdoutOrigin

# Write processed data to csv
df.to_csv('combined_trade.csv', index=False, encoding='utf-8-sig')
##############################################################################
# Now perform EDA for variable selection