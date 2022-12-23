 # Maritime Trade

# Background
The United States has been one of the greatest advocates for free trade since the end of WWII, and the importance of it to the United States economy cannot be underestimated. As many as 40 million jobs in the United States are directly supported by international trade as of 2018 (Trade Partnership LLC), and as much as half of the entire U.S. manufacturing industry or 6.3 million jobs is supported by exports (International Trade Administration, 2018). Imports introduce new goods, and boost the annual earning power of the average American by over $18,000 annually (Hufbauer and Lu, 2017).  

COVID-19 disrupted the global supply chain.  The first effect was a dramatic and rapid reduction in the labor force in affected locations, as workers fell ill or were forced to quarantine (Cambridge University Press, 2020). International trade was impacted further as borders were closed to some commerce and travel (Kerr, 2020). This further exacerbated supply chain disruptions within and among countries, as goods themselves were prioritized differently by those in quarantine.  Most international trade tanked dramatically, especially for manufactured goods and automobiles. However, food and agriculture supply lines were relatively unaffected in the short term, due to narrow time windows between harvest and consumption, and the fact that even during a pandemic, the population still needs to eat (Kerr, 2020) 

It is still too early to determine if COVID-19 has permanently disrupted the global supply chain, or if it has just hastened trends already underway, related to the geography of trading partners and the basket of goods traded among them.  These long-term trends will need to be discerned over years, isolating the effect of any prolonged economic recession, especially among countries that are not employing economic stimulus measures. The applied tariff rates of countries is one change that may or may not prove transitory or affected by the virus.  These rates may have a confounding effect with the disruptions caused by the direct or indirect measures of COVID-19.  It is still too early to tell if the pandemic will lead to a more decentralized global economic system, but this is a possible long term scenario (Cambridge University Press, 2020).

## Proposed questions and solutions to guide our project
-	How has the composition and/or volume of maritime imports and exports from the US changed over time?  This will be addressed with analytics.

-	Are there any confounding effects impacting the volume of imports and exports of the targeted commodities?

-	How did COVID-19 impact the volume and composition of international maritime trade?

# Data Collection:
The questions presented in the analysis are complex, interconnected, and require a variety of data sources to answer sufficiently. Data must establish historical import and export volumes and composition . There also must be data that demonstrate COVID-19’s effects on the supply chains and workforce. Finally, data must be included that establishes potential confounding effects on trade volumes, including historical rates for applied tariffs, currency exchange, and unemployment. 

## Import & Export Datasets
The U.S. Customs and Border Protection collects information about each shipment. Descartes Datamyne, as well as other industry data sources such as IHS Markit’s PIERS and Panjiva, compile this information and enhance it with additional data fields. This project queried data from Descartes Datamyne databases for U.S. imports and exports separately,  from January 1, 2010 through December 31, 2020.  The geographic scope includes all maritime trade shipments through the United States’ ports, to and from all trading partners. The data wss downloaded in a 70 queries of 1 million records or less, separately for  imports and exports, on February 4, 2020.

### Macroscopic Details	
-	Imports - The data contains 38,048,999 rows and 17 columns containing date of import arrival, business name and location, product identifying features, port of arrival and departure, source country, and other characteristics relating to the shipment.  

-	Exports - The data 9,289,307 rows and 14 columns containing similar data types as the import data.

Datasets will be concatenated into one table. Data consists of all bills of lading (shipment receipts) into or out of a U.S. maritime port from January 1, 2010 through December 31, 2020. Maritime trade is best measured in volumes, such as metric tonnage, TEUs, or total shipments, however the last two are relevant to maritime trade only. Trade in metric tonnage is a major indicator of the economic vitality of the national and global economy.

## COVID-19 Cases and Deaths in U.S. States
The New York Times has released, and is continuing to release, a series of data related to COVID-19 from state and local governments as well as health departments. This information is obtained by journalists employed by the newspaper company across the US who are actively tracking the development of the pandemic. This source  compiles information related tof COVID-19 across the county and state level in the United States over time. The data starts with the first reported COVID-19 case in the state of Washington on January 21, 2020. The total number of cases and deaths is reported as either “confirmed” or “probable” COVID-19. The number of cases includes all reported cases, including instances of individuals who have recovered or died. There are geographic exceptions in the states of New York, Missouri, Alaska, California, Nebraska, Illinois, Guam and Puerto Rico. Data was pulled from the NY Times github page https://github.com/nytimes/covid-19-data on February 5, 2021. 
### Macroscopic Details	
The data ranges from January 21, 2020 to February 5, 2021 with the dimensions of 1,001,656 rows and 6 columns including date, county, state, fips (geographic indicator), cases and number of deaths.

## COVID-19 State Mandated Closures
The US federal government enabled the states to declare the official closure of public areas and services with the goal of flattening the growth rate of new COVID-19 cases. This would reduce person-to-person contact since the transmission route of COVID-19 includes being able to be aerosolized via droplets. This included working-from-home options to maintain daily operations by working remotely. 

Source: Kaiser Family Foundation https://www.kff.org/policy-watch/stay-at-home-orders-to-fight-covid19/, initially compiled on April 5, 2020. The data consists of 51 rows and 3 columns containing US state, date announced and effective closure date.

## Tariffs
The World Trade Organization compiles the annual applied tariff rates that member countries set towards the world, on the harmonized service code level.  The dataset includes the number of tariff line items, annual averages, and minimum and maximum bounds.  Countries were selected based on the United States’ top trading partners, measured in metric tonnage, during the year 2019.  Tariff rates were collected for these countries for the years 2010 through 2020. This dataset also notes if the U.S. had a free trade agreement with these top trading partners for each of the 11 years.  This qualitative variable will compliment our average applied tariff rate by HS chapter class. 

Source: The World Trade Organization via the data query tool: http://tariffdata.wto.org/ReportersAndProducts.aspx, downloaded on February 8, 2021.

Source: Office of the United States Trade Representative: https://ustr.gov/issue-areas/industry-manufacturing/industrial-tariffs/free-trade-agreements, downloaded on February 11, 2021.
### Macroscopic Details	
The original datasource included only the primary countries and 18 attributes.  Subsequent additions reduced the working attributes to six, but expanded the scope of countries involved, increasing total records from 4,853 to 11,892. The United States and 62 other countries are included.  These are the major trading partners of the U.S., as of 2019, measured in metric tonnage.  All European countries are included, regardless of their individual trade numbers with the U.S.   Data was collected from the WTO and the Office of the United States Trade Representative to create a new table.  Each year includes 23 records for average applied tariff rates for the HS chapter classes included in this project.  There are additional attributes to indicate whether a free trade agreement exists between the country and the U.S., and whether that country is a member of the European Union. There is missing annual data if a country was not a member of the World Trade Organization (“WTO”) for any of the time period, if it was a member of the European Union (they are reported as simply the European Union), and if they have not yet submitted their 2019 or 2020 data to the WTO.  This was later adjusted by assuming that the applied tariff rates were held constant for these countries if no updates were published.

## Currency
Currency rates are extracted from the website investing.com which collects live rates for currencies, currency codes, financial news, commodities and crypto currencies. The data set contains eight columns (month, year, open price, low, high, currency code and country) and 2,145 rows. The timeframe is from January 2010 to Feb 2021. All countries that are included in the tariff table are included in the currency dataset.

Source: Stock Market Quotes & Financial News. (n.d.). Retrieved February 12, 2021, from https://www.investing.com/.  

### Macroscopic Details	
The data queried has 8,175 records and 10 attributes using a time frame from January 2010 to Feb 2021 The currency rate is an important variable to address our research questions as it has a great impact on imports and exports. If a currency rate increases, importing and exporting volumes may fall for a country as the cost of goods is too great and may look elsewhere to source. As we build our model, we would like to understand if currencies have any correlation with the other variables. 

## Unemployment	
The Bureau of Labor Statistics conducts a Current Population Survey every month of households in the United States. The survey is carried out to create datasets encompassing the labor force, employment, unemployment, people not included in the labor force, and other labor force statistics. The collection of the survey data was impacted by the COVID-19 pandemic. The data was collected by workers conducting the survey by telephone while the Bureau of Labor Statistics encouraged businesses to submit their data electronically. The dataset shows the raw count of national civilian unemployment by month (in thousands of people) as well as the national civilian unemployment rate also by month.

Source: United States, BLS. “Charts Related to the Latest ‘The Employment Situation’ News Release   |   More Chart Packages.” U.S. Bureau of Labor Statistics, U.S. Bureau of Labor Statistics, 9 Feb. 2021, www.bls.gov/charts/employment-situation/civilian-unemployment-rate.htm. 

### Macroscopic Details	
The data consists of 133 records of 3 attributes including a date for the national civilian unemployment data for the United States only from January 2010 through December 2020. With the onset of the COVID-19 pandemic and stay at home orders across the United States, the pandemic increased unemployment rates far above where they had been throughout much of the previous decade. 

# Data Preprocessing, Aggregation and Structuring:
Within the Data folder, there is a subfolder named “Keys_and_Dictionaries”.  This will be used to map unique values into homogenized value names, with the goal of reducing the overall number of factor levels of certain categorical variables. 

The main table, or Fact table, consists of individual trade shipment records. A series of preprocessing steps was required to consolidate the information into one table that could be joined with our other datasets. Datasets were downloaded from a database in batches of 1,000,000 records or less, due to a maximum download constraint established by the Dataymen Descartes system. Import and export records were accessed and downloaded separately from two different system interfaces.In total there were 70 trade datasets, 55 consisting of import records and 15 representing export records. All import files and all export files were concatenated separately in Python. The result was two datasets, one for imports and another for exports. This was completed because attribute names and counts differ between the import and export datasets.

## Key Assumptions:  
-	Dropped companies with less than 100 metric tons over the full time span.

-	Dropped transactions for metric tonnage, TEUs, and TCVUSD that had over 3.5 standard deviations above the mean.  The range was then further reduced from 0 to 2,054 metric tons, to 0 to 250 metric tons, for the final models.  This further reduction accounted for less than 1% of the remaining records.

-	Dropped internal ports and airports.

-	Countries without tariff and price rates account for <14% of the metric tonnage and <13% of the shipments.  We assume these countries play a peripheral role in U.S. trade trends.

-	The U.S. unemployment rate accurately reflects the current labor demand.

-	Feature engineering was done correctly, with appropriate bins for US_company, foreign_company, and carrier.

-	COVID-19 cases and deaths from New York Times reporting so the final number of cases and deaths might be different.

-	Data about state mandated closures were utilized from one source at one point in time, primarily around March and April 2020.

-	The effective date of closure of Charleston was used for the state closure of South Carolina due to the port being in Charleston.

-	Only including 23 HS codes in the entire dataset, which we thought were most relevant.

-	Used average applied tariff across an entire HS 2 chapter code.  If a country failed to report in a given year, it was assumed the applied tariff rates remained constant from the year before.

## Additional Feature Engineering and Binning: 
Factor groupings were established for the following variables:

-	US_Company

-	Foreign_Company

-	US_Port

-	Foreign_Port of a Foreign_Country

-	Foreign_Country

-	Foreign_Company

-	Carriers

-	HS_Class

The same thresholds were used to differentiate companies involved in trade by the total amount of metric tonnage over the entire time period.

All companies that trade less than 100 metric tons are omitted from the dataset. These are not seen as significant players in international trade.  Many of these entries are individuals with shipments at or below 1 metric ton. The model is primarily focused on companies which regularly have shipments that at least meet the minimum thresholds. After removing records associated with the very small individuals and companies, we can create six factor levels for US_Company and Foreign_Company: 

-	micro =  100-1000

-	small = 1001-10000

-	medium = 10001-100000

-	large = 100001-1000000

-	huge = 1000000+

-	unknown = NOT AVAILABLE

Only 23 harmonized service classes of shipments are being considered, which are grouped into the six categories below:

-	Edible

-	Edible with Processing

-	Vices

-	Pharma

-	Raw Input

-	Finished Goods


The tariff and exchange rate datasets needed some data manipulation for the country variables. Both of these datasets had entries that included the EU as a whole without breaking down the records by individual country. This causes an issue when trying to account for changes on a national level. In order to solve this issue, the EU records in the  dataset were duplicated 27 times (for each EU country) and a new attribute was created to indicate if the country belonged to the EU for both the tariff and exchange rate datasets. 

The same limitations for the tariff dataset. like the U.K. projections potentially being off, and including Croatia as an EU country as of 2010, while it actually joined in 2013 still apply.  The tariff table was inner merged with the fact table in such a way that only three attributes remained in the tariff table, which were “Average_Tariff”, “Free_Trade_Agreement_with_the_US”, and the new “European_Union” attribute, while having the same number of records. 

# Variable Selection and Transformation for Modeling:
Outliers with observations >= 3.5 standard deviations from the mean was utilized. Detecting and dropping these outliers will eliminate instances where a company has one abnormally huge shipment in a given year. This should keep the model focused on those company’s typical shipping behavior. Steel and furniture have much higher average weight than the other HS classes included in this dataset and so a higher threshold was selected to avoid eliminating these heavier HS classes. 

## Steps for Variable Selection:

-	Variable relevance, missing observations, distribution testing & correlations

-	Feature importance via testing Random Forest and XGBoost models for the research questions

-	Pandas Profiling 

-	Principal Component Analysis

-	Data exploration using `Tableau` and `PowerBI`

# Modelling


## Machine Learning
Given the introduction of COVID-19 in late 2019/early 2020, the components associated with maritime trade imports/exports, with an emphasis on various sets of yearly sets, were used in modeling. ML models used the following libraries:
- `XGBoost`
- `RAPIDS`: `XGBoost` 
- `Lightgbm` 
- `Scikit-learn`: Linear


### HPO
For hyperparameter tuning, `Hyperopt`, and `GridSearchCV` were utilized to determine which components of the model parameters resulted in the lowest error using various regression metrics to predict metric tonnage. Various trial/experiment sizes were completed to determine which parameters when incorporated into the model resulted in the lowest error.


### Model Explanations
`ELI5`, `SHAP` and `LIME` were incorporated to explain the results after hyperparameter tuning the models.


## Deep Learning
`Tensorflow` was used for evaluating the proposed questions:
- Multilayer perceptron (MLP) regression
- LSTM


### HPO
For hyperparameter tuning, the `num_layers`, `layer_size` and `learning_rate` were tested using `KerasTuner`.


## Time Series
Univariate GARCH models were evaluated using `rugarch` in `R`. The data was formatted into a time series, aggregrated, assessed if stationary using the Augmented Dickey Fuller (ADF) test, and both the ARMA and GARCH orders were created before the forecast models were fit and subsequent predictions.