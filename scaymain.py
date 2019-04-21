# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 18:38:42 2019

@author: The Prince
"""


#%% (1) Load relevant datasets (cholera, healthsite, population)
import pandas as pd
from scayfunctions import *

# Load national cholera data
# url = "https://docs.google.com/spreadsheets/d/1P0ob0sfz3xqG8u_dxT98YcVTMwzPSnya_qx6MbX-_Z8/pub?gid=27806487&single=true&output=csv"

# Load (subnational) governorate-level cholera data
url1 = 'https://docs.google.com/spreadsheets/d/1P0ob0sfz3xqG8u_dxT98YcVTMwzPSnya_qx6MbX-_Z8/pub?gid=0&single=true&output=csv'
cholera = load_dataset(url1)
# Remove commas and convert case data to numeric
cholera['Cases'] = cholera['Cases'].str.replace(',', '')
cholera['Cases'] = pd.to_numeric(cholera['Cases'])
# Drop unnecessary columns
cholera.drop(columns=['COD Gov English','COD Gov Arabic','COD Gov Pcode'],inplace=True)

# Correct governorate-level data in cholera dataset 
clean_governorate_dataset(cholera)    

# Load healthsite data
url2 = 'http://data.humdata.org/dataset/8ed6967a-13c0-4b57-9873-563970a1a35f/resource/af952140-3b69-407c-9d59-a1dab2cca008/download/yemen.csv'
healthsite = load_dataset(url2)

# Load population data
url3 = 'http://data.humdata.org/dataset/8ded1878-b737-4922-adf4-05216bd46674/resource/88590116-ea8f-4daf-ae32-3db8a015a319/download/yem_pop_adm1.csv'
population = load_dataset(url3)

# Retain only governorate name and total population
population = population.filter(items=['admin1Name_en','T'])
population.rename(columns={'admin1Name_en':'Governorate','T':'Population'},inplace=True)

#%% (2) Geospatial visualization
import geopandas as gpd
import matplotlib.pyplot as plt

# Load shapefile 
shp = gpd.read_file('shapefiles/governorate.shp')

# Count number of healthsites in each governorate 
count_healthsites(shp,healthsite)

# Aggregate geographic, cholera, population and healthsite data
overall = combine_datasets(cholera,population,shp)

# Plot (separate) geographic maps of Yemen illustrating relevant statistics    
plot_maps(overall)

#%% (3) Principal components analysis (PCA) of Yemen Governorate data
# Done to determine best clusters of governorates on the basis of several features:
# Shape_Len, Shape_Area, Healthsites, Cases, Deaths, CFR (%), Attack Rate, Population 
import numpy as np
import seaborn as sns

# Set grid style using seaborn
sns.set_style("darkgrid")

# Remove geometry column
X = overall.drop(columns=['geometry'])

# Plot pairwise relationships of governorate data
g = sns.pairplot(X)
g.fig.suptitle('Pairwise Relationships of Governorate Data')
fig.subplots_adjust(top=.9,bottom=.1)

import warnings
warnings.filterwarnings("ignore")

# Standardize data prior to PCA
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Xsc = sc.fit_transform(X)

# Standard PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
Xpca = pca.fit_transform(Xsc)
# Output percentage of total variance explained through first two PCs
print(pca.explained_variance_ratio_.sum())

# K-means clustering of standard PCA
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state = 0)
labels = kmeans.fit_predict(Xpca)

# Plot clusters along with map representations
plot_pca(Xpca,overall,labels,'Standard PCA')

# Kernel PCA
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 2, kernel = 'rbf')
Xkpca = kpca.fit_transform(Xsc)

# K-means clustering of Kernel PCA
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=6, random_state = 0)
labels = kmeans.fit_predict(Xkpca)

# Plot clusters along with map representations
plot_pca(Xkpca,overall,labels,'Kernel PCA')

#%% (4) Time series analysis & forecasting (National data)
# Load national cholera dataset
url4 = 'https://docs.google.com/spreadsheets/d/1P0ob0sfz3xqG8u_dxT98YcVTMwzPSnya_qx6MbX-_Z8/pub?gid=27806487&single=true&output=csv'
country = load_dataset(url4)
country = country.drop(country.index[0])
country[country.columns[1:5]] = country[country.columns[1:5]].apply(pd.to_numeric,errors='coerce')

# Resample national data and train ARIMA model
z = extract_weekly_data(country)
pred,_,_ = train_ARIMA_model(z,'Yemen',False,country.iloc[0,0],0.7,4)

# Plot results related to national data
import matplotlib.dates as mdates
fig, ax = plt.subplots(figsize=(16,16))
ax.plot(z,label='Actual')
ax.plot(pred,label='Predictions')
ax.axvline(pd.to_datetime(country.iloc[0,0]),color='k',ls=':')
ax.set_title('Actual Cases vs ARIMA Predictions (National)',fontsize=20)
ax.set_xlabel('Date')
ax.set_ylabel('Cases')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_minor_locator(mdates.MonthLocator((3,6,9,12)))
ax.xaxis.set_major_formatter(mdates.DateFormatter("\n%Y"))
ax.xaxis.set_minor_formatter(mdates.DateFormatter("%b"))
ax.legend()
fig.subplots_adjust(top=.9,bottom=.1)

#%% (5) Time series analysis & forecasting (Subnational data)
import warnings
warnings.filterwarnings("ignore")
# Extract names of all governorates
regions = cholera['Governorate'].cat.categories.tolist()
results = pd.DataFrame(columns=['Region','Values','Cases']) 
# Iterate through all governorates
for name in regions:
    tmp1 = pd.DataFrame(columns=['Region','Values','Cases']) 
    tmp2 = pd.DataFrame(columns=['Region','Values','Cases'])       
    # Load specific subnational data
    region = cholera[cholera['Governorate'] == name]
    # Extract (average) weekly subnational data
    y = extract_weekly_data(region,name,False)
    # Train ARIMA model
    pred,_,_ = train_ARIMA_model(y,name,False,region.iloc[0,0],0.7)
    # Create dummy dataframes to store data
    tmp1['Cases'] = y
    tmp1['Region'] = name
    tmp1['Values'] = 'Actual'
    tmp2['Cases'] = pred
    tmp2['Region'] = name
    tmp2['Values'] = 'Predicted'
    # Concatenate to overall results dataframe
    results = pd.concat([results, tmp1, tmp2])
results['Date'] = results.index
   
#train_perc = [0.7, 0.7, 0.5, 0.7, 0.8, 0.8, 0.7, 0.7, 0.7, 0.7, 0.7, 0.5, 0.8,
#              0.4, 0.4, 0.7, 0.4, 0.7, 0.7, 0.7, 0.5, 0.7]    

# Plot results related to subnational data
import seaborn as sns
import matplotlib.dates as mdates

def plot_timestamp(data,**kwargs):
    plt.axvline(pd.to_datetime('2018-02-18'),color='k',ls=':')

g = sns.FacetGrid(results,col='Region',col_wrap=5,hue='Values',sharey=False)
g = g.map(plot_timestamp,'Date')
for i in range(len(g.axes)):
    g.axes[i].xaxis.set_major_locator(mdates.YearLocator())
    g.axes[i].xaxis.set_minor_locator(mdates.MonthLocator((3,6,9,12)))
    g.axes[i].xaxis.set_major_formatter(mdates.DateFormatter("\n%Y"))
    g.axes[i].xaxis.set_minor_formatter(mdates.DateFormatter("%b"))
g = g.map(sns.lineplot,'Date','Cases')
g.add_legend()
g.fig.suptitle('Actual Cases vs ARIMA Predictions (Subnational)',fontsize=20)
g.fig.subplots_adjust(top=.9,bottom=.1)

