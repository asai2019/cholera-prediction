# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 18:10:30 2019

@author: The Prince
"""


def load_dataset(url):
    '''Loads a dataset by connecting to url specifying dataset location
    
    Args:
        url (str): url
        
    Returns:
        df (DataFrame): dataframe specified by url
    '''
    import pandas as pd
    import io, requests
    s = requests.get(url).content
    df = pd.read_csv(io.StringIO(s.decode('utf-8')))
    return df

def clean_governorate_dataset(df):
    '''Removes misspellings, punctuation errors in governorate names
    
    Args:
        df (DataFrame): dataframe containing governorate-level cholera data
      
    '''
    df['Governorate'] = df['Governorate'].str.replace('AL Mahrah','Al Maharah')
    df['Governorate'] = df['Governorate'].str.replace('Ma\'areb','Marib')
    df['Governorate'] = df['Governorate'].str.replace('-',' ')
    df['Governorate'] = df['Governorate'].str.replace('_',' ')
    # There should only be 22 governorates
    df['Governorate'] = df['Governorate'].astype('category')
     
def count_healthsites(shp,healthsite):
    '''Finds number of healthsites in each governorate and store as column 
    
    Args:
        shp (DataFrame): dataset containing geographic specifications
        healthsite (DataFrame): dataset containing healthsite locations
    
    '''
    from shapely.geometry import Point
    # Reuse existing column to store healthsite data
    shp.rename(columns={'ADM1_EN':'Governorate','validTo':'Healthsites'},inplace=True)
    for i in range(shp.shape[0]):
        mask = [shp.iloc[i,-1].intersects(Point(lat,lon)) for lat,lon in zip(healthsite.X,healthsite.Y)]
        shp.iloc[i,-2] = int(sum(mask))
    # Retain relevant parameters
    shp = shp.filter(items=['Shape_Leng','Shape_Area','Governorate','Healthsites','geometry'])    
        
def combine_datasets(cholera,population,shp):
    '''Combines datasets containing various subnational-level statistics
    
    Args:
        cholera (DataFrame): cholera dataset
        population (DataFrame): Yemen population dataset
        shp (DataFrame): dataset containing geographic specifications and healthsite information
        
    Returns:
        overall (Dataframe): combined governorate-level dataset
        
    '''
    # Retain most recent (or max) values from cholera dataset grouped by 
    # governorate
    subnational = cholera.groupby(['Governorate']).max().drop(columns='Date').reset_index()
    # Hadramaut = Say'on + Moklla
    subnational.loc[subnational.shape[0]] = subnational.iloc[15,:] + subnational.iloc[19,:]
    subnational.iloc[subnational.shape[0]-1,0] = 'Hadramaut'
    subnational = subnational.drop(index=[15,19])
    # Join dataframes together using governorate as a key
    overall = shp.set_index('Governorate').join(population.set_index('Governorate'), how = 'inner').join(subnational.set_index('Governorate'), how = 'inner')
    # Retain only necessary columns
    overall = overall.filter(items=['Shape_Leng','Shape_Area','geometry','Healthsites',
                             'Population','Cases','Deaths','CFR (%)','Attack Rate (per 1000)'])
    return overall

def plot_maps(df):
    '''Plot individual features/columns pertaining to geodataframe
    
    Args:
        df (DataFrame): combined governorate-level dataset
    
    '''
    import matplotlib.pyplot as plt
    # Initialize a 3x2 grid of subplots
    fig, axes = plt.subplots(3,2,figsize=(20,16))
    fig.suptitle('Map of Yemen by Features',fontsize=20)
    # Plot population, healthsites, cases, deaths, case fatality rate, 
    # and attack rate on Yemen map
    df.plot(ax=axes[0,0],column='Population',cmap='Reds')
    axes[0,0].set_title('Population')
    axes[0,0].axis('off')
    df.plot(ax=axes[0,1],column='Healthsites',cmap='Blues')
    axes[0,1].set_title('Number of Healthsites')
    axes[0,1].axis('off')
    df.plot(ax=axes[1,0],column='Cases',cmap='Oranges')
    axes[1,0].set_title('Number of Cholera Cases')
    axes[1,0].axis('off')
    df.plot(ax=axes[1,1],column='Deaths',cmap='Purples')
    axes[1,1].set_title('Number of Cholera Deaths')
    axes[1,1].axis('off')
    df.plot(ax=axes[2,0],column='CFR (%)',cmap='Greens')
    axes[2,0].set_title('Cholera Case Fatality Rate')
    axes[2,0].axis('off')
    df.plot(ax=axes[2,1],column='Attack Rate (per 1000)',cmap='Greys')
    axes[2,1].set_title('Cholera Attack Rate')
    axes[2,1].axis('off')
#     Plot map of Yemen with healthsite data
#    fig, ax = plt.subplots()
#    # Convert to lat/long and add healthsite location data
#    overall.to_crs(epsg=4326).plot(ax=ax,alpha=0.4).axis('equal')
#    healthsite.plot(x='X',y='Y',kind='scatter',s=2,ax=ax)
#    fig.suptitle('Map of Yemen with Healthsite Locations')
      
def plot_pca(Xdr,X,labels,title):
    '''Plots clusters obtained fromn PCA along with associated visualization on Yemen map  
    
    Args:
        Xdr (ndarray): reduced-dimensionality dataset
        X (DataFrame): combined governorate-level dataset
        labels (array): cluster labels for governorates
        title (str): title for plot
    
    '''
    import numpy as np
    import matplotlib.pyplot as plt    
    fig, axes = plt.subplots(1,2,figsize=(20,8))
    n_clusters = labels.max() + 1
    unique_labels = set(labels)
    colors = [plt.cm.Accent(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        class_member_mask = (labels == k)
        xy = Xdr[class_member_mask]
        axes[0].plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)    
    axes[0].set_title('Estimated number of clusters: %d' % n_clusters)
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].axis('equal')
    #plt.show()  
    #fig, ax = plt.subplots(figsize=(16, 12))
    #ax.plot(Xkpca[:, 0], Xkpca[:, 1], 'o', color='C1')
    for i, country in enumerate(X.index):
        axes[0].annotate(country, (Xdr[i, 0] + 0.01, Xdr[i, 1] + 0.01), fontsize=14, alpha=0.7)
    X['Cluster'] = labels
    X.plot(ax=axes[1],column='Cluster',cmap='Accent')
    axes[1].axis('off')
    fig.suptitle(title,fontsize=20)
    
def extract_weekly_data(df,name='',plotCases=False):
    '''Computes average weekly case data extracted from entire case data
    
    Args: 
        df (DataFrame): cholera dataset
        name (str): name of governorate
        plotCases (boolean): whether to plot cases
        
    Returns:
        y (DataFrame): cholera dataset with resampled weekly case data
        
    '''
    import pandas as pd
    import matplotlib.pyplot as plt
    weekly = df[:]
    weekly['Cases'] = weekly['Cases'].diff(-1)
    weekly['Date'] = pd.to_datetime(weekly.Date, format = '%Y-%m-%d')
    weekly.set_index('Date',inplace=True)
    # Resample case data to represent weekly average cases
    y = weekly['Cases'].resample('W').mean()
    # Replace nan values with averages
    average_resamples(y)
    # Plot original and averaged case data
    if plotCases == True:
        plt.figure(figsize=(12, 6))
        plt.plot(weekly.Cases)
        plt.plot(y)
        plt.title(name)
        plt.grid(True)
        plt.show()
    return y

# Imputes resamples by distributing averages to NaNs
# Since this resampling may contain NaNs, we will have to fill in NaNs 
# appropriately, for proper time series analysis and forecasting
# However, we cannot simply fill in NaNs with any value, since it is based on 
# actual counts that must match reported cases
# Therefore, we resort to "spreading out"/distributing cases evenly between 
# successive dates for which data is not available and when data is eventually 
# updated
def average_resamples(df):
    '''Imputes missing samples by replacing NaNs with average values
    
    Args:
        df (DataFrame): cholera dataset for individual governorate 
    
    '''
    import numpy as np
    idx = np.array(np.where(df.isna()))
    nans = list()
    for i in range(idx.min(),idx.max()+2):
        nans.append(i)
        if i not in idx:
            val = df[i]/len(nans)
            df[nans] = val
            nans = []    

# This function tests for stationarity by applying the augmented Dickey-Fuller
# test.  
def test_stationarity(ts,name,confidence=0.05):
    '''Tests for stationarity of time series by applying augmented Dickey-Fuller test
    
    Args:
        ts (DataFrame): cholera dataset for individual governorate
        name (str): name of governorate
        confidence (float): confidence level for significance
        
    Returns:   
        isStationary (boolean): indicator of stationarity
        adftest[1] (float): p-value
        
    '''
    from statsmodels.tsa.stattools import adfuller
    adftest = adfuller(ts, autolag = 'AIC')
    # Test p-value against critical value
    # If the p-value is less than the critical level for statistical 
    # significance, the time series is not non-stationary.
    if adftest[1] < confidence:
        print('Time series {} is stationary.'.format(name))
        isStationary = 1
    else:
        print('Time series {} is not stationary.'.format(name))
        isStationary = 0
    return isStationary, adftest[1]


def train_ARIMA_model(y, name, plotCases = True, end_frame = '2018-02-18', 
                      train_start = 0.6, forecast_window = 4):
    '''Trains ARIMA model, and then predicts future observations with best-performing model
    
    Args:
        y (DataFrame): cholera dataset for individual governorate
        name (str): name of governorate
        plotCases (boolean): indicator to plot predicted and actual case data
        end_frame (str): end date of available data
        train_start (float): initial percentage of training set to train on
        forecast_window (int): number of weeks to forecast ahead
        
    Returns:
        pred (DataFrame): predicted observations from ARIMA model
        order (tuple): best performing ARIMA parameters
        rmse (float): root mean squared error between predictions and data
    
    '''
    import datetime
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from statsmodels.tsa.arima_model import ARIMA
    from sklearn.metrics import mean_squared_error
    # Obtain best (p,d,q) tuple via grid search 
    order = time_series_grid_search(y, train_start, forecast_window)
    # Fit model with resulting best parameters 
    model = ARIMA(y, order).fit()
    # Obtain in- and out-of-sample model predictions
    # For differenced models
    if order[1] > 0:
        # Specify end date
        end_date = pd.to_datetime(end_frame) + datetime.timedelta(weeks=forecast_window+1)
        pred = np.maximum(model.predict(end=end_date,dynamic=False,typ='levels'),0)
    else:
        end_date = pd.to_datetime(end_frame) + datetime.timedelta(weeks=forecast_window+1)
        pred = np.maximum(model.predict(end=end_date,dynamic=False),0)   
    if not any(order):
        end_date = pd.to_datetime(end_frame) + datetime.timedelta(weeks=forecast_window)
        pred = np.maximum(model.predict(end=end_date,dynamic=False),0) 
    # Account for differencing by shifting back one period
    if order[1] > 0:
        pred = pred.shift(periods=-1, freq='W')
    else:
        # Account for autoregression or moving average by shifting index
        if order[0] > 0 or order[2] > 0:
            pred = pred.shift(periods=-1)
            pred = pred[:-1]
    # Report RMSE between true and predicted values
    rmse = np.sqrt(mean_squared_error(y,pred[0:len(y)]))
    print('The RMSE of our forecasts is {}'.format(round(rmse, 2)))
    # Plots case data (if requested)
    if plotCases == True:
        plt.figure()
        ax = y.plot(label='Observed')
        pred.plot(ax=ax, label='In-sample Predictions')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cases')
        plt.title(name)
        plt.legend()
        plt.show()   
    pred.columns = ['Cases']    
    return pred, order, rmse



def time_series_grid_search(y, train_start = 0.7, forecast_window = 4):
    '''Performs grid search over admissible ARIMA parameters
    
    Args:
        y (DataFrame): cholera dataset for individual governorate
        train_start (float): initial percentage of training set to train on
        forecast_window (int): number of weeks to forecast ahead
        
    Returns:
        best_order (tuple): best performing ARIMA parameters
    
    '''
    import itertools
    import numpy as np
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    best_error, best_order = np.inf, None
    for order in pdq:
        try:
            rmse = time_series_CV(y,order,train_start,forecast_window)
            #print('ARIMA%s RMSE=%.3f' % (order, rmse))
            if rmse < best_error:
                best_error, best_order = rmse, order 
        except:
            continue
    #print('Best ARIMA%s RMSE=%.3f' % (best_order, best_error))
    return best_order


def time_series_CV(X, arima_order, train_start=0.7, step_size=1):
    '''Performs time series cross validation using ARIMA modeling
    
    Args:
        X (DataFrame): cholera dataset for individual governorate
        arima_order (tuple): ARIMA parameters
        initial_frac (float): initial percentage of training set to train on
        step_size (int): number of weeks to forecast ahead
        
    Returns:
        error (float): root mean squared error between predictions and data
    
    '''
    import numpy as np
    from statsmodels.tsa.arima_model import ARIMA
    from sklearn.metrics import mean_squared_error
	# Prepare training dataset
    train_size = int(len(X) * train_start)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
	# Make predictions
    predictions = list()
    # Iterate over each step of test set
    # Allows for new ARIMA model to be trained over each time step, 
    # improving model accuracy
    for t in range(len(test)-step_size+1):
        model = ARIMA(history, order=arima_order).fit()
        yhat = model.forecast(steps=step_size)[0]
        predictions.append(yhat[step_size-1])
        history.append(test[t])
	# Calculate out-of-sample test error (RMSE)
    error = np.sqrt(mean_squared_error(test[step_size-1:], predictions))
    return error