import pandas as pd 
import numpy as np
from statsmodels.tsa.stattools import adfuller 
from statsmodels.stats.diagnostic import acorr_ljungbox

def get_Stat(df_CDS_data):
    df_CDS_Stats = df_CDS_data.mean(axis=0).to_frame(name='Mean') #.rename(columns={0: 'Mean'})
    df_CDS_Stats['Stdv'] = df_CDS_data.std(axis=0).to_frame(name='Stdv')
    df_CDS_Stats = df_CDS_Stats.rename(columns={'key_0': 'Dates'})

    df_CDS_Stats = pd.merge(
        df_CDS_Stats, pd.DataFrame(
            {'1st order Autocorr' : df_CDS_data.apply(lambda col: col.autocorr(lag=1)),
            "Skew" : df_CDS_data.apply(lambda col: col.skew()),
            'Kurt' : df_CDS_data.apply(lambda col: col.kurtosis()),
            }, 
            index= df_CDS_Stats.index
        ),
        left_index=True,
        right_index=True,
        )
    
    df_CDS_Stats = pd.merge(
        df_CDS_Stats,
        df_CDS_data.quantile([0.05,0.10,0.25,0.5,0.75,0.9,0.95]).transpose(), 
        left_index=True,
        right_index=True)
    
    df_CDS_Stats = df_CDS_Stats.rename(
        columns={0.05 : 'Q.5%', 0.10:'Q.10%',0.25 : 'Q.25',0.5: 'Median',0.75 : 'Q.75%',0.9 : 'Q.90%',0.95 : 'Q.95%'})
    
    return df_CDS_Stats

def get_Stat_Summary(df_CDS_data):
    df_CDS_Stats = get_Stat(df_CDS_data)
    df_CDS_Stats_Summary = pd.DataFrame({'Mean' : df_CDS_Stats.apply(lambda col: col.mean())})
    
    df_CDS_Stats_Summary = pd.merge(
        df_CDS_Stats_Summary,
        df_CDS_Stats.quantile([0.05,0.10,0.25,0.5,0.75,0.9,0.95]).transpose(), 
        left_index=True,
        right_index=True)
    
    df_CDS_Stats_Summary = df_CDS_Stats_Summary.rename(
        columns={0.05 : 'Q.5%', 0.10:'Q.10%',0.25 : 'Q.25',0.5: 'Median',0.75 : 'Q.75%',0.9 : 'Q.90%',0.95 : 'Q.95%'})
    return df_CDS_Stats_Summary

def adfuller_test(series):
    result = adfuller(series)
    return result[1]

def ljung_box_test(series, lags=10):
    result = acorr_ljungbox(series, lags=[lags], return_df=True)
    return result.iloc[0]['lb_pvalue']


def get_graph_data(df_CDS_data, df_LogDiff_CDS):

    df_CDS_data_mean = pd.DataFrame({'Mean' : df_CDS_data.mean(axis=1)})
    df_CDS_data_mean = pd.merge(
        df_CDS_data_mean,
        df_CDS_data.quantile([0.10,0.25,0.75,0.9],axis=1).transpose(), 
        left_index=True,
        right_index=True)
    
    df_CDS_data_mean = df_CDS_data_mean.rename(
            columns={ 0.10:'Q.10%',0.25 : 'Q.25',0.75 : 'Q.75%',0.9 : 'Q.90%'})
    
    df_LogDiff_CDS_mean = df_LogDiff_CDS.mean(axis=1)

    return df_CDS_data_mean, df_LogDiff_CDS_mean
