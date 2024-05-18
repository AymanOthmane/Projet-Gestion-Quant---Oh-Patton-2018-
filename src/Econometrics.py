"""
This module is responsible for implementing all the econometric models specified in the article 
for the statistical study of data on cds spreads. 

"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class AR_Model:
    def __init__(self, df_main, lags, df_market=None, maxlags = 13):
        """
        Initialize the ARXModel class with dataframes and lag order for the main variable.
        The market variable is always included with exactly one lag.
        
        Parameters
        ----------
        df_main: A pandas DataFrame containing the main time series data.
        lags: An integer indicating the number of lags for the main variable in the AR model.
        df_market: A pandas DataFrame containing the market variable data. (Optionnal)
        maxlags: Maximum lags for model selector
        """
        self.df_main = df_main
        self.df_market = df_market
        self.lags = lags
        self.maxlags = maxlags
        self.model = None
        self.results_Coef = None
        self.residuals = []
        self.results_pvalues = None
        
    def fit_models(self):
        """
        Fit an AutoReg model to each column in the DataFrame with dynamic lag specification.
        """
        resultsCoef_list = []
        resultspvalues_list = []

        for col in self.df_main:
            self.model = AutoReg(
                endog = self.df_main[col], 
                lags= 5,
                exog=self.df_market, 
                )
            res = self.model.fit()
            Coef_dict = {
                'Ticker': col,
                **{f'Phi_lag{i}': coef for i, coef in enumerate(res.params, start=1)},
                }
            pval_dict = {
                'Ticker': col,
                **{f'pvalue_lag{i}': pval for i, pval in enumerate(res.pvalues, start=1)},
                }
        
            
            self.residuals.append(res.resid)
            resultsCoef_list.append(Coef_dict)
            resultspvalues_list.append(pval_dict)

        self.results_Coef = pd.DataFrame(resultsCoef_list)
        self.results_pvalues = pd.DataFrame(resultspvalues_list)
        self.results_Coef.set_index('Ticker', inplace=True)
        self.results_pvalues.set_index('Ticker', inplace=True)

        # self.residuals =pd.sDataFrame(self.residuals, index=self.results_Coef.index).transpose()



            