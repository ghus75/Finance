"""
Module containing utilities for portfolio calculations
"""
import pandas as pd
import numpy as np


### Returns and volatilities ###

def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns.
    r is either a pandas DataFrame or a pandas Series containing a time history of returns.
    periods_per_year = 252 for daily returns, 12 for monthly returns, etc.
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods) - 1
    
def portfolio_return(weights, returns):
    """
    Computes the return on a portfolio from constituent returns and weights.
    weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix.
    """
    return weights.T @ returns
    
def annualize_vol(r, periods_per_year):
    """
    Annualizes the volatility of a set of returns.
    """
    return r.std()*(periods_per_year**0.5)    

def sharpe_ratio(r, annual_riskfree_rate, periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of returns.
    r is given for an arbitrary period, while risk free rate is ususally already annualized
    """
    # convert the annual riskfree rate to per period,
    # amounts to reversing the annualization operation
    rf_per_period = (1+annual_riskfree_rate)**(1/periods_per_year)-1
    # difference r-rf can now be calculated since both r and rf have the same time resolution
    excess_ret = r - rf_per_period
    # Now, re-annualize enverything:
    # annualize (r-rf)
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    # annualize volatility
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol
    
def portfolio_vol(weights, covmat):
    """
    Computes the vol of a portfolio from a covariance matrix and constituent weights.
    weights are a numpy array or N x 1 maxtrix and covmat is an N x N matrix.
    """
    return (weights.T @ covmat @ weights)**0.5
    
def drawdown(return_series: pd.Series):
    """Takes a time series of asset returns.
       returns a DataFrame with columns for the wealth index, the previous peaks, and 
       the percentage drawdown
    """
    wealth_index = (1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({"Wealth": wealth_index, 
                         "Previous Peak": previous_peaks, 
                         "Drawdown": drawdowns})

### Value at Risk ###

def var_historic(r, level=5):
    """
    Returns the historic Value at Risk at a specified level
    i.e. returns the number such that "level" percent of the returns
    fall below that number, and the (100-level) percent are above
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")

def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
     = average of all returns which are worse than the VaR
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")
        
from scipy.stats import norm
def var_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gaussian VaR of a Series or DataFrame
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    """
    # compute the Z score assuming it was Gaussian = # of std deviations from the mean
    z = norm.ppf(level/100) # percent point function for a Normal distrib
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = r.skew()
        k = r.kurtosis() # excess kurtosis
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 -3*z)*(k)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z*r.std(ddof=0))
    
def summary_stats(r, riskfree_rate=0.03, periods_per_year=12):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    ann_r = r.aggregate(annualize_rets, periods_per_year=periods_per_year)
    ann_vol = r.aggregate(annualize_vol, periods_per_year=periods_per_year)
    ann_sr = r.aggregate(sharpe_ratio, annual_riskfree_rate=riskfree_rate, periods_per_year=periods_per_year)
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    skew = r.skew()
    kurt = r.kurtosis()
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    hist_cvar5 = r.aggregate(cvar_historic)
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
    })




