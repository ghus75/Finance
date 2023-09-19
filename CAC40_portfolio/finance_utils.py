"""
Module containing utilities for portfolio calculations
"""
import pandas as pd
import numpy as np
from scipy.optimize import minimize

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
    r is given for an arbitrary period, while risk free rate is usually already annualized
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
    

def summary_stats(r, riskfree_rate, periods_per_year):
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
        "Excess Kurtosis": kurt,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
    })

### Markowitz portfolio ###

def minimize_vol(target_return, er, cov):
    """
    Returns the optimal weights that achieve the target return
    given a set of expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # each of the n weights is in [0.0; 1.0]
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    return_is_target = {'type': 'eq',
                        'args': (er,),
                        'fun': lambda weights, er: target_return - portfolio_return(weights,er)
    }
    weights = minimize(portfolio_vol, init_guess,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,return_is_target),
                       bounds=bounds)
    return weights.x

# Efficient frontier
def optimal_weights(n_points, er, cov):
    """
    Find optimal weights for a given set of returns and a covariance matrix
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights

def plot_ef(n_points, er, cov):
    """
    Plots the multi-asset efficient frontier
    """
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })
    return ef.plot.line(x="Volatility", y="Returns", style='.-')

# Max Sharpe ratio portfolio
def msr(riskfree_rate, er, cov):
    """
    Returns the weights of the portfolio that gives you the maximum sharpe ratio
    given the riskfree rate and expected returns and a covariance matrix
    
    er = annualized returns for each asset of the portfolio
    cov = covariance matrix between the returns of each asset
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # each of the n weights is in [0.0; 1.0]
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio
        of the given portfolio
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol
    
    # minimize the negative Sharpe ratio
    weights = minimize(neg_sharpe, init_guess,
                       args=(riskfree_rate, er, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x

# GMV portfolio
def gmv(cov):
    """
    Returns the weights of the Global Minimum Volatility portfolio
    given a covariance matrix
    """
    n = cov.shape[0]
    # parameters of msr:
    # riskfree_rate = 0 bacause it's not going to get used anyway
    # weights = np.repeat(1,n) : put equal weights everywhere so that optimization is done only on the cov matrix
    return msr(0, np.repeat(1, n), cov)
    
### Backtesting ###

def backtest_ws(r, estimation_window, weighting, **kwargs):
    """
    Backtests a given weighting scheme, given some parameters:
    
    r : asset returns to use to build the portfolio
    
    estimation_window: the window to use to estimate parameters.
    It is a sliding window of length `estimation_window` (in whatever unit of time is used for the returns).
    The window moves ahead in time for each time unit of the returns(days, months...), and for
    every new position of the windows, portfolio is rebalanced according to the recalculated weights
    
    weighting: the weighting scheme to use, must be a function that takes "r", 
    and a variable number of keyword-value arguments (**kwargs)
    """
    n_periods = r.shape[0]
    
    # windows is a list of tuples which gives us the (integer) location of the start and stop (non inclusive)
    # for estimation_window=60 we have : windows = [(0, 60), (1, 61), ...]
    windows = [(start, start+estimation_window) for start in range(n_periods-estimation_window+1)]
    
    # DataFrame containing weights for each asset for all dates at the end of the sliding window
    # Its size is (n_periods - estimation_window + 1) x (number of assets)
    weights = [weighting(r.iloc[win[0]:win[1]], **kwargs) for win in windows]
    
    # convert to a DataFrame starting at the end of the first estimation window,
    weights = pd.DataFrame(weights, index=r.iloc[estimation_window-1:].index, columns=r.columns)
    
    # returns = weighted sum of returns accross all assets
    # Note that weights are only available starting at t=`estimation_window`, since 
    # all returns of initial estimation_window (for t = 0 ... estimation_window)
    # were used to calculate the weights at t=estimation_window.
    # So the first `estimation_window` points of `returns` will be NaNs.
    returns = (weights * r).sum(axis="columns",  min_count=1) #mincount is to generate NAs if all inputs are NAs
    
    return returns

# Weighting schemes
def weight_ew(r):
    """
    Returns the weights of the EW portfolio based on the asset returns "r" as a DataFrame
    """
    n = len(r.columns)
    return pd.Series(1/n, index=r.columns)

# GMV portfolio using sample covariance
def sample_cov(r, **kwargs):
    """
    Returns the sample covariance of the supplied returns
    Uses the average of the returns... so it should be fed with a lot of data
    """
    return r.cov()

def weight_gmv(r, cov_estimator, **kwargs):
    """
    Weighting function used in backtest_ws().
    Produces the weights of the GMV portfolio given a covariance matrix of the returns 
    cov_estimator = matrix that is used to estimate the covariance
    """
    est_cov = cov_estimator(r, **kwargs)
    return gmv(est_cov)

# GMV portfolio using Constant Correlation model
def cc_cov(r, **kwargs):
    """
    Estimates a covariance matrix by using the Elton/Gruber Constant Correlation model
    """
    rhos = r.corr()
    n = rhos.shape[0]
    # this is a symmetric matrix with diagonals all 1 - so the mean correlation is ...
    rho_bar = (rhos.values.sum()-n)/(n*(n-1))
    ccor = np.full_like(rhos, rho_bar)
    np.fill_diagonal(ccor, 1.)
    sd = r.std()
    ccov = ccor * np.outer(sd, sd)
    return pd.DataFrame(ccov, index=r.columns, columns=r.columns)    
