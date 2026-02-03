#Momentum/Mean-reversion dual strategy backtesting

import yfinance as yf
import pandas as pd
import numpy as np

#Data downloading
#Define the list of tickers to be analyzed
ticker=["SPY","QQQ","GLD","USO","TLT","AAPL","TSLA"]

#Download historical data from Yahoo Finance
data=yf.download(ticker, start='2015-10-06',end='2025-10-06', auto_adjust=True)


#Data cleaning
#Reindex to business days to handle missing trading days
expected_dates = pd.date_range(start='2015-10-06', end='2025-10-06', freq='B')
#Forward-fill to ensure a continuous time series on business days
data = data.reindex(expected_dates, method='ffill')

#Get the adjusted closing prices for strategy calculations
close=data['Close'].copy()

#I- Momentum Strategy Implementation (SMA Crossover)

#Get the 50 days and 200 days Simple Moving Averages (SMA)
sma_50d=close.rolling(window=50).mean()
sma_200d=close.rolling(window=200).mean()

#Create a momentum time series (ts_momentum)
#Position is Long (+1) when SMA(50) > SMA(200) and Short (-1) otherwise
ts_momentum=np.sign(sma_50d-sma_200d)


#II- Mean Reversion Strategy Implementation (RSI)

#Calculate daily returns
daily_returns=close.pct_change()

#Separate daily gains and losses
gains=daily_returns.mask(daily_returns<0,0)
losses=daily_returns.mask(daily_returns>0,0).abs()

#Calculate Exponential Weighted Moving Average (EWMA) for gains and losses
period=14 # Standard RSI period
#Use the com (center of mass) parameter for EWMA calculation: com = span - 1
avg_gain=gains.ewm(com=period-1, adjust=False).mean()
avg_loss=losses.ewm(com=period-1, adjust=False).mean()

#Calculate Relative Strength (RS)
rs=avg_gain/avg_loss

#Handle division by zero which results in infinity, replacing them with 0
rs.replace([np.inf, -np.inf], 0, inplace=True)

#Calculate Relative Strength Index (RSI)
rsi=100-(100/(1+rs))

#Create a mean-reversion time series (ts_meanreversion)
oversold_level=30
overbought_level=70

#Position is Long (+1) when RSI < 30 (Oversold), Short (-1) when RSI > 70 (Overbought), else Flat (0)
ts_meanreversion=np.where(rsi>overbought_level,-1,np.where(rsi<oversold_level,1,0))

#Convert the numpy array back to a DataFrame with correct index and column names
ts_meanreversion = pd.DataFrame(ts_meanreversion, index=rsi.index, columns=rsi.columns)
#Ensure column names match the Close data for alignment
ts_meanreversion.columns = [f'{col}' for col in ts_meanreversion.columns]


#III- Dual Strategy Logic

#The Dual Strategy takes a position only when BOTH strategies agree.
#Long (+1) if (Momentum == 1) AND (Mean Reversion == 1)
#Short (-1) if (Momentum == -1) AND (Mean Reversion == -1)
#Flat (0) otherwise

#Filling initial NaNs with 0 (no position)
ts_momentum = ts_momentum.fillna(0)
ts_meanreversion = ts_meanreversion.fillna(0)

#Create the Boolean conditions for Long and Short
condition_long = (ts_momentum == 1) & (ts_meanreversion == 1)
condition_short = (ts_momentum == -1) & (ts_meanreversion == -1)

#Apply the conditions to create the Dual Strategy time series
ts_dual = pd.DataFrame(0, index=ts_momentum.index, columns=ts_momentum.columns)
ts_dual[condition_long] = 1
ts_dual[condition_short] = -1

#Now ts_dual represents the position: 1 (Long), -1 (Short), 0 (Flat)


#Backtesting Setup

TC=0.0005 #Transction cost (0.05%)

#Shifting the position by 1 day to avoid lookahead bias
#Position decided today (based on yesterday's close/indicators) is applied to tomorrow's open trade
ts_momentum=ts_momentum.shift(1).fillna(0)
ts_meanreversion=ts_meanreversion.shift(1).fillna(0)
ts_dual=ts_dual.shift(1).fillna(0) # Dual strategy also shifted

#Daily returns calculation (using Open to Close as the daily trade return)
open_prices=data['Open'].copy()
bt_returns=(close/open_prices)-1

#Reindex returns to match the strategy time series index
momentum_returns=bt_returns.reindex(ts_momentum.index)
meanreversion_returns=bt_returns.reindex(ts_meanreversion.index)
dual_returns=bt_returns.reindex(ts_dual.index) # Dual strategy returns

#IV- Gross Returns Calculation

#Gross returns: strategy position * daily trade return
momentum_gross_returns=momentum_returns*ts_momentum
meanreversion_gross_returns=meanreversion_returns*ts_meanreversion
dual_gross_returns=dual_returns*ts_dual # Dual strategy gross returns

#V- Net Returns Calculation (Adjusted for Transaction Costs)

#Transaction cost calculation: TC is applied whenever the position *changes* (diff != 0)
momentum_trades=ts_momentum.diff().abs()
momentum_TC=TC*momentum_trades

meanreversion_trades=ts_meanreversion.diff().abs()
meanreversion_TC=TC*meanreversion_trades

dual_trades=ts_dual.diff().abs() # Dual strategy trades
dual_TC=TC*dual_trades


#Net returns: Gross returns - Transaction costs
momentum_net_returns=momentum_gross_returns-momentum_TC
meanreversion_net_returns=meanreversion_gross_returns-meanreversion_TC
dual_net_returns=dual_gross_returns-dual_TC # Dual strategy net returns

#VI- Cumulative Returns

momentum_cum_returns=(1+momentum_net_returns).cumprod()
meanreversion_cum_returns=(1+meanreversion_net_returns).cumprod()
dual_cum_returns=(1+dual_net_returns).cumprod() # Dual strategy cumulative returns

#Dropping the initial synthetic row used for starting the cumulative product at 1
momentum_cum_returns = momentum_cum_returns.iloc[1:]
meanreversion_cum_returns = meanreversion_cum_returns.iloc[1:]
dual_cum_returns = dual_cum_returns.iloc[1:]

#VII- Key Performance Metrics Calculation

def calculate_kpis(cum_returns, trading_days=252):
    #Daily net returns (to calculate vol and drawdown)
    #Using cum_returns / cum_returns.shift(1) - 1 to get daily returns before .fillna(0)
    net_returns = cum_returns.pct_change().fillna(0)
    
    #Annualized Return
    total_returns = cum_returns.iloc[-1] - 1
    #Calculate the number of years in the backtesting period
    years = (cum_returns.index[-1] - cum_returns.index[0]).days / 365.25
    #Compounding formula for annualized return
    annualized_return = (1 + total_returns)**(1/years) - 1
    
    #Annualized Volatility
    annualized_volatility = net_returns.std() * np.sqrt(trading_days)
    
    #Sharpe Ratio (Assuming a risk-free rate of 0 for simplicity)
    risk_free_rate = 0.0
    #Sharpe Ratio formula: (Annualized Return - Risk-Free Rate) / Annualized Volatility
    #Handle division by zero for assets with zero volatility
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
    sharpe_ratio.replace([np.inf, -np.inf], 0, inplace=True) # Handle division by zero/inf
    

    
    #Maximum Drawdown (MDD)
    # Calculate the running maximum (peak)
    peak = cum_returns.cummax()
    #Calculate the daily drawdown: (Current Cumulative Return / Peak) - 1
    drawdown = (cum_returns / peak) - 1
    #Max Drawdown is the minimum (most negative) of the daily drawdown
    max_drawdown = drawdown.min()


    #Compile the results into a DataFrame
    kpis = pd.DataFrame({
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }).T
    
    return kpis

#Calculate the KPIs for each strategy
momentum_kpis_all = calculate_kpis(momentum_cum_returns)
meanreversion_kpis_all = calculate_kpis(meanreversion_cum_returns)
dual_kpis_all = calculate_kpis(dual_cum_returns) # Dual strategy KPIs


print("\n" + "="*85)
print("Momentum/Mean-reversion Dual Strategy Performance Comparison by Ticker")
print("="*85)

#Créez un DataFrame unique de comparaison pour un affichage propre sans troncation
comparison_list = []
for ticker_name in ticker:
    #Récupérer les métriques pour le ticker actuel
    mom_data = momentum_kpis_all[ticker_name]
    mr_data = meanreversion_kpis_all[ticker_name]
    dual_data = dual_kpis_all[ticker_name]
    
    #Créer le dictionnaire de résultats formatés pour le ticker
    formatted_results = {
        'Metric': ['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio', 'Max Drawdown'],
        f'{ticker_name} - Momentum': [
            f"{mom_data['Annualized Return']:.2%}",
            f"{mom_data['Annualized Volatility']:.2%}",
            f"{mom_data['Sharpe Ratio']:.3f}",
            f"{mom_data['Max Drawdown']:.2%}"
        ],
        f'{ticker_name} - Mean-Reversion': [
            f"{mr_data['Annualized Return']:.2%}",
            f"{mr_data['Annualized Volatility']:.2%}",
            f"{mr_data['Sharpe Ratio']:.3f}",
            f"{mr_data['Max Drawdown']:.2%}"
        ],
        f'{ticker_name} - Dual (Agreement)': [
            f"{dual_data['Annualized Return']:.2%}",
            f"{dual_data['Annualized Volatility']:.2%}",
            f"{dual_data['Sharpe Ratio']:.3f}",
            f"{dual_data['Max Drawdown']:.2%}"
        ]
    }
    #Créer un DataFrame pour le ticker et l'ajouter à la liste
    comparison_df_ticker = pd.DataFrame(formatted_results).set_index('Metric')
    comparison_list.append(comparison_df_ticker)



for df in comparison_list:
    ticker_name = df.columns[0].split(' - ')[0] #Extract ticker name from column name
    print(f"\n### {ticker_name} Performance Metrics")
    print("-" * 75)
    #Utilisez to_string pour garantir l'affichage complet sans ellipses
    print(df.to_string())
    print("-" * 75)