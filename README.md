# Quantitative Portfolio Construction with Diverse Trading Strategies

The data consists of 50 selected stocks from the S&P 500 from 2000-01 to 2021-12. A market-cap-weighted portfolio of these stocks is set as the benchmark portfolio. The in-sample period ranges from 2000-01 to 2015-12. The in-sample period is used to create a quantitative strategy portfolio (QS) which aims to improve upon the benchmark portfolio. The QS is then tested on the out-of-sample period from 2016-01 to 2021-12. 

The performance of the QS is measured using Sharpe Ratio, Variance, and Excess returns with respect to the benchmark.

Different constraints will be imposed on the QS.
- Factor exposure (beta values) deviation from the benchmark is set to a maximum of 0.05. 
- Weight deviation for each stock can be a maximum of 0.10 of the benchmark weight. For example, if the weight of a stock is 1%, the weight of the stock in the QS must be between 0.9% and 1.1%.
- Drawdown relative to the benchmark must be a maximum of 1% per month. Thus, the QS should not lose more than 1% of its value per month compared to the benchmark.

Data not included in this repository.