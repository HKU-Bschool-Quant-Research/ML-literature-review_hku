# ML-literature-review_hku
## Prerequisite

- Read the listed papers
- [WRDS](https://wrds-web.wharton.upenn.edu) account with subscription to CRSP, Compustat and IBES.
- Python

## Files

- [Characteristics list](https://github.com/Feng-CityUHK/EquityCharacteristics/blob/master/chars60_summary.csv)

### Main Files
- accounting_60_hxz.py  -- most annual, quarterly and monthly frequency characteristics
- functions.py -- impute and rank functions
- merge_chars.py -- merge all the characteristics from different pickle file into one pickle file
- impute_rank_output_bchmk.py -- impute the missing values and standardize raw data
- iclink.py -- preparation for IBES
- pkl_to_csv.py -- converge the pickle file to csv

### Single Characteristic Files
- beta.py -- 3 months rolling CAPM beta
- rvar_capm.py, rvar_ff3.py -- residual variance of CAPM and fama french 3 factors model, rolling window is 3 months
- rvar_mean.py -- variance of return, rolling window is 3 months
- abr.py -- cumulative abnormal returns around earnings announcement dates
- myre.py -- revisions in analysts’ earnings forecasts
- sue.py -- unexpected quarterly earnings
- ill.py -- illiquidity, rolling window is 3 months
- maxret_d.py -- maximum daily returns, rolling window is 3 months
- std_dolvol.py -- std of dollar trading volume, rolling window is 3 months
- std_turn.py -- std of share turnover, rolling window is 3 months
- bid_ask_spread.py -- bid-ask spread, rolling window is 3 months
- zerotrade.py -- number of zero-trading days, rolling window is 3 months

## How to use

1. run accounting_60_hxz.py
2. run all the single characteristic files (you can run them in parallel)
3. run merge_chars.py
4. run impute_rank_output_bckmk.py (you may want to comment the part of sp1500 in this file if you just need the all stocks version)

## Outputs

### Data

The date range is 1972 to 2019. The stock universe is top 3 exchanges (NYSE/AMEX/NASDAQ) in US.

The currant time of data is $ret_t = chars_{t-1}$

1. chars_raw_no_impute.feather (all data with original missing value)
2. chars_raw_imputed.feather (impute missing value with industry median/mean value)
3. chars_rank_no_imputed.feather (standardize chars_raw_no_impute.pkl)
4. chars_rank_imputed.feather (standardize chars_raw_imputed.pkl)

### Information Variables:

- stock indicator: gvkey, permno
- time: datadate, date, year ('datadate' is the available time for data and 'date' is the date of return)
- industry: sic, ffi49
- exchange info: exchcd, shrcd
- return: ret (we also provide original return and return without dividend, you can keep them by modifing impute_rank_output_bchmk.py)
- market equity: me/rank_me

## Method

### Equity Characteristics

This topic is summaried by **Green Hand Zhang** and **Hou Xue Zhang**.

### Portfolio Characteristics

Portfolio charactaristics is the equal-weighted / value-weighted averge of the characteristics for all equities in the portfolio.

The portfolios includes and not limited to:

- Characteristics-sorted Portfolio, see the listed papers and also [Deep Learning in Characteristics-Sorted Factor Models](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3243683)
- DGTW Benchmark, see [DGTW 1997 JF](https://doi.org/10.1111/j.1540-6261.1997.tb02724.x)
- Industry portfolio

