# Stock Market Prediction with Machine Learning
This is a project I have done using matlab in the spring of 2018. It is an implementation of machine learning on stock market data.

For this project, no libraries were used. The deep learning algorithm was written manually based on mathematical principles.

SP500.pdf includes background, method, and how to run the programs found in this repository. Sort of a walk-through. It is intended for people who don't have a thorough understanding of stock market analysis.

Figures include the figures found in the SP500.pdf file.

project_stock.m is the main algorithm that trains with the historical data and predicts the SP500's next day price.

Others folder include;
2 csv files are the daily data for S&P 500 and VIX index, both retrieved from Yahoo Finance. S&P goes back to 1950s while VIX goes back to 1990s. I used a modified version of realized standard deviation to extrapolate VIX into 1950s.

The rest are matlab files which test the algorithm with some toy data, retrieve and plot the financial data. 

Some nice figures from this project:

![alt text](https://github.com/danyelkoca/matlab-project/blob/master/Figures/Figure%202.png)
![alt text](https://github.com/danyelkoca/matlab-project/blob/master/Figures/Figure%2012.png)
![alt text](https://github.com/danyelkoca/matlab-project/blob/master/Figures/Figure%2014.png)

