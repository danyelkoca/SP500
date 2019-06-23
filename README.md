# Stock Market Prediction with Machine Learning
This is a project I have done using matlab in the spring of 2018. It is an implementation of machine learning on stock market data.

For this project, no libraries were used. The deep learning algorithm was written manually based on mathematical principles.

SP500.pdf includes background, method, and how to run the programs found in this repository. Sort of a walk-through. It is intended for people who don't have a thorough understanding of stock market analysis.

project_stock.m is the main algorithm that trains with the historical data and predicts the S&P500's next day price.

Figures file include the figures found in the SP500.pdf file.

Data file includes the S&P500 and VIX historical data, both retrieved from Yahoo Finance. S&P goes back to 1950s while VIX goes back to 1990s. I used a modified version of realized standard deviation to extrapolate VIX into 1950s.

Others file include codes that test the machine learning algorithm with toy data. There is also a code that can visualize the S&P500 along with a technical indicator for the desired period.

Some nice figures from this project:

![alt text](https://github.com/danyelkoca/matlab-project/blob/master/Figures/Figure%202.png)
![alt text](https://github.com/danyelkoca/matlab-project/blob/master/Figures/Figure%2012.png)
![alt text](https://github.com/danyelkoca/matlab-project/blob/master/Figures/Figure%2014.png)

