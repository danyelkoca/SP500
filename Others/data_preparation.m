% Matlab Project
% July 15, 2018

% This program gets the data found in the directory
% computes indicators, attaches them to a matrix
% returns this matrix, name of the columns
% an input matrix and a label matrix ready for machine learning.


function [M,A,input, labels, stock_min, stock_max] = data_preparation(filename1, filename2)

%{
---
MATRIX M
---
(number) indicates the features that are included in the analysis. 
(#) indicates the intermediary columns that are deleted at the end of this
program.
Others are intermediate steps.

--  Original data
1.  column = DATE
2.  column = OPEN
3.  column = HIGH
4.  column = LOW
5.  column = CLOSE
6.  column = ADJUSTED CLOSE
7.  column = VOLUME

--  Below is attached in this program.
8.  column = 50-Day Simple Moving Average (1)
9.  column = 12-Day Exponential Moving Average (#)
10. column = 26-Day Exponential Moving Average (2)
11. column = Up (#)
12. column = Down (#)
13. column = Smoothed MMU (#)
14. column = Smoothed MMD (#)
15. column = Relative Strength Index (3)
16. column = Implied Volatility Index (VIX) (4)
17. column = 10-Day Realized Volatility (5)
18. column = B% (6)
19. column = Alexander's Filter (7)
20. column = Stochastic Osciallator: K% (#)
21. column = Stochastic Osciallator: D% (8)
22. column = Typical Price - OMIT IN ANALYSIS
23. column = Positive Raw Money Flow (#)
24. column = Negative Raw Money Flow (#)
25. column = 14-day Money Flow Index (MFI) (9)
26. column = Accumulation - Distribution Line (ADL) (10)
27. column = 3-Day EMA of ADL (#)
28. column = 10-Day EMA of ADL (#)
29. column = Chaikin Oscillator (11)
30. column = Moving Average Convergence - Divergence Oscillator (MACD) (#)
31. column = MACD Signal Line (#)
32. column = MACD - Signal Line (12)
33. column = Williams R% oscillator (13)
%}

% Get the data
M = csvread(filename1,1,0);
M = [M , zeros(length(M) , 25)];
VIX = csvread(filename2,1,1);
L = length(M);
V = length(VIX);
M((L- V + 1):L,16) = VIX(:);

% Start computing the indicators

% 8th column : 50 Day Simple Moving Average
for i = 50:L
    M(i,8) = mean(M(i-49:i, 6));
end

% 9th  - 10th columns : 12-, 26-Day Exponential Moving Averages, respectively
n = 2 / 13; m = 2 / 27;
M(1,9) = M(1,6) ;  M(1,10)  =  M(1,6);
for i = 2:L
    M(i,9)  = n * M(i,6) + (1-n) * M(i-1,9);
    M(i,10) = m * M(i,6) + (1-m) * M(i-1,10);
end

% 11 th column: U , 12th column: D > Intermediate steps for calculation of
% Relative Strength Index: RSI
for i = 2:L
    if M(i,6) > M(i-1,6)
        M(i,11) = M(i,6) - M(i-1,6);
    elseif M(i,6) < M(i-1,6)
        M(i,12) = M(i-1,6) - M(i,6);
    end
end

% 13th column = SMMU , 14th column = SMMD, 15th column = RSI
k = 1/14;
M(1,13) = M(1,11) ;  M(1,14)  =  M(1,12); M(i,15) = 0;
for i = 2:L
    M(i,13) = k * M(i,11) + (1-k) * M(i-1, 13);
    M(i,14) = k * M(i,12) + (1-k) * M(i-1, 14);
    M(i,15) = 100 - 100 / ( 1+ ( M(i,13) / M(i,14) ));
end

% 16th column = Implied Volatility = VIX
for i = 1: L
    if M(i,16) == 0
        M(i,16) = std( M(i+1:i+30 , 6));
    else
        M(1:i,16) = M(1:i,16)  - mean(M(1:i,16)) + mean(VIX);
        break
    end
end

% 17th column = 10-Day Relized Volatility
for i = 10:L
    M(i,17) = std( M(i-9:i,6) );
end

% 18th column = B% : Bollinger Bands' Ratio
for i = 20:L
  up = mean( M(i-19:i,6)) + 2* std(M(i-19:i,6));
  low = mean(M(i-19:i,6)) - 2* std(M(i-19:i,6));
  M(i,18) = (M(i,6) - low) / (up - low) ; 
end

% 19th column: Alexander's Filter
for i = 2:L
    if ( M(i,6) / M(i-1,4)) > 1.01
        M(i,19) = 1;
    elseif (M(i,6) / M(i-1, 3)) < 0.99
        M(i,19) = -1;
    end
end

% 20th column: Stochastic Oscillator : K%
% 33rd column = Williams R% Oscillator
for i = 14:L
    lowest_low = min(M(i-13:i,4));
    highest_high = max(M(i-13:i,3));
    M(i,20) = (M(i,6) - lowest_low) / (highest_high - lowest_low);
    M(i,33) = (highest_high - M(i,6))/(highest_high - lowest_low) * (-100);
end

% 21st column: Stochastic Oscillator : D%
% D% is a  3-day smoothed version of K%
for i = 16:L
     M(i,21) = mean(M(i-2:i,20));
end

% 22nd column: Typical price: Preparation for calculation of Money Flow
% Index: MFI
for i =1:L
    M(i,22) = mean([M(i,3) , M(i,4) , M(i,6)]); 
end

% 23rd column: Positive Raw Money Flow
% 24th column: Negative Raw Money Flow
for i = 2:L
    if M(i,22) > M(i-1, 22)
        M(i,23) = M(i,22) * M(i,7); % Positive money flow
     elseif M(i,22) < M(i-1,22)
        M(i,24) = M(i,22) * M(i,7); % Negative money flow
     end
end

% 25th column: 14-day Money Flow Index
for i = 15:L
  money_flow_ratio = sum(M(i-13:i,23)) / sum(M(i-13:i,24));
  M(i,25) = 100 - 100 / (1+ money_flow_ratio);
end

% 26th column: Accumulation - Distribution 
for i = 1:L
    if M(i,3) ~= M(i,4)
        money_flow_multiplier = (2* M(i,6) - M(i,4) - M(i,3)) / ( M(i,3) - M(i,4) );
        money_flow_volume = money_flow_multiplier * M(i,7);
        if i==1
            M(i,26) = money_flow_volume
        else
            M(i,26) = M(i-1,26) + money_flow_volume;
        end
    end
end

% 27th column: 3-day Exponential Moving Average of ADL
% 28th column: 10-day Exponential Moving Average  of ADL
p = 2/4 ; r = 2/11;
M(1,27) = M(1,26) ; M(1,28) = M(1,26);
for i = 2:L
    M(i,27) = p * M(i,26) + (1-p) * M(i-1,27);
    M(i,28) = r * M(i,26) + (1-r) * M(i-1,28);
end

% 29th column: Chaikin Oscillator
for i = 1:L
     M(i,29) = M(i,27) - M(i,28);
end

% 30th column = MACD Convergence Line
for i =1:L
     M(i,30) = M(i,9) - M(i,10);
end

% 31st column = Signal line
M(1,31) = M(1,30); r = 1 / 10;
for i =2:L
    M(i,31) = r * M(i,30) + (1-r) * M(i-1,31);
end

% 32nd column = MACD - Signal Line
for i = 1:L
    M(i,32) = M(i,30) - M(i,31);
end

% Delete the columns of intermediate steps
M(:,31) = [] ; M(:,30) = [] ; M(:,28) = [] ; M(:,27) = [] ;
M(:,24) = [] ; M(:,23) = [] ; M(:,22) = [] ; M(:,20) = [] ; 
M(:,14) = [] ; M(:,13) = [] ; M(:,12) = [] ; M(:,11) = [] ;
M(:,9) =  [] ; M(:,7) =  [] ; M(:,5) =  [] ; M(:,4) =  [] ; 
M(:,3) =  [] ; M(:,2) =  [] ; 

% Titles of date, adjusted close (dependent variable), and features
A  = {'Date', 'Adj Close', '50-D SMA','26-D EMA',...
      'RSI', 'VIX', '10-D Real Vol', 'B%', 'ALF', 'D%', 'MFI',...
      'ADL', 'Chaikin Osc', 'MACD - Signal', 'Williams R%'};
      
% Clear intermediate variables
clear RS; clear VIX; clear filename1; clear filename2; 
clear highest_high; clear i; clear k; clear low; clear lowest_low;
clear m; clear money_flow_index; clear money_flow_multiplier;
clear money_flow_ratio; clear money_flow_volume; clear n; 
clear p; clear q; clear r; clear up; clear r; clear V;

%{
At the end; the columns of M are:
1  - Date 
2  - Adj Close
3  - 50-D SMA
4  - 26-D EMA
5  - RSI
6  - VIX
7  - 10-D Real Vol
8  - B
9  - ALF
10 - D%
11 - MFI
12 - ADL
13 - Chaikin Osc
14 - MACD - Signal
15 - Williams R%
%}

M = M(51:end,:); % Delete first 50 raws because 50-day Simple moving average
% feature does not have observations

L = length(M);
labels = log10(M(2:L,2)); % get the log of next day's price because it grows exponentially
input = M;

% Normalize labels
stock_min = min(labels); stock_max = max(labels);
for i =1:L-1
    labels(i) = (labels(i) - stock_min) / (stock_max - stock_min); 
end

input(L,:) = []; % Delete last row since it does not have label
M(L,:) = [];     % Delete last row 

input(:,2) = []; % Delete adjusted close column
input(:,1) = []; % Delete date column
input(:,1) = log10(input(:,1)); %Normalize SMA because it grows exponentially
input(:,2) = log10(input(:,2)); %Normalize EMA because it grows exponentially

% Normalize input
for i=1:13
    if i ~= 7 % Don't normalize Alexander's Filter because it is binary
        column_min = min(input(:,i));
        column_max = max(input(:,i));
        input(:,i) = (input(:,i) - column_min) / (column_max - column_min);
    end
end
end
