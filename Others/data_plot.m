% Matlab Project
% July 15, 2018

% This program visualizes the closing price of S&P 500 and 
% a desired indicator. 


function data_plot
[M,A] = data_preparation("SP.csv", "VIX.csv");

% Prompt user for input
in = { ' ', 'Indicator Visualizer -- All rights reserved :)' ,' ',...
    'Please enter the number that corresponds to the indicator to see the time series:',...
    '1 - 50 Day Simple Moving Average', '2 - 26 Day Exponential Moving Average',...
     '3 - Relative Strength Index', '4 - Implied Volatility Index', '5 - 10 Day Realized Volatility',...
     '6 - B% Oscillator',"7 - Alexander's Filter", '8 - D% Oscillator', '9 - Money Flow Index',...
     '10- Accumulation - Distribution Line', '11- Chaikin Oscillator',...
     '12- Moving Average Convergence - Divergence -- Signal', '13- Williams R%'...
     'and the time range: {indicator_no,''dd.MM.yyyy'',''dd.MM.yyyy''} ',...
     'e.g.: {12,''01.04.1987'',''05.04.1989''}',...
     'Or leave the range section empty if you want to see the time series for the whole dataset. e.g.: 5'};

in=sprintf('%s\n',in{:});
in = input(in);

clc;

% Convert dates to datetime objects
C = num2cell(M(:,1)); C = cellfun(@num2str, C, 'UniformOutput', false);
TOP =  datetime(C,'InputFormat','yyyyMMdd') ; dn = TOP';

% Check if the input is valid
switch class(in)
    case 'double'
        y = in;
        min_lim = min(dn);
        max_lim = max(dn);
    case 'cell'
        y = in{1};
        min_lim = datetime( in{2},'InputFormat','dd.MM.yyyy');
        max_lim = datetime( in{3},'InputFormat','dd.MM.yyyy');
    otherwise
        error('Please check the input: example:[12,''01.04.1987'',''05.04.1999'']')
end

% Plot the indicator
switch y
    case {1,2}
        clf
        p = semilogy(dn, M(:,2), dn, M(:,y+2));
        p(2).LineWidth = 3;
        legend('S&P 500 Price', A{y+2}, 'location', 'northwest')
        xlim([min_lim , max_lim])
    case 7
        clf
        figure(1)
        subplot (4, 1, [1,3]);
        semilogy(dn , M(:,2) )
        xlim([min_lim , max_lim])
        legend('S&P 500 Price', 'location', 'northwest')
        subplot (4, 1, 4);
        plot(dn, M(:,y+2), '.')
        xlim([min_lim , max_lim])
        ylim([-1.2, 1.2])
        legend(A{y+2}, 'location', 'northwest')
    case 10
        clf
        yyaxis left
        semilogy(dn, M(:,2))
        yyaxis right
        plot( dn, M(:,y+2))
        legend('S&P 500 Price', A{y+2}, 'location', 'northwest')
        xlim([min_lim , max_lim])
    case 12
        clf
        figure(1)
        subplot (4, 1, [1,3]);
        semilogy(dn , M(:,2) )
        xlim([min_lim , max_lim])
        legend('S&P 500 Price', 'location', 'northwest')
        subplot (4, 1, 4);
        plot(dn, M(:,y+2))
        hold on
        plot(dn, zeros(length(dn), 1))
        hold off
        xlim([min_lim , max_lim])
        legend(A{y+2}, 'location', 'northwest')
    case {3,4,5,6,8,9,11,13}
        clf
        figure(1)
        subplot (4, 1, [1,3]);
        semilogy(dn , M(:,2) )
        xlim([min_lim , max_lim])
        legend('S&P 500 Price', 'location', 'northwest')
        subplot (4, 1, 4);
        plot(dn, M(:,y+2))
        xlim([min_lim , max_lim])
        legend(A{y+2}, 'location', 'northwest')
    otherwise
        error('Please enter a number between 1-13 (inclusive).')
end

end
