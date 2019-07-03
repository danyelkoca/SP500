% Matlab Project
% July 19, 2018

% This program retrieves the financial data from the directory,
% calculates 13 technical indicators,
% and applies the deep learning model.

% Running time may take up to 1 minute.

clear;
clc;
close all;

[M,A,input,label,stock_min, stock_max] = data_preparation("SP.csv", "VIX.csv");
clear M; clear A;
[L,N] = size(input); % L = number of observations, N = number of features

% Shuffle the data in order to get rid of the upward price pressure
idx = randperm(length(label));
input1 = input; % edit
label1 = label; % edit
predicted_labels1 = zeros(L,1);
label = label(idx);
input = input(idx,:);

a = 1;              % Rate of gradient descent
input_neuron = N;   % Number of neurons in input layer
hidden_neuron = 25; % Number of neurons in hidden layer
output_neuron = 1;  % Number of neurons in output layer

                    % Divide the data into training and test
p = 0.9;            % Partition  rate of training and test
training            = input(1:round(p * L),:); 
A2 = zeros(1,round(p*L));
label_training      = label(1:round(p * L),:);
test                = input(round(p * L) + 1 : L ,:);
label_test          = label(round(p * L) + 1 : L ,:);
loss_store          = zeros(length(training),1); % Array to store loss values for plotting
predicted_labels    = zeros(length(test),output_neuron); % Array for storing predicted values in testing

% Weight - bias matrices
w1 =   randn(input_neuron,hidden_neuron);
b1 =   randn(1,hidden_neuron);
w2 =   randn(hidden_neuron,output_neuron);
b2 =   randn(1,output_neuron);

no_epoch = 100; % Number of epochs

% Gradient descent algorithm
for epoch = 1:no_epoch
    for i = 1: length(training)
        x = training(i,:);
        z1 = x * w1 + b1;
        a1 = sigmoid(z1);
        z2 = a1 * w2 + b2;
        a2 = sigmoid(z2);
        l = label_training(i,:);
        e = l - a2;
        loss =  sum( e.^2 );
        if epoch == no_epoch % Get the loss at the last epoch
            loss_store(i) = loss;
        end

        % Gradient calculation
        delta_w2 = (-2 * sigmoid_der(z2) .* e) .* a1';
        delta_b2 = (-2 * sigmoid_der(z2) .* e);
        delta_w1 = (-2 *  (e .* sigmoid_der(z2)) * w2' .* sigmoid_der(z1)) .* x';
        delta_b1 = (-2 *  (e .* sigmoid_der(z2)) * w2' .* sigmoid_der(z1));

        % Weight - bias matrices update
        w1 = w1 - a.*delta_w1;
        w2 = w2 - a.*delta_w2;
        b1 = b1 - a.*delta_b1;
        b2 = b2 - a.*delta_b2;
    end
end

% Plot the loss over the last epoch
fig1 = figure('Name','Loss at The Last Epoch'); movegui(fig1,'west');
figure(1); plot(loss_store, '.'); title('Loss at The Last Epoch')
xlabel('Iterations'); ylabel('Loss'); xlim([0,p*L])

% Testing the model 
for i = 1: length(test)
    x = test(i,:);
    z1 = x * w1 + b1 ;
    a1 = sigmoid(z1);
    z2 = a1 * w2 + b2;
    a2 = sigmoid(z2);
    predicted_labels(i,:) = a2; % Store the predicted values
end

% Compute and plot the accuracy
fig2 = figure('Name','Accuracy'); movegui(fig2,'east');
figure(2); plot(predicted_labels, label_test, '.');
rsquared = num2str(r_squared(predicted_labels, label_test));
accuracy = strcat('Test Accuracy: R-squared=', rsquared);
xlim([0,1]); ylim([0,1]); title(accuracy);
xlabel('Predicted Labels'); ylabel('Actual Labels');


for i = 1: length(input1)
    x = input1(i,:);
    z1 = x * w1 + b1 ;
    a1 = sigmoid(z1);
    z2 = a1 * w2 + b2;
    a2 = sigmoid(z2);
    predicted_labels1(i,:) = 10 ^ ( a2 * (stock_max - stock_min) + stock_min); % Store the predicted values
end

label1 =  (label1 .* (stock_max - stock_min)) + stock_min;
label1 = 10 .^ label1;

len = length(predicted_labels1);
semilogy( 1:len, predicted_labels1, 'k' ,1:len, label1, 'b') 




% Accuracy function
function t = r_squared(x,y)
    ss_res = sum( (y-x).^2 );
    m = mean(y);
    ss_tot = sum( (y-m).^2 );
    t = 1 -  ss_res / ss_tot;
end

% Activation function: Sigmoid
function t = sigmoid(t)
    t = arrayfun(@(x) 1 / (1+exp(-x)) , t);
end

% Gradient of sigmoid function
function t = sigmoid_der(t)
    t = arrayfun(@(x) (1 / (1+exp(-x))) * (1- (1 / (1+exp(-x)))) , t);
end
