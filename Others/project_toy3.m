% Matlab Project
% July 19, 2018

% This program tests the model with toy data - 3

% Running time may take up to 1 minute.

clear;
clc;
close all;

n = 300; % Proportinal to the size of input  
l = 5; %Determines how spread out the data points are

x = linspace(-5,5,n);
x = x';
input = [x , x.^3 + l * randn(n,1)];
input = [input ; [x , x.^3 + l * randn(n,1) - 100] ];
input = [input ; [x , x.^3 + l * randn(n,1) + 100] ];
label = ones(n,1);
label = [label ; zeros(2*n,1)];

[L,N] = size(input);

% Plot the input data   
fig1 = figure('Name','Data Distribution'); movegui(fig1,'northwest');
figure(1); hold on;      
scatter(input(1:n,1), input(1:n,2) , 'k.');
scatter(input(n+1:3*n,1), input(n+1:3*n,2) , 'g.');
leg1 = legend('Label 0','Label 1','Location' , 'north'); 
leg1.FontSize = 14; hold off;
hold off;

% Shuffle the data in order to get rid of the upward price pressure
idx = randperm(length(label));
label = label(idx);
input = input(idx,:);

a = 5;              % Rate of gradient descent
input_neuron = N;   % Number of neurons in input layer
hidden_neuron = 10; % Number of neurons in hidden layer
output_neuron = 1;  % Number of neurons in output layer

                    % Divide the data into training and test
p = 0.8;            % Partition  rate of training and test
training            = input(1:round(p * L),:); 
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

no_epoch = 50; % Number of epochs

% Gradient descent algorithm & Plotting the loss at each step
fig2 = figure('Name','Epoch vs. Loss in Real Time'); movegui(fig2,'north');
figure(2) ;  hold on;
for epoch = 1:no_epoch
    loss = 0;
    t = length(training);
    for i = 1: t
        x = training(i,:);
        z1 = x * w1 + b1;
        a1 = sigmoid(z1);
        z2 = a1 * w2 + b2;
        a2 = sigmoid(z2);
        l = label_training(i,:);
        e = l - a2;
        loss =  loss + sum( e.^2 );

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
    loss_store(epoch) = loss/t;
    xlim([0,no_epoch]); ylim([0,0.7]); 
    xlabel('Epochs'); ylabel('Loss'); axis manual ;
    scatter(epoch,loss_store(epoch),100); drawnow; 
end
hold off;


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
fig3 = figure('Name','Accuracy on the Test Data'); movegui(fig3,'northeast');
figure(3); hold on;
accuracy = 0;
for i = 1: length(test)
    if round(predicted_labels(i,:)) ==  label_test(i,:)
        accuracy = accuracy + 1;
        plot(test(i,1), test(i,2)  , 'b.' ); hold on;
    elseif round(predicted_labels(i)) ~=  label_test(i)  
        plot(test(i,1), test(i,2)  , 'r.' ); hold on;
    end
end

accuracy = accuracy / length(test);
accuracy = num2str(accuracy);
accuracy = strcat('Accuracy = ', accuracy);
leg2 = legend(accuracy, 'Location', 'North');
leg2.FontSize = 14; hold off;

% Plotting the predicted distribution
[X,Y] = meshgrid(-5:0.1:5 ,-250:1:250 );
[xsize, ysize] = size(X);
tera_labels = zeros(size(X));

for i = 1: xsize
    for j = 1: ysize
    x = [X(i,j) , Y(i,j)];
    z1 = x * w1 + b1 ;
    a1 = sigmoid(z1);
    z2 = a1 * w2 + b2;
    a2 = sigmoid(z2);
    tera_labels(i,j) = a2; % Store the predicted values
    end
end

fig4 = figure('Name','Label Distribution of the Model: Yellow (1) -> Blue (0)') ;
movegui(fig4,'south');xlim([-5,5]); ylim([-250,250]); figure(4);
mesh(X,Y,tera_labels);
az = 0;
el = 90;
view(az, el);

% Activation function: Sigmoid
function t = sigmoid(t)
    t = arrayfun(@(x) 1 / (1+exp(-x)) , t);
end

% Gradient of sigmoid function
function t = sigmoid_der(t)
    t = arrayfun(@(x) (1 / (1+exp(-x))) * (1- (1 / (1+exp(-x)))) , t);
end
