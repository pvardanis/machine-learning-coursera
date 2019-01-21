function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

a1 = [ones(size(X, 1),1), X]; % adding a_0(1) / input units in layer 1 
z2 = a1 * Theta1';
a2 = sigmoid(z2); % hidden layer units
a2 = [ones(size(a2, 1),1), a2] % adding a_0(2) / units in hidden layer 2 
z3 = a2 * Theta2';
y_pred = sigmoid(z3); % 5000 x 26
[max_label,p] = max(y_pred,[],2) % row max, max value = predicted class






% =========================================================================


end
