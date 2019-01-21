function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h_theta = sigmoid(X*theta); % m x 1 hypothesis matrix

% J cost computation
term1 = -y'*log(h_theta);
term2 = (1-y)'*log(1-h_theta);
unregularized_cost = (term1-term2)/m ;
reg_term = (lambda/(2*m)) * (theta(2:length(theta))'*theta(2:length(theta)));
J = unregularized_cost + reg_term;

% Gradient computation
theta(1) = 0;
theta_reg = (lambda/m) * theta;
grad = (X'*(h_theta-y))./m + theta_reg ; % n x 1 matrix




% =============================================================

end
