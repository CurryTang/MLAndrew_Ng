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
for i = 1:m
    xi = X(i, :);
    yi = y(i, :);
    h_theta = sigmoid(dot(theta, xi));
    J = J + (-yi * log(h_theta) - (1 - yi) * log(1 - h_theta));
end
J = J / m;
J = J + (lambda / (2 * m)) * (sum(theta.^2) - theta(1)^2)



for k = 1:size(theta)
    sum = 0;
    for j = 1:m
        xi_prime = X(j, :);
        yi_prime = y(j, :);
        h_theta_prime = sigmoid(dot(theta, xi_prime));
        sum = sum + (h_theta_prime - yi_prime) * xi_prime(k);
    end
    grad(k) = sum / m;
    if (k != 1)
        grad(k) = grad(k) + (lambda / m) * theta(k);
end











% =============================================================

end
