function g = sigmoidGradient(z)
% Berechnet den Gradienten der sigmoid-Funktion
g = zeros(size(z));

g = sigmoid(z).*(1 - sigmoid(z));

end
