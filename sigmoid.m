function g = sigmoid(z)
% Berechnung der sigmoid-Funktion fuer die Logistische Regression

g = 1.0 ./ (1.0 + exp(-z));
end
