function label = nnLabel(Theta1, Theta2, X)
% Gibt einen Vektor zurueck, der fuer jedes Beispiel das wahrscheinlichste Label enthaelt
m = size(X, 1);
K = size(Theta2, 1);

label = zeros(size(X, 1), 1);

h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
[dummy, label] = max(h2, [], 2);


end
