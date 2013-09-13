function [E, grad] = lrFehler(theta, X, y, lambda)
% Gibt E als Wert des Fehlers der regularisierten logistischen Regression
% bei gegebenem theta, X,y und lambda und grad als den Gradienten des Fehlers zurueck

% Initialisierung
m = length(y); % Anzahl der Beispiele
E = 0; 
grad = zeros(size(theta));

% Hilfsvariablen
e = ones(m,1);  
t = theta;  
t(1) = 0;

% Vektorisierte Berechnung von E und grad
E = -(1/m)*((log(sigmoid(X*theta)))'*y + (log(e - sigmoid(X*theta)))'*(e - y)) + lambda/(2*m)*(t'*t) ;
 
grad = (1/m)*X'*(sigmoid(X*theta) - y) + lambda/m*t;
grad = grad(:);

end
