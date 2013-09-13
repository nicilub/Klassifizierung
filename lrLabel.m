function label = lrLabel(Theta, X)
% Gibt einen Vektor zurueck, der fuer jedes Beispiel, das wahrscheinlichste Label enthaelt

m = size(X, 1);
label = zeros(m, 1);

% Hinzufuegen einer Spalte von Einsen
X = [ones(m, 1) X];
 
[dummy,I] = max((sigmoid(X * Theta'))');    

label = I';

end
