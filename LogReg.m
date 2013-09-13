function [Theta] = LogReg(X, y, K, lambda)
% Berechnet fuer jedes Label die optimalen Parameter der Logistischen Regression. 
% Input: Trainingsmenge X, Labelvektor y, Anzahl der Label K, Regularisierungsparameter lambda
% Output: K x (n + 1) - dimensionale Matrix Theta, bei der eine Zeile k die Parameter enthaelt, bei der
% der Fehler E zum Label k minimiert wird

% Setzen benoetigter Parameter
[m n] = size(X);
Theta = zeros(K, n + 1); % Output Matrix, die fuer jedes Label die optimalen Parameter 
                         % der logistischen Regression angibt

% Hinzufuegen einer Spalte mit Einsen
X = [ones(m, 1) X];
%
 
%y_label = zeros(m,1);
	
for l = 1:K 
	
	v = l * ones(m,1);
	y_label = (v == y); % Gibt fuer jedes Beispiel an, ob es zu Label l gehoert (0 = nein, 1 = ja)

     hilfs_theta = zeros(n + 1, 1); 
	 % Setze Optionen fuer die Funktion fmincg
	 optionen = optimset('GradObj', 'on', 'MaxIter', 50); 
% Berechnung der Parameter, die den Fehler der Logistischen Regression minimieren
% mit Hilfe von fmincg
    Theta(l,:) = (fmincg (@(t)(lrFehler(t, X, y_label, lambda)), hilfs_theta, optionen))'; 
	
     
end 

end