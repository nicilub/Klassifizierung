function [E grad] = nnFehler(nnParameter, n, S, K, X, y, lambda)
%
% Berechnet den Fehler des Neuronalen Netzwerkes und den Gradienten mit Hilfe von Backpropagation
%
% Initialisierung

Theta1 = reshape(nnParameter(1:S * (n + 1)), S, (n + 1));

Theta2 = reshape(nnParameter((1 + (S * (n + 1))):end), K, (S + 1));

m = size(X, 1);
        
E = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Berechnung des Fehlers %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Fuege Spalte von Einsen zu X und transponiere 
A1 = [ones(m,1) X]';

% Berechne die Activations fuer jedes Beispiel in Matrix A 
Z2 = Theta1*A1;  
A2 = [ones(1,m); sigmoid(Z2)];  
Z3 = Theta2*A2;
A3 = sigmoid(Z3);

% Berechne Matrix Y, die den binaeren Vektor fuer jedes y_i enthaelt 
% (fuer y_i = 3 ist dieser (0 0 1 0 0 0 0 0 0 0))
Y = zeros(m,K);

for i = 1:m
	for j = 1:K
		Y(i,j) = (y(i) == j);
	end
end  



% Berechne den Fehler E  

E = (-1/m) * (sum(diag(Y * log(A3))) + sum(diag((1 - Y) * log(1 - A3))));
 
% Regularisierung   

Theta1_reg = Theta1(:, 2:(n + 1));
Theta2_reg = Theta2(:, 2:(S + 1));

E = E + lambda/(2*m)*(sum(sum(Theta1_reg.*Theta1_reg)) + sum(sum(Theta2_reg.*Theta2_reg)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Berechnung des Gradienten mit Backpropagation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  

Y = Y'; 

for i =1:m    

	delta3 = A3(:,i) - Y(:,i);  
	delta2 = Theta2' * delta3; 
	delta2 = delta2(2:end);
    delta2 = delta2 .* sigmoidGradient(Z2(:,i));  

	Theta1_grad = Theta1_grad + delta2 * A1(:,i)';  
	Theta2_grad = Theta2_grad + delta3 * A2(:,i)';  
end

Theta1_grad = 1/m*Theta1_grad; 
Theta2_grad = 1/m*Theta2_grad;  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Regularisierung %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 	

Theta1_grad_reg =  Theta1; 
Theta2_grad_reg =  Theta2; 

Theta1_grad_reg(:,1) = zeros(size(Theta1(:,1))); 
Theta2_grad_reg(:,1) = zeros(size(Theta2(:,1))); 

Theta1_grad = Theta1_grad + lambda/m *  Theta1_grad_reg;
Theta2_grad = Theta2_grad + lambda/m *  Theta2_grad_reg;


grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
