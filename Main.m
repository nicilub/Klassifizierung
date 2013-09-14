%
% Klassifizierung von handgeschriebenen Zahlen mit Hilfe One-vs-all Logistische Regression
% mit Regularisierung und mit einem Neuronalen Netzwerk mit 4 Schichten. 
% Die Input-Schicht besteht aus 400, die zwei inneren Schichten aus
% jeweils 25 und die Output-Schicht aus 10 "Neuronen".
% 
clear all; close all; clc
% Setzen benoetigter Parameter
%n  = 400;  % Anzahl der Features (20x20 Pixel)
S = 25; % Anzahl der "Neuronen" pro versteckte Schicht (zwei)
K = 10;          % 10 Labels, von 1 bis 10 (die Zahl 0 entspricht Label 10)

% Zeilenvektoren, in die die Genauigkeit der Algorithmen fuer jeden Durchlauf der 
% Kreuzvalidierung geschrieben werden
lrGenauigkeit = zeros(1,10); 
nnGenauigkeit = zeros(1,10);

%
%%%%%%%%%%%%%%%%%%%%%%%%%% Laden der Daten und Visualisierung %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
fprintf('\nLaden der Daten...\n');
fprintf('\n');
load('Daten.mat'); % Daten in die Matrix X und in den Label-Vektor y geladen

[m_konst n] = size(X);
zuf_beisp = randperm(m_konst,100); % Zufaellige Wahl von 100 Beispielen 
A = X(zuf_beisp,:);

% Visualisiere die Daten mit Hilfe der Funktion DatenVis
DatenVis(A);

fprintf('Pause. Druecke Enter zum Fortfahren.\n');
pause;

% Erzeuge einen m_konst-dimensionalen Spaltenvektor p, der die Zahlen 1,..,10 mit der gleichen
% Haeufigkeit und in zufaelliger Reihenfolge enthaelt, um die Daten fuer die
% Kreuzvalidierung in 10 gleich grosse Mengen zu splitten
% 
p = 1:m_konst; 
p = mod(p,K) + 1;
r = randperm(m_konst);
p = p(r)';

%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% One vs all %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%

lambda = 1; % Regularisierungsparameter
Trainingszeit_onevsall = zeros(1,10);
% 10-fache Kreuzvalidierung
for i = 1:10

e = i* ones(m_konst,1);
X_test = X(p == e,:); % Testmenge: alle Zeilen mit dem Index, bei dem p als Eintrag i hat
y_test = y(p == e);  % zugehoeriger Label-Vektor
X_train = X(p ~= e,:); % Als Trainingsmenge werden die verbleibenden Zeilen genommen
y_train = y(p ~= e); % zugehoeriger Label-Vektor

m = size(X_train, 1); % Anzahl der Trainings-Beispiele
%
% Trainieren der Daten 
%
tic
fprintf('\nTraining One-vs-All Logistische Regression...Durchlauf %1.0f\n',i);
fprintf('\n');

[Theta] = LogReg(X_train, y_train, K, lambda);

Trainingszeit_onevsall(i) = toc;

% Berechnen der Genauigkeit 
lr_label = lrLabel(Theta, X_test);
lrGenauigkeit(i) = mean(double(lr_label == y_test)) * 100;

end % Ende der Kreuzvalidierung
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Neuronales Netzwerk %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Initialisierung der Startparameter fuer das Neuronale Netztwerk
fprintf('\nInitialisierung der Parameter des Neuronalen Netzwerkes...\n');
fprintf('\n');

initial_Theta1 = zufInitialParameter(n, S);
initial_Theta2 = zufInitialParameter(S, K);
initial_nnParameter = [initial_Theta1(:) ; initial_Theta2(:)];

lambda = 1; % Regularisierungsparameter
optionen = optimset('MaxIter', 50); % Maximale Anzahl der Iterationsschritte in fmincg gleich 50
Trainingszeit_nn = zeros(1,10);

% 10-fache Kreuzvalidierung
for i = 1:10

e = i* ones(m_konst,1);
X_test = X(p == e,:);
y_test = y(p == e);
X_train = X(p ~= e,:);
y_train = y(p ~= e);

m = size(X_train, 1); % Anzahl der Trainings-Beispiele
%
% Trainieren der Daten 
%
tic
fprintf('\nTraining  Neurales Netzwerk...Durchlauf %1.0f \n',i);
fprintf('\n');

% function handle
f = @(t) nnFehler(t,n, S, K, X_train, y_train, lambda);
nnParameter = fmincg(f, initial_nnParameter, optionen);

% Bestimme Theta1 und Theta2 aus nnParameter
Theta1 = reshape(nnParameter(1:S * (n + 1)), S, (n + 1));

Theta2 = reshape(nnParameter((1 + (S * (n + 1))):end), K, (S + 1));

Trainingszeit_nn(i) = toc;

% Berechnen der Genauigkeit 
nn_label = nnLabel(Theta1, Theta2, X_test);
nnGenauigkeit(i) = mean(double(nn_label == y_test)) * 100;

end % Ende der Kreuzvalidierung

fprintf('\nGenauigkeit der Klassifizierung One vs All in allen Durchläufen:\n%5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f\n', lrGenauigkeit);
fprintf('\nDurchschnittliche Genauigkeit: %5.2f\n', mean(lrGenauigkeit));
fprintf('\nGenauigkeit der Klassifizierung Neuronales Netzwerk in allen Durchläufen:\n%5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f\n', nnGenauigkeit);
fprintf('\nDurchschnittliche Genauigkeit: %5.2f\n', mean(nnGenauigkeit));

fprintf('\nTrainingszeiten:');
zeit_onvsall = mean(Trainingszeit_onevsall);
zeit_nn = mean(Trainingszeit_nn);
fprintf('\n');
fprintf('\nDurchschnittliche Trainingszeit One-vs-All:  %4.2f s \n', zeit_onvsall);
fprintf('\nDurchschnittliche Trainingszeit Neuronales Netzwerk: %1.0f min %4.2f s \n', floor(zeit_nn/60), mod(zeit_nn,60));



