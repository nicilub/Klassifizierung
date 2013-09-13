function h = DatenVis(X)
%
% Stellt die Daten in einem grauen Display dar
%
% Graues Bild
colormap(gray);

% Berechnung der Seitenlaenge
[m n] = size(X); % m = Anzahl der Beispiele von Bildern, n = Anzahl der Pixel eines Bildes
seitenl = round(sqrt(n));

% Berechnung der Anzahl der Felder pro Zeile und pro Spalte
anz_zeilen = floor(sqrt(m));
anz_spalten = ceil(m / anz_zeilen);

% Luecke zwischen den Bildern
luecke  = 1;

% Setze leeres Display
vis_array = - ones(luecke + anz_zeilen * (seitenl + luecke), luecke + anz_spalten * (seitenl + luecke));

% Kopiere jedes Beispiel
beisp = 1;
for j = 1:anz_zeilen
	for i = 1:anz_spalten
		if beisp > m, 
			break; 
		end
		
		% Bestimmung des maximalen Wert in einem Feld
		max_wert = max(abs(X(beisp, :)));
		vis_array(luecke + (j - 1) * (seitenl + luecke) + (1:seitenl), ...
		              luecke + (i - 1) * (seitenl + luecke) + (1:seitenl)) = ...
						reshape(X(beisp, :), seitenl, seitenl) / max_wert;
		beisp = beisp + 1;
	end
	if beisp > m, 
		break; 
	end
end

%  Display
h = imagesc(vis_array, [-1 1]);

% ohne Axen
axis image off

drawnow;

end
