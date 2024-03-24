@echo off
echo Copie des dossiers en cours...

rem Spécifiez les chemins complets des dossiers source et de destination
set "sourceFolder1=C:\Users\NicolasEBY\Documents\GitHub\electrical_and_neural_signal_processing\docs_sphinx\_build\html"
set "destinationFolder=C:\Users\NicolasEBY\Documents\GitHub\electrical_and_neural_signal_processing\docs"

rem Assurez-vous que le dossier de destination existe
mkdir "%destinationFolder%" 2>nul

rem Copiez les dossiers vers le dossier de destination
xcopy /E /I "%sourceFolder1%" "%destinationFolder%\"

echo Copie terminée.
pause