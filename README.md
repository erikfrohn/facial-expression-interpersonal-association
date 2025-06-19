# Facial Associative Coupling in Expressions as a Non-Invasive Indicator of Team Performance 
E.T. Frohn

code belonging to paper. 

# Repository structure
Preprocessing is excluded from this repository. 
`\data\` contains all data: Action Units, Facial Factors, CRQA output
`\data\au` contains all action unit files, without any structure.
`\data\{pair}\features\` contains all facial factor data for the specific pair
`\data\{pair}\correlations\` contains all CRQA output, both delay profiles and overall RR values.

# Running
- main.ipynb - data processing
- analysis.ipynb - visualisation

# Utility files
- video_transformation.py: preprocessing that was done outside of the current repository
- feature_selection.py: transforming Action Units (17) to Facial Factors (6)
- toolbox.py: needed for gridsearch, gridsearch visualisation and NEM visualisation.
- validation_testing.py: real vs fake validation
- visualisation.py: relation TPxRR and TPxRR|Zoom visualisation and significance validation.
