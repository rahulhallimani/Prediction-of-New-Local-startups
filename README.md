Starbucks Effect on Startups in Hamburg
Overview
This project analyzes the potential increase in the number of startups in various districts of Hamburg if a new Starbucks were to be introduced. The analysis is based on data related to business registrations, deregistrations, and company establishments.

Approach
Data Normalization:

Data related to business registrations, deregistrations, and establishments were normalized and inserted into an SQLite database.
Prediction Model:

A function was defined to predict the growth in the number of startups based on a study indicating that a new Starbucks increases startups by 5-11.8%.
Data Visualization:

The predicted increase in startups was visualized using a bar chart.
Files
hamburg_business_database.db: SQLite database containing normalized data.
RAHUL_Hamburg.py: Python script for data insertion, prediction, and visualization.
combined_predicted_growth.csv: CSV file containing the predicted growth of startups per district.
startup_growth_plot.png: Plot showing the predicted increase in startups per district.
README.md: This file.
Usage
To reproduce the analysis:

Ensure all dependencies are installed (e.g., pandas, sqlite3, matplotlib).
Run ``RAHUL_Hamburg.py`.
View the generated plot and combined_predicted_growth.csv for results.
