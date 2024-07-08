#!/usr/bin/env python
# coding: utf-8

# In[93]:


pip install pandas sqlalchemy pyodbc


# In[94]:


import pandas as pd

file_path = r"C:\Users\LENOVO\Downloads\D_I_2_j23_HH (1) (1).xlsx"
excel_data = pd.ExcelFile(file_path)


# In[95]:


print(excel_data.sheet_names)


# In[96]:


t1_1 = excel_data.parse(' T1_1')
columns = t1_1.iloc[1].fillna('') + ' ' + t1_1.iloc[2].fillna('') + ' ' + t1_1.iloc[3].fillna('')
columns = columns.str.strip().str.replace('\n', ' ').str.replace('  ', ' ')
t1_1_cleaned = t1_1.iloc[4:].copy()
t1_1_cleaned.columns = columns
t1_1_cleaned.reset_index(drop=True,inplace=True)


# In[97]:


t1_1_cleaned


# In[98]:


# Split the DataFrame into two parts
year_wise = t1_1_cleaned.loc[1:14]  # Remove the first row with null values
month_wise_2023 = t1_1_cleaned.loc[17:28].reset_index(drop=True)
year_wise .columns = ['Year','Business_regs_total', 'change_prev_reg_yr_%', 'New_building_no.', 'Business_deregs_total', 'change_prev_dereg_yr_%', 'Including_task_no.']


# In[99]:


year_wise


# In[100]:


month_wise_2023.columns = ['Month','Business_regs_total', 'change_prev_reg_yr_%', 'New_building_no.', 'Business_deregs_total', 'change_prev_dereg_yr_%', 'Including_task_no.']
month_wise_2023


# In[101]:


t2_1 = excel_data.parse(' T10u11_1')
columns = t2_1.iloc[1].fillna('') + ' ' + t2_1.iloc[2].fillna('') + ' ' + t2_1.iloc[3].fillna('')
columns = columns.str.strip().str.replace('\n', ' ').str.replace('  ', ' ')
t2_1_cleaned = t2_1.iloc[4:].copy()
t2_1_cleaned.columns = columns
t2_1_cleaned.reset_index(drop=True, inplace=True)


# In[102]:


t2_1_cleaned


# In[ ]:





# In[103]:


# Split the DataFrame into two parts
Business_reg_dereg = t2_1_cleaned.loc[1:8]  # Remove the first row with null values
Business_reg_dereg  = Business_reg_dereg .drop(Business_reg_dereg .index[-1])

Business_reg_dereg.columns = ['Districts','Business_regs_total', 'change_prev_reg_yr_%', 'New_building_no.', 'Business_deregs_total', 'change_prev_dereg_yr_%', 'Including_task_no.']



# In[104]:


Business_reg_dereg 


# In[105]:


Business_establi_closure= t2_1_cleaned.loc[19:31].reset_index(drop=True)
new_columns = Business_establi_closure.iloc[0] + '_' + Business_establi_closure.iloc[3] + '_' + Business_establi_closure.iloc[2].fillna('')
Business_establi_closure.columns = new_columns
# Remove the first three rows from the DataFrame
Business_establi_closure = Business_establi_closure.iloc[4:]

# Display the transformed DataFrame
Business_establi_closure


# In[106]:


Business_establi_closure.columns = ['Districts','New_companies_total', 'new_c_Business_start_ups', 'other_startups', 'complete_task_total', 'C_task_Business_start-ups', 'other_shutdowns']

# Remove rows where all values are NaN
Business_establi_closure= Business_establi_closure.dropna(how='all')
Business_establi_closure = Business_establi_closure.drop(Business_establi_closure.index[-1])


Business_establi_closure


# In[107]:


import sqlite3
import os
import pandas as pd
df = pd.DataFrame(year_wise)
db_file = 'year_wise.db'

# Create a connection to SQLite database
conn = sqlite3.connect(db_file)

# Create a cursor object to execute SQL commands
cursor = conn.cursor()

# Create Year_wise table
create_table_sql = """
CREATE TABLE IF NOT EXISTS Year_wise (
    Year INTEGER PRIMARY KEY,
    Business_regs_total INTEGER,
    change_prev_reg_yr REAL,
    New_building_no INTEGER,
    Business_deregs_total INTEGER,
    change_prev_dereg_yr REAL,
    Including_task_no INTEGER
);
"""
cursor.execute(create_table_sql)

# Insert data into Year_wise table
for index, row in df.iterrows():
    insert_sql = """
    INSERT INTO Year_wise (Year, Business_regs_total, change_prev_reg_yr, New_building_no, Business_deregs_total, change_prev_dereg_yr, Including_task_no)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """

# Commit changes and close connection
conn.commit()
conn.close()

print(f"DataFrame successfully inserted into '{db_file}' SQLite database.")


# In[108]:


year_wise


# In[ ]:





# In[109]:


month_wise_2023


# In[110]:


Business_reg_dereg 


# In[111]:


Business_establi_closure


# In[112]:


import sqlite3
import pandas as pd

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('hamburg_business_database.db')
cursor = conn.cursor()

# Create tables
cursor.execute('''CREATE TABLE IF NOT EXISTS year_wise(
                    Year INTEGER PRIMARY KEY,
                    Business_regs_total INTEGER,
                    change_prev_reg_yr REAL,
                    New_building_no INTEGER,
                    Business_deregs_total INTEGER,
                    change_prev_dereg_yr REAL,
                    Including_task_no INTEGER)''')
# Insert data into year_wise table
year_wise.to_sql('year_wise', conn, if_exists='replace', index=False)

cursor.execute('''CREATE TABLE IF NOT EXISTS month_wise_2023 (
                    Month TEXT PRIMARY KEY,
                    Business_regs_total INTEGER,
                    change_prev_reg_yr REAL,
                    New_building_no INTEGER,
                    Business_deregs_total INTEGER,
                    change_prev_dereg_yr REAL,
                    Including_task_no INTEGER)''')
# Insert data into month_wise_2023 table
month_wise_2023.to_sql('month_wise_2023', conn, if_exists='replace', index=False)

cursor.execute('''CREATE TABLE IF NOT EXISTS Business_reg_dereg  (
                    District TEXT PRIMARY KEY,
                    Business_regs_total INTEGER,
                    change_prev_reg_yr REAL,
                    New_building_no INTEGER,
                    Business_deregs_total INTEGER,
                    change_prev_dereg_yr REAL,
                    Including_task_no INTEGER)''')
# Insert data into Business_reg_dereg table
Business_reg_dereg.to_sql('Business_reg_dereg', conn, if_exists='replace', index=False)

cursor.execute('''CREATE TABLE IF NOT EXISTS Business_establi_closure (
                    District TEXT PRIMARY KEY,
                    New_companies_total INTEGER,
                    new_c_Business_start_ups INTEGER,
                    other_startups INTEGER,
                    complete_task_total INTEGER,
                    C_task_Business_start_ups INTEGER,
                    other_shutdowns INTEGER)''')

# Insert data into Business_establi_closure table
Business_establi_closure.to_sql('Business_establi_closure', conn, if_exists='replace', index=False)
conn.commit()
conn.close()


# In[ ]:





# In[113]:


def new_c_Business_start_ups(Business_establi_closure, starbucks_effect_range=(0.05, 0.118)):
    predictions = []
    for index, row in Business_establi_closure.iterrows():
        lower_bound_increase = row['new_c_Business_start_ups'] * starbucks_effect_range[0]
        upper_bound_increase = row['new_c_Business_start_ups'] * starbucks_effect_range[1]
        predictions.append({
            'District': row['Districts'],
            'Current_New_Companies': row['new_c_Business_start_ups'],
            'Predicted_Increase_Lower_Bound': lower_bound_increase,
            'Predicted_Increase_Upper_Bound': upper_bound_increase
        })
    return predictions

# Predict growth
new_c_Business_start_ups = new_c_Business_start_ups(Business_establi_closure)
new_c_Business_start_ups_df = pd.DataFrame(new_c_Business_start_ups)


# In[114]:


print(Business_establi_closure.columns)


# In[115]:


# Rename columns if necessary
Business_establi_closure.columns = ['Districts', 'New_companies_total', 'new_c_Business_start_ups', 'other_startups', 'complete_task_total', 'C_task_Business_start_ups', 'other_shutdowns']


# In[116]:


import matplotlib.pyplot as plt

# Plotting the predicted growth
plt.figure(figsize=(12, 6))
plt.bar(new_c_Business_start_ups_df['District'], new_c_Business_start_ups_df['Predicted_Increase_Lower_Bound'], color='lightblue', label='min.increase in new Business startups with a new Starbucks')
plt.bar(new_c_Business_start_ups_df['District'],new_c_Business_start_ups_df['Predicted_Increase_Upper_Bound'], color='blue', alpha=0.5, label='max.increase in new Business startups with a new Starbucks')
plt.xlabel('District')
plt.ylabel('Number of Startups')
plt.title('Predicted Increase in new_c_Business_start_ups_df per District with a New Starbucks')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[117]:


def other_startups(Business_establi_closure, starbucks_effect_range=(0.05, 0.118)):
    predictions = []
    for index, row in Business_establi_closure.iterrows():
        lower_bound_increase = row['other_startups'] * starbucks_effect_range[0]
        upper_bound_increase = row['other_startups'] * starbucks_effect_range[1]
        predictions.append({
            'District': row['Districts'],
            'Current_New_Companies': row['other_startups'],
            'Predicted_Increase_Lower_Bound': lower_bound_increase,
            'Predicted_Increase_Upper_Bound': upper_bound_increase
        })
    return predictions

# Predict growth
other_startups = other_startups(Business_establi_closure)
other_startups_df = pd.DataFrame(other_startups)


# In[118]:


# Plotting the predicted growth
plt.figure(figsize=(12, 6))
plt.bar(other_startups_df['District'], other_startups_df['Predicted_Increase_Lower_Bound'], color='Orange', label='min.increase in Other startups with a new Starbucks')
plt.bar(other_startups_df['District'], other_startups_df['Predicted_Increase_Upper_Bound'], color='yellow', alpha=0.5, label='max.increase in new other startups with a new Starbucks')
plt.xlabel('District')
plt.ylabel('Number of Startups')
plt.title('Predicted Increase in Other Startups per District with a New Starbucks')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[119]:


# Save the predicted growth data to CSV
new_c_Business_start_ups_df.to_csv('new_c_Business_start_ups.csv', index=False)
other_startups_df.to_csv('other_startups.csv', index=False)


# In[120]:


# Combine the DataFrames
combined_df = pd.merge(new_c_Business_start_ups_df, other_startups_df, on='District', suffixes=('_new_Business_startups', '_other_startups'))

# Save the combined DataFrame to a new CSV file
combined_df.to_csv('combined_predicted_growth.csv', index=False)


# # Starbucks Effect on Startups in Hamburg
# 
# ## Overview
# 
# This project analyzes the potential increase in the number of startups in various districts of Hamburg if a new Starbucks were to be introduced. The analysis is based on data related to business registrations, deregistrations, and company establishments.
# 
# ## Approach
# 
# 1. **Data Normalization**: 
#    - Data related to business registrations, deregistrations, and establishments were normalized and inserted into an SQLite database.
#    
# 2. **Prediction Model**:
#    - A function was defined to predict the growth in the number of startups based on a study indicating that a new Starbucks increases startups by 5-11.8%.
# 
# 3. **Data Visualization**:
#    - The predicted increase in startups was visualized using a bar chart.
# 
# ## Files
# 
# - `hamburg_business.db`: SQLite database containing normalized data.
# - `RAHUL_Hamburg.py`: Python script for data insertion, prediction, and visualization.
# - `combined_predicted_growth.csv`: CSV file containing the predicted growth of startups per district.
# - `startup_growth_plot.png`: Plot showing the predicted increase in startups per district.
# - `README.md`: This file.
# 
# ## Usage
# 
# To reproduce the analysis:
# 1. Ensure all dependencies are installed (e.g., `pandas`, `sqlite3`, `matplotlib`).
# 2. Run ``RAHUL_Hamburg.py`.
# 3. View the generated plot and `combined_predicted_growth.csv` for results.
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




