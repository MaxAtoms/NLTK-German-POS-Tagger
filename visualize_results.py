import pandas as pd
import matplotlib.pyplot as plt
import re

# Read the CSV file
df = pd.read_csv('results.csv', header=None, names=['Model', 'Accuracy', 'Correct', 'Total', 'Description'])

# Filter out rows where 'Accuracy' is non-numeric (e.g., if there are strings or headers in the data)
df = df[pd.to_numeric(df['Accuracy'], errors='coerce').notna()]

# Convert 'Accuracy' to a float, multiply by 100 for percentage, and round to 2 decimal places
df['Accuracy'] = df['Accuracy'].astype(float) * 100
df['Accuracy'] = df['Accuracy'].round(2)

# Extract 'Iteration' from 'Description' and add 'Tagdict' status
df['Iteration'] = df['Description'].str.extract(r'(\d+) iteration').astype(float)
df['Tagdict'] = df['Description'].apply(lambda x: 'With Tagdict' if 'Tagdict True' in x else 'Without Tagdict')

# Filter out rows where 'Iteration' is missing
df = df.dropna(subset=['Iteration'])

# Define datasets for plotting with corrected filtering expressions
datasets = {
    'TIGER corpus with 90/10 split (with tagdict)': df[(df['Model'] == 'tiger_tagger') & (df['Tagdict'] == 'With Tagdict')],
    'TIGER corpus with 90/10 split (without tagdict)': df[(df['Model'] == 'tiger_tagger') & (df['Tagdict'] == 'Without Tagdict')],
    'TIGER tagger on novelette.conll': df[df['Description'].str.contains(r'novelette', case=False, na=False)],
    'TIGER tagger on ted.conll': df[df['Description'].str.contains(r'ted', case=False, na=False)],
    'TIGER tagger on sermononline.conll': df[df['Description'].str.contains(r'sermononline', case=False, na=False)],
    'TIGER tagger on wikipedia.conll': df[df['Description'].str.contains(r'wikipedia', case=False, na=False)],
    'TIGER tagger on opensubtitles.conll': df[df['Description'].str.contains(r'opensubtitles', case=False, na=False)]
}

# Debugging output to verify dataset content before plotting
for label, data in datasets.items():
    print(f"{label} data points:\n{data[['Iteration', 'Accuracy']]}")  # Check if each dataset has data

# Plot each dataset with a unique color and style
plt.figure(figsize=(14, 10))
for label, data in datasets.items():
    if not data.empty:  # Plot only if data exists for the dataset
        plt.plot(data['Iteration'], data['Accuracy'], marker='o', label=label)

# Configure plot labels and limits
plt.title("Evaluation of the German Tagger")
plt.xlabel("Iterations")
plt.ylabel("Accuracy (%)")
plt.legend(loc="lower right")
plt.grid(True)

# Define x-axis and y-axis tick marks
plt.xticks([1, 5, 10, 15])
plt.yticks(range(60, 101, 5))  # Display y-axis from 60 to 100 in steps of 5
plt.ylim(60, 100)  # Set y-axis limits to show from 60% to 100%

plt.legend(loc="center right")  # Place the legend above the plot area

# Display the plot
plt.tight_layout()
plt.show()
