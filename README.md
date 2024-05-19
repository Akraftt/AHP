# load the csv file that is downloaded from google forms.
file_path = 'FormAnswers.csv'
data = pd.read_csv(file_path)
n_criteria = 5

def create_pairwise_matrix(answers):
    matrix = np.ones((n_criteria, n_criteria))
    try:
        matrix[0, 1] = answers[0]  # Cost vs Compatibility
        matrix[1, 0] = 1 / answers[0]
        
        matrix[1, 2] = answers[1]  # Compatibility vs User Acceptance
        matrix[2, 1] = 1 / answers[1]
        
        matrix[2, 3] = answers[2]  # User Acceptance vs Management
        matrix[3, 2] = 1 / answers[2]
        
        matrix[3, 4] = answers[3]  # Management vs Conditional Access
        matrix[4, 3] = 1 / answers[3]
        
        matrix[0, 2] = answers[4]  # Cost vs User Acceptance
        matrix[2, 0] = 1 / answers[4]
        
        matrix[0, 3] = answers[5]  # Cost vs Management
        matrix[3, 0] = 1 / answers[5]
        
        matrix[0, 4] = answers[6]  # Cost vs Conditional Access
        matrix[4, 0] = 1 / answers[6]
        
        matrix[1, 3] = answers[7]  # Compatibility vs Management
        matrix[3, 1] = 1 / answers[7]
        
        matrix[1, 4] = answers[8]  # Compatibility vs Conditional Access
        matrix[4, 1] = 1 / answers[8]
        
        matrix[2, 4] = answers[9]  # User Acceptance vs Conditional Access
        matrix[4, 2] = 1 / answers[9]
    except ZeroDivisionError:
        raise ValueError("Zero Divison Error!.")
    return matrix

#Consistency Ratio (C.R.)
def calculate_consistency_ratio(matrix):
    eigvals, eigvecs = np.linalg.eig(matrix)
    max_eigval = np.max(eigvals).real
    CI = (max_eigval - n_criteria) / (n_criteria - 1)
    RI = 1.12  # Random Index 5 = 1.12, update this accordingly.
    CR = CI / RI
    return CR

numerical_data = data.iloc[:, -10:]
numerical_data = numerical_data.apply(pd.to_numeric, errors='coerce')
numerical_data = numerical_data.dropna()
consistency_ratios = []

for idx, row in numerical_data.iterrows():
    answers = row.values
    try:
        pairwise_matrix = create_pairwise_matrix(answers)
        CR = calculate_consistency_ratio(pairwise_matrix)
        consistency_ratios.append(CR)
    except ValueError as e:
        print(f"could not process row: error. {idx}: {e}")
        consistency_ratios.append(np.nan)

# Add C.R. to the google forms csv
data['Consistency Ratio'] = consistency_ratios

#new csv file, same structure as the 
output_file_path = 'FormAnswers_with_CR.csv'
data.to_csv(output_file_path, index=False)

print(f"Consistency Ratios have been calculated and saved to {output_file_path}")





# this code block has some errors, the calculating are correct, however the printing of numbes isnt ideal and should be avoided
# 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'FormAnswers_with_CR.csv'
data = pd.read_csv(file_path)

cr_values = data['Consistency Ratio'].dropna()

mean_cr = np.mean(cr_values)
std_cr = np.std(cr_values)

within_2_std = cr_values[(cr_values >= mean_cr - 2 * std_cr) & (cr_values <= mean_cr + 2 * std_cr)]
outside_2_std = cr_values[(cr_values < mean_cr - 2 * std_cr) | (cr_values > mean_cr + 2 * std_cr)]

print("Consistency Ratios within 2 standard deviations:")
print(within_2_std)

print("\nConsistency Ratios outside 2 standard deviations:")
print(outside_2_std)

plt.figure(figsize=(10, 6))
sns.histplot(cr_values, kde=True, color='green', bins=10)
plt.axvline(mean_cr - 2 * std_cr, color='red', linestyle='dashed', linewidth=1)
plt.axvline(mean_cr + 2 * std_cr, color='red', linestyle='dashed', linewidth=1)
plt.title('Respondents\' CR Values Normal Distribution Curve')
plt.xlabel('CR Values')
plt.ylabel('Density')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Manually enter C.R values in the np array, provides a much easier result to read.
cr_values = np.array([
    0.132414, 0.119932, 0.189173, 0.185251, 0.188793, 0.267255, 
    0.469368, 0.400274, 0.188231, -1.98e-16, 0.201415, 0.151454, 
    0.218951, 0.141289, 0.077303, 0.18286, 0.22204
])

mean_cr = np.mean(cr_values)
std_cr = np.std(cr_values)

within_2_std = cr_values[(cr_values >= mean_cr - 2 * std_cr) & (cr_values <= mean_cr + 2 * std_cr)]
outside_2_std = cr_values[(cr_values < mean_cr - 2 * std_cr) | (cr_values > mean_cr + 2 * std_cr)]

within_2_std_list = within_2_std.tolist()
outside_2_std_list = outside_2_std.tolist()

print("Consistency Ratios within 2 standard deviations:")
for cr in within_2_std_list:
    print(f"{cr:.6f}")

print("\nConsistency Ratios outside 2 standard deviations:")
for cr in outside_2_std_list:
    print(f"{cr:.6f}")

plt.figure(figsize=(10, 6))
sns.histplot(cr_values, kde=True, color='green', bins=10)
plt.axvline(mean_cr - 2 * std_cr, color='red', linestyle='dashed', linewidth=1)
plt.axvline(mean_cr + 2 * std_cr, color='red', linestyle='dashed', linewidth=1)
plt.title('Respondents\' CR Values Normal Distribution Curve')
plt.xlabel('CR Values')
plt.ylabel('Density')
plt.show()



import pandas as pd
import numpy as np

file_path = 'Updated.csv'
data = pd.read_csv(file_path)
n_criteria = 5

def create_pairwise_matrix(answers):
    matrix = np.ones((n_criteria, n_criteria))
    
    # Fill in the matrix with the provided answers
    matrix[0, 1] = answers[0]  # Cost vs Compatibility
    matrix[1, 0] = 1 / answers[0]
    
    matrix[1, 2] = answers[1]  # Compatibility vs User Acceptance
    matrix[2, 1] = 1 / answers[1]
    
    matrix[2, 3] = answers[2]  # User Acceptance vs Management
    matrix[3, 2] = 1 / answers[2]
    
    matrix[3, 4] = answers[3]  # Management vs Conditional Access
    matrix[4, 3] = 1 / answers[3]
    
    matrix[0, 2] = answers[4]  # Cost vs User Acceptance
    matrix[2, 0] = 1 / answers[4]
    
    matrix[0, 3] = answers[5]  # Cost vs Management
    matrix[3, 0] = 1 / answers[5]
    
    matrix[0, 4] = answers[6]  # Cost vs Conditional Access
    matrix[4, 0] = 1 / answers[6]
    
    matrix[1, 3] = answers[7]  # Compatibility vs Management
    matrix[3, 1] = 1 / answers[7]
    
    matrix[1, 4] = answers[8]  # Compatibility vs Conditional Access
    matrix[4, 1] = 1 / answers[8]
    
    matrix[2, 4] = answers[9]  # User Acceptance vs Conditional Access
    matrix[4, 2] = 1 / answers[9]
    return matrix

def calculate_weights(matrix):
    eigvals, eigvecs = np.linalg.eig(matrix)
    max_eigval_index = np.argmax(eigvals)
    eigvec = eigvecs[:, max_eigval_index].real
    weights = eigvec / np.sum(eigvec)
    return weights

aggregated_matrix = np.zeros((n_criteria, n_criteria))
numeric_data = data.iloc[:, 1:11].astype(float)

for idx, row in numeric_data.iterrows():
    answers = row.values
    pairwise_matrix = create_pairwise_matrix(answers)
    aggregated_matrix += pairwise_matrix

aggregated_matrix /= len(numeric_data)
weights = calculate_weights(aggregated_matrix)
criteria = ["Cost", "Compatibility", "User Acceptance", "Management", "Conditional Access"]
weights_dict = {crit: weight for crit, weight in zip(criteria, weights)}

for crit, weight in weights_dict.items():
    print(f"{crit}: {weight:.4f}")

print("Aggregated Matrix:\n", aggregated_matrix)
print("Weights:", weights)
