import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read data from CSV file
data = pd.read_csv('data.csv', encoding='utf-8',delimiter=',')

# Convert the data to a NumPy array

print("Data columns: ",data.columns)
#print(data['PasCmp'])
selected_columns = data[['Calories','Fat(g)','Carb.(g)','Fiber(g)','Protein','Sodium']]
print(selected_columns)
X = selected_columns.to_numpy()

# Calculate the covariance matrix
cov_matrix = np.cov(X, rowvar=False)
print('---- Covariance Matrix ------')
print(cov_matrix)
# Find the eigenvalues and eigenvectors of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
eigenvalues1, eigenvectors1 = np.linalg.eig(cov_matrix)
print('----- Eigenvalues ------')
print(eigenvalues1)
print('----- Eigenvectors ------')
print(eigenvectors1)
# Sort the eigenvalues in descending order and get the corresponding indices
print('----- Biggest Eigenvalues ------')
sorted_indices = np.argsort(eigenvalues)[::-1]
print(eigenvalues[sorted_indices[:2]])

largest_eigenvectors = eigenvectors[:, sorted_indices[:2]]

projected_data = X @ largest_eigenvectors

# Plot the 2D projected data
plt.scatter(projected_data[:, 0], projected_data[:, 1], alpha=0.5)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA with 2 Principal Components')

# Save the plot as a jpg image
plt.savefig('pca_plot1.jpg', dpi=300, bbox_inches='tight')

# Show the plot
#plt.show()
