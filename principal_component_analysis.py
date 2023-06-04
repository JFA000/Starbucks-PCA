import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.lines import Line2D

def perform_pca(X, num_components=2):
    cov_matrix = np.cov(X, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    largest_eigenvectors = eigenvectors[:, sorted_indices[:num_components]]
    projected_data = X @ largest_eigenvectors
    return projected_data, largest_eigenvectors

data = pd.read_csv('starbucks_nutrition_facts.csv', encoding='utf-8', delimiter=',')
selected_columns = data[['Calories', 'Fat(g)', 'Carb.(g)', 'Fiber(g)', 'Protein', 'Sodium']]
X = selected_columns.values
projected_data, principal_components = perform_pca(X, num_components=3)

# Define colors based on the values in the axes
min_vals = np.min(projected_data, axis=0)
max_vals = np.max(projected_data, axis=0)
cmap = cm.get_cmap('coolwarm')  # Choose a colormap with contrasting colors
normalized_axes = (projected_data - min_vals) / (max_vals - min_vals)
colors = cmap(normalized_axes[:, 0])  # Use the values of the first axis to determine colors

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(projected_data[:, 1], projected_data[:, 0], projected_data[:, 2], c=colors, alpha=0.5, s=50)

ax.set_xlabel(selected_columns.columns[1])
ax.set_ylabel(selected_columns.columns[0])
ax.set_zlabel(selected_columns.columns[2])
ax.set_title('Análise de Componentes Principais')

# Definir rótulos para a legenda
legend_labels = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10)
                 for color in colors]

# Adicionar a legenda
ax.legend(legend_labels, [selected_columns.columns[1], selected_columns.columns[0], selected_columns.columns[2]], loc='upper left')

# Ajustar a escala dos eixos
ax.auto_scale_xyz(projected_data[:, 1], projected_data[:, 0], projected_data[:, 2])

# Ajustar os ângulos de visualização
ax.view_init(elev=30, azim=45)

# Adicionar grid
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# Ajustar espaçamento entre subplots
plt.tight_layout()

plt.show()
