import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib import cm, lines

def perform_pca(X, num_components=2):
    cov_matrix = np.cov(X, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    largest_eigenvectors = eigenvectors[:, sorted_indices[:num_components]]
    projected_data = X @ largest_eigenvectors
    return projected_data, largest_eigenvectors

data = pd.read_csv('starbucks_nutrition_facts.csv', encoding='utf-8', delimiter=',')
X = data[['Calories', 'Fat(g)', 'Carb.(g)', 'Fiber(g)', 'Protein', 'Sodium']].values
projected_data, principal_components = perform_pca(X, num_components=3)

min_vals, max_vals = np.min(projected_data, axis=0), np.max(projected_data, axis=0)
cmap = cm.get_cmap('coolwarm')
colors = cmap((projected_data - min_vals) / (max_vals - min_vals))[:, 0]

fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': '3d'})
scatter = ax.scatter(projected_data[:, 1], projected_data[:, 0], projected_data[:, 2], c=colors, alpha=0.5, s=50)

ax.set(xlabel='Fat(g)', ylabel='Calories', zlabel='Carb.(g)', title='Análise de Componentes Principais')

legend_labels = [lines.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10)
                 for color in colors]
ax.legend(legend_labels, ['Fat(g)', 'Calories', 'Carb.(g)'], loc='upper left')

ax.auto_scale_xyz(projected_data[:, 1], projected_data[:, 0], projected_data[:, 2])
ax.view_init(elev=30, azim=45)
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

plt.tight_layout()
plt.show()
