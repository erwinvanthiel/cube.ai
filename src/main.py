import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from cube_environment import CubeEnvironment, Axis

cube = CubeEnvironment(3)
print(cube.get_reward())
cube.turn_slice(Axis.X, 2)
cube.turn_slice(Axis.X, 2)
cube.turn_slice(Axis.X, 2)
cube.turn_slice(Axis.X, 1)
print(cube.get_reward())



# List of cube faces
matrices = cube.get_faces()

# Define the custom colors
colors = ['white', 'blue', 'red', 'orange', 'green', 'yellow']
faces = ['TOP', 'BACK', 'RIGHT', 'LEFT', 'FRONT', 'BOTTOM']
cmap = ListedColormap(colors)

# Create a figure and a grid of subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Flatten the array of axes for easy iteration
axs = axs.flatten()

for i, ax in enumerate(axs):
    # Display each matrix as a heatmap using seaborn
    sns.heatmap(matrices[i], ax=ax, cmap=cmap, vmin=0, vmax=5, cbar=False, square=True)
    ax.set_title(faces[i])

    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()