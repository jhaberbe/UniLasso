import seaborn as sns
import matplotlib.pyplot as plt

def plot_coefficient_vector(coef, ax=None, title="Coefficient Vector Heatmap", figsize=(10, 1)):
    """
    Plot the coefficient vector as a heatmap.
    
    Parameters:
    - coef: Coefficient vector to plot.
    - ax: Matplotlib axis to plot on. If None, creates a new figure.
    - title: Title string. Defaults to 'Coefficient Vector Heatmap'.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(coef.reshape(1, -1), ax=ax, cbar=False, xticklabels=False, yticklabels=False, cmap="bwr")
    ax.set_title(title)
    plt.show()