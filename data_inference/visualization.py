"""Module for Visualization"""
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

sns.set_theme()

def visualize_ax(t_train, y_train, t_test_true, y_test_true, t_test, result, ax, size = 4):
    """
    Visualize the Gaussian Process Prediction
    """
    if size > 0:
        samples = np.random.multivariate_normal(result[0], result[1], size)
        ax.plot(t_test, samples[0], "-", color='blue', alpha=0.5, label = "Posterior sample function")
        for el in samples[1:]:
            ax.plot(t_test, el, "-", color='blue', alpha=0.5)


    ax.fill_between(t_test, result[0] - np.sqrt(np.diag(result[1])), result[0] + np.sqrt(np.diag(result[1])), color='blue', alpha=0.2, label = "1 Std")
    ax.fill_between(t_test, result[0] - 2*np.sqrt(np.diag(result[1])), result[0] + 2*np.sqrt(np.diag(result[1])), color='blue', alpha=0.1, label = "2 Std")

    #sns.lineplot(x = t_test, y = result[0], color = sns.color_palette()[0], label = "Prediction")
    ax.plot(t_test, result[0], "-", color = "red", label = "Mean Prediction", linewidth=1)
    ax.plot(t_train, y_train, ".", color = "red", label = "Observation")
    ax.plot(t_test_true, y_test_true, ".", color = "blue", label = "Ground Truth")
    ax.set_xlabel("Time (Day)")
    ax.legend()
    return ax
