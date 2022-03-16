# -*- coding: utf-8 -*-
"""
@author: NelsonRCM
"""

import matplotlib.pyplot as plt
import seaborn as sns


def pred_scatter_plot(real_values, pred_values, title, xlabel, ylabel, savefig, figure_name):
    fig, ax = plt.subplots()
    ax.scatter(real_values, pred_values, c='dodgerblue',
               edgecolors='black')
    # ax.plot([real_values.min(),real_values.max()],[real_values.min(),real_values.max()],'k--',lw = 4)
    ax.plot(real_values, real_values, 'k--', lw=4)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    plt.title(title)
    plt.xlim([4, 11])
    plt.ylim([4, 11])
    plt.show()
    if savefig:
        fig.savefig(figure_name, dpi=1200, format='eps')
