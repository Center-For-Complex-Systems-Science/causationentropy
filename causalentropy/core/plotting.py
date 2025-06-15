import numpy as np
import matplotlib.pyplot as plt
from stats import auc


def roc_curve(self, TPRs, FPRs):
    plt.plot(FPRs, TPRs)
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC')
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    AUC = auc(TPRs, FPRs)
    plt.text(0.4, 0.1, 'AUC = ' + format(AUC, '.4f'))