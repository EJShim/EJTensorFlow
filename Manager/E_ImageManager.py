import numpy as np
import matplotlib.pyplot as plt


def ShowMnistImage(image):
    result = np.zeros((28,28))
    idx = 0

    for i in range(28):
        for j in range(28):
            result[i][j] = image[idx]
            idx = idx + 1

    plt.matshow(result)
    plt.show()
