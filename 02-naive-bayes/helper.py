import pickle
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os.path
from sklearn.naive_bayes import MultinomialNB
import pandas as pd


MNIST_FILE = 'mnist.p'
OUTPUT_FILE = 'accs.csv'
H = 28
W = 28


def img_grid(imgs, n_rows):
    # reshape to 28*28 pics
    imgs = imgs.reshape(-1, H, W)
    # reshape to a grid
    n_columns = len(imgs) // n_rows
    imgs = np.array(imgs).reshape(n_rows, n_columns, H, W)
    imgs = imgs.swapaxes(1, 2).reshape(n_rows * H, n_columns * W)

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(imgs, cmap="gray")
    ax.axis('scaled')
    plt.show()


def getExpandido(x, w, h):
    """Funcion para agrupar píxeles por grupos.
    Debe tomar un conjunto X de N matrices de HxW, y devuelve un nuevo conjunto
     X_expandido de N*width*height matrices de HxW según lo visto en la presentación
    Pasos:
    1. Si X es unidimensional, reshapearlo a matrices
    2. Muestrear cada matriz para generar width*height matrices de menor dimensión (H/height x W/width)
    3. Concatenar estas matrices
    4. Repetir cada pixel para llevar las matrices a la dimension original HxW

    Funciones útiles: -samplear vectores por paso
                      -np.concatenate()
                      -np.repeat()
    """
    x = x.reshape(-1, H // h, h, W // w, w)  # split each img into h*w groups
    x = x.swapaxes(1, 2).swapaxes(2, 4).swapaxes(3, 4)  # split groups
    x = x.reshape(-1, H // h, W // w)  # split each img into h*w images
    x = x.repeat(h, axis=1).repeat(w, axis=2)  # resize images
    return x.reshape(-1, H * W)


def getNewGrayLevels(x, bins):
    """
    Función que transforme los valores de gris de x
    para llevarlos a una nueva escala con len(bins) valores de gris,
    donde cada elemento de bins es un valor de la escala.

    """
    delta = 255 / (bins - 1)  # new minimum diff between two values
    x = x.astype(float) + delta / 2  # ensure that the result of // be rounded instead of truncated
    return np.round((x // delta) * delta).astype(np.uint8)  # round x to multiples of delta


if __name__ == '__main__':
    if 1:  # os.path.isfile(OUTPUT_FILE):
        if os.path.isfile(MNIST_FILE):
            X, y = pickle.load(open(MNIST_FILE, 'rb'))
            X = X.astype(np.uint8)
            y = y.astype(np.int8)
        else:
            X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True)
            X = X.astype(np.uint8)
            y = y.astype(np.int8)
            pickle.dump((X, y), open(MNIST_FILE, 'wb'))

        imgs = X.reshape(-1, 28, 28)

        ratios = (5, 1, 1)  # train / validation / test

        # get test set
        X_train_valid, X_test, y_train_valid, y_test = train_test_split(
            X, y, random_state=42, stratify=y,
            test_size=ratios[2] / sum(ratios)
        )

        # split remainder between train and validation
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_valid, y_train_valid, random_state=42, stratify=y_train_valid,
            test_size=ratios[1] / sum(ratios[:2])
        )

        grey_levels = (10, 64, 128, 255)
        alphas = (0.1, 0.01, 0.001, 0.0001)
        group_sizes = tuple(np.array(size) for size in ([1, 1], [2, 2], [2, 1], [4, 4]))

        print(np.all(getNewGrayLevels(X_train, 256)))
        # accuracies = []
        #
        # for size in group_sizes:
        #     x_exp = getExpandido(X_train, *size)
        #     y = y_train.repeat(size[0] * size[1])
        #
        #     for grey in grey_levels:
        #         x = getNewGrayLevels(x_exp, grey)
        #
        #         for alpha in alphas:
        #             print(f'Computing: S={size}, G={grey}, A={alpha}...')
        #
        #             clf = MultinomialNB(alpha=alpha)
        #             clf.fit(x, y)
        #
        #             accuracy = clf.score(X_valid, y_valid)
        #             accuracies.append({
        #                 'Grey levels': grey,
        #                 'Alpha': alpha,
        #                 'Pixel grouping': size,
        #                 'Accuracy': accuracy
        #             })
        #
        #             print('Accuracy: ', accuracy)
        #             print(2 * '-------------------------------------')
        #
        # accuracies = pd.DataFrame(accuracies)
        # accuracies.to_csv(OUTPUT_FILE)
    else:
        results = pd.read_csv(OUTPUT_FILE)
        acc = results['Accuracy'].max()
        print(results[results['Accuracy'] == acc])
