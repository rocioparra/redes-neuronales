import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict


label_dict = {
    i: article for i, article in enumerate(
    ['T-shirt/top', 
     'Trouser',
     'Pullover',
     'Dress',
     'Coat',
     'Sandal',
     'Shirt',
     'Sneaker',
     'Bag',
     'Ankle boot']
    )
}


def img_grid(imgs, n_rows):
    # reshape to a grid
    n_columns = len(imgs)//n_rows
    imgs = np.array(imgs).reshape((n_rows, n_columns, 28, 28))
    imgs = imgs.swapaxes(1, 2).reshape((n_rows*28, n_columns*28))
    
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(20,15))
    ax.imshow(imgs, cmap="gray")
    ax.axis('scaled')
    plt.show()


class Article:
    """
    Article object: name + all associated images & a priori prob
    article.
    article.imgs -> all images, access order [img][row][col]
    article.from_pxls -> all images, access order [row][col][img]
    article.priori -> a priori probability
    """
    def __init__(self, name, index, pics, priori):
        self.name = str(name)
        self.index = index
        self.priori = priori        

        self.imgs = np.array(pics)  
        self.from_pxls = self.imgs.swapaxes(0, 1).swapaxes(1, 2) 
    
    def __len__(self):
        return len(self.imgs)
    
    def __str__(self):
        return self.name


def clothes_dict(names, pics):
    """
        create dictionary where
        - key: name of article (dress, bag...)
        - value: Article object
    """
    clothes = defaultdict(list)
    
    # classify images by article
    for i, article in enumerate(names):
        clothes[label_dict[article]].append(pics[i])
    
    # replace image lists for article objects, and calculate a priori probs
    for i, name in label_dict.items():
        clothes[name] = Article(name=name, index=i, pics=clothes[name], 
                                priori=len(clothes[name])/len(pics))
    
    return clothes