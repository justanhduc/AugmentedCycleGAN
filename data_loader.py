import numpy as np
import imageio
import os
import neuralnet as nn
from random import shuffle
from skimage import transform

path = 'D:/1_Share/edges2shoes'
image_size = 64


def split_image(img):
    _, w, _ = img.shape
    return img[:, :w//2], img[:, w//2:]


class Edges2Shoes(nn.DataManager):
    def __init__(self, placeholders, bs, n_epochs, type='train', shuffle=False, **kwargs):
        super(Edges2Shoes, self).__init__(placeholders=placeholders, batch_size=bs, n_epochs=n_epochs, shuffle=shuffle,
                                          **kwargs)
        self.type = type
        self.load_data()

    def load_data(self):
        self.path = os.path.join(path, self.type)
        image_list = os.listdir(self.path)
        image_list = image_list[:self.kwargs.get('num_data', len(image_list))]
        image_list2 = list(image_list)
        shuffle(image_list2)

        self.dataset = (image_list, image_list2)
        self.data_size = len(image_list)

    def generator(self):
        if shuffle:
            shuffle(self.dataset[0])
            shuffle(self.dataset[1])

        for i in range(0, len(self), self.batch_size):
            edges = np.array([split_image(imageio.imread(os.path.join(self.path, file)))[0]
                              for file in self.dataset[0][i:i + self.batch_size]], 'float32')
            shoes = np.array([split_image(imageio.imread(os.path.join(self.path, file)))[1]
                              for file in self.dataset[1][i:i + self.batch_size]], 'float32')
            yield edges, shoes


if __name__ == '__main__':
    train_data = Edges2Shoes(None, 100, 20, shuffle=True)
    for res in train_data:
        print(res[1][0].shape[1], res[1][1].shape[1])

