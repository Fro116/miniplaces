import os
import numpy as np
import scipy.misc
import h5py
from sklearn.cluster import KMeans
#np.random.seed(123)


train_file = h5py.File('../../data/miniplaces_128_train.h5', "r")
val_file = h5py.File('../../data/miniplaces_128_val.h5', "r")
test_file = h5py.File('../../data/miniplaces_128_test.h5', "r")
        
# loading data from .h5
class DataLoaderH5(object):
    def __init__(self, **kwargs):
        self.load_size = int(kwargs['load_size'])
        self.fine_size = int(kwargs['fine_size'])
        self.data_mean = np.array(kwargs['data_mean'])
        self.phase = kwargs['phase']

        if self.phase == 'training':
            self.list_im = train_file['images']
            self.list_lab = train_file['labels']
        elif self.phase == 'validation':
            self.list_im = val_file['images']
            self.list_lab = val_file['labels']
        else:
            self.list_im = test_file['images']
            self.list_lab = test_file['labels']            
        self.num = len(self.list_im)
        self.perm = [x for x in range(self.num)]
        print('# Images found:', self.num)

        if self.phase == 'training' or self.phase == 'validation':
            self.shuffle()
            
        self._idx = 0

    def next_batch(self, batch_size):
        if self.phase == 'training':
            crop_width = np.random.random_integers(int(self.fine_size*3/4), self.fine_size)
            crop_height = np.random.random_integers(int(self.fine_size*3/4), self.fine_size)
        else:
            crop_width = self.fine_size
            crop_height = self.fine_size

        labels_batch = np.zeros(batch_size)
        images_batch = np.zeros((batch_size, crop_height, crop_width, 3))
        for i in range(batch_size):
            index = self.perm[self._idx]
            image = np.array(self.list_im[index])
            image = image.astype(np.float32)/255. - self.data_mean
#            gray = np.dot(image[...,:3], [0.299, 0.587, 0.114])
#            image[:, :, 0] = gray
#            image[:, :, 1] = gray
#            image[:, :, 2] = gray                                
#            image = image[:, ::-1, :]
            if self.phase == 'training':
                flip = np.random.random_integers(0, 1)
                if flip>0:
                    image = image[:,::-1,:]
                gray = np.random.random_integers(0, 1)
                if gray>0:
                    gray = np.dot(image[...,:3], [0.299, 0.587, 0.114])
                    image[:, :, 0] = gray
                    image[:, :, 1] = gray
                    image[:, :, 2] = gray
                offset_h = np.random.random_integers(0, self.load_size-crop_height)
                offset_w = np.random.random_integers(0, self.load_size-crop_width)
            else:
                offset_h = (self.load_size-crop_height)//2
                offset_w = (self.load_size-crop_width)//2                
            image = image[offset_h:offset_h+crop_height, offset_w:offset_w+crop_width, :]
            images_batch[i, ...] = image
            labels_batch[i, ...] = self.list_lab[index]
            self._idx += 1
            if self._idx == self.num:
                self._idx = 0
                if self.phase == 'training' or self.phase == 'validation':
                    self.shuffle()            

        return images_batch, labels_batch

    def size(self):
        return self.num

    def reset(self):
        self._idx = 0

    def shuffle(self):
        self.perm = np.random.permutation(self.num)
