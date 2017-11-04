import os
import numpy as np
import scipy.misc
import h5py
#np.random.seed(123)


train_file = h5py.File('../../data/miniplaces_128_train.h5', "r")
val_file = h5py.File('../../data/miniplaces_128_val.h5', "r")
        
# loading data from .h5
class DataLoaderH5(object):
    def __init__(self, **kwargs):
        self.load_size = int(kwargs['load_size'])
        self.fine_size = int(kwargs['fine_size'])
        self.data_mean = np.array(kwargs['data_mean'])
        self.phase = kwargs['phase']
        first_label = int(kwargs['first_label'])
        second_label = int(kwargs['second_label'])

        if self.phase == 'training':
            images_per_label = 1000
            self.list_im = np.concatenate((np.array(train_file['images'][images_per_label * first_label : images_per_label * (first_label+1)]),
                                          np.array(train_file['images'][images_per_label * second_label : images_per_label * (second_label+1)])))
            self.list_lab = [0 for x in range(images_per_label)] + [1 for x in range(images_per_label)]
        elif self.phase == 'validation':
            f = val_file
            im_set = np.array(f['images'])
            lab_set = np.array(f['labels'])        
            self.list_im = []
            self.list_lab = []
            for i in range(len(im_set)):
                lab = lab_set[i]
                if self.phase == 'validation':
                    if lab == first_label:
                        self.list_im.append(im_set[i])
                        self.list_lab.append(0)
                    elif lab == second_label:
                        self.list_im.append(im_set[i])
                        self.list_lab.append(1)
                else:
                    self.list_im.append(im_set[i])
                    self.list_lab.append(-1)
        self.num = len(self.list_im)
        self.perm = [range(self.num)]
        print('# Images found:', self.num)

        if self.phase == 'training' or self.phase == 'validation':
            self.shuffle()
            
        self._idx = 0

    def next_batch(self, batch_size):
        labels_batch = np.zeros(batch_size)
        images_batch = np.zeros((batch_size, self.fine_size, self.fine_size, 3))

        for i in range(batch_size):
            index = self.perm[self._idx]
            image = self.list_im[index]
            image = image.astype(np.float32)/255. - self.data_mean
            if self.phase == 'training':
                flip = np.random.random_integers(0, 1)
                if flip>0:
                    image = image[:,::-1,:]
                offset_h = np.random.random_integers(0, self.load_size-self.fine_size)
                offset_w = np.random.random_integers(0, self.load_size-self.fine_size)
            else:
                offset_h = (self.load_size-self.fine_size)//2
                offset_w = (self.load_size-self.fine_size)//2

            images_batch[i, ...] = image[offset_h:offset_h+self.fine_size, offset_w:offset_w+self.fine_size, :]
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

# Loading data from disk
class DataLoaderDisk(object):
    def __init__(self, **kwargs):
        self.load_size = int(kwargs['load_size'])
        self.fine_size = int(kwargs['fine_size'])
        self.data_mean = np.array(kwargs['data_mean'])
        self.phase = kwargs['phase']
        self.data_root = os.path.join(kwargs['data_root'])
        first_label = int(kwargs['first_label'])
        second_label = int(kwargs['second_label'])

        # read data info from lists
        self.list_im = []
        self.list_lab = []
        with open(kwargs['data_list'], 'r') as f:
            for line in f:
                path, lab =line.rstrip().split(' ')
                path = os.path.join(self.data_root, path)
                lab = int(lab)
                if self.phase == 'training' or self.phase == 'validation':
                    if lab == first_label:
                        self.list_im.append(path)
                        self.list_lab.append(0)
                    elif lab == second_label:
                        self.list_im.append(path)
                        self.list_lab.append(1)
                else:
                    self.list_im.append(path)
                    self.list_lab.append(-1)
                    
        self.list_im = np.array(self.list_im, np.object)
        self.list_lab = np.array(self.list_lab, np.int64)
        self.num = self.list_im.shape[0]
        print('# Images found:', self.num)

        if self.phase == 'training' or self.phase == 'validation':
            self.shuffle()
            
        self._idx = 0
        
    def next_batch(self, batch_size):
        images_batch = np.zeros((batch_size, self.fine_size, self.fine_size, 3)) 
        labels_batch = np.zeros(batch_size)
        for i in range(batch_size):
            image = scipy.misc.imread(self.list_im[self._idx])
            image = image.astype(np.float32)/255.
            image = image - self.data_mean
            if self.phase == 'training':
                flip = np.random.random_integers(0, 1)
                if flip>0:
                    image = image[:,::-1,:]
                offset_h = np.random.random_integers(0, self.load_size-self.fine_size)
                offset_w = np.random.random_integers(0, self.load_size-self.fine_size)
            else:
                offset_h = (self.load_size-self.fine_size)//2
                offset_w = (self.load_size-self.fine_size)//2

            images_batch[i, ...] =  image[offset_h:offset_h+self.fine_size, offset_w:offset_w+self.fine_size, :]
            label = self.list_lab[self._idx]
            labels_batch[i, ...] = label
            
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
        perm = np.random.permutation(self.num)
        self.list_im[:, ...] = self.list_im[perm, ...]
        self.list_lab[:] = self.list_lab[perm, ...]

    def data_mean(data_root, data_list, load_size):
        mean = np.zeros((load_size, load_size, 3))
        num_images = 0
        with open(data_list, 'r') as f:
            for line in f:
                path, lab =line.rstrip().split(' ')
                im_path = os.path.join(data_root, path)
                image = scipy.misc.imread(im_path)
                image = image.astype(np.float32)/255.
                mean += image
                num_images += 1
                print(num_images)
        return mean / num_images
