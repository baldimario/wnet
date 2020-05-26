import numpy as np
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, samples, window=2,
                 to_fit=True, batch_size=32,
                 n_channels=1, shuffle=False, conv=False, single_output=False, raw_input=False, use_ulaw=True):
        self.samples = samples
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.window = window
        self.conv = conv
        self.single_output = single_output
        self.raw_input = raw_input
        self.use_mulaw = use_ulaw
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [k for k in indexes]

        # Generate data
        X = self._generate_X(list_IDs_temp)

        if(self.conv):
            X = np.expand_dims(X, axis=2)

        if self.to_fit:
            y = self._generate_y(list_IDs_temp)

            if (self.conv):
                Y = np.expand_dims(X, axis=2)
            return X, y
        else:
            return X

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """

        if self.use_mulaw:
            self.samples = self.ulaw(self.samples)

        #self.indexes = np.arange(len(self.samples))
        self.indexes = np.arange(np.floor(len(self.samples) - self.window - 1).astype('int64'))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _generate_X(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        if self.raw_input:
            X = np.empty((self.batch_size, self.window, 1))
        else:
            X = np.empty((self.batch_size, self.window, self.n_channels))

        q = (np.max(self.samples)-np.min(self.samples))/(self.n_channels-1)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample

            subdata = self.samples[ID: ID+self.window]

            if self.raw_input:
                X[i, ] = np.expand_dims(subdata, axis=-1)
            else:
                subdata = (subdata/q).astype('int32')

                onehot = np.zeros((self.window, self.n_channels))
                onehot[np.arange(self.window), subdata] = 1

                X[i,] = onehot
                #X[i,] = self._load_grayscale_image(self.image_path + self.labels[ID])

        return X

    def _generate_y(self, list_IDs_temp):
        """Generates data containing batch_size masks
        :param list_IDs_temp: list of label ids to load
        :return: batch if masks
        """

        if self.single_output:
            y = np.empty((self.batch_size, self.n_channels), dtype=int)
        else:
            y = np.empty((self.batch_size, self.window, self.n_channels))

        q = (np.max(self.samples) - np.min(self.samples)) / (self.n_channels-1)


        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            #subdata = self.samples[ID + self.window]

            if self.single_output:
                    subdata = self.samples[ID + self.window]
            else:
                subdata = self.samples[ID + 1: ID + self.window + 1]

            subdata = (subdata/q).astype('int32')
            # Store sample
            onehot = np.zeros(self.n_channels)
            onehot[subdata] = 1

            y[i, ] = onehot
            #y[i,] = self._load_grayscale_image(self.mask_path + self.labels[ID])

        return y


    def ulaw(self, x, u=255):
        x = np.sign(x) * (np.log(1 + u * np.abs(x)) / np.log(1 + u))
        return x

    def ulaw2lin(self, xx, u=255.):
        max_value = np.iinfo('uint8').max
        min_value = np.iinfo('uint8').min
        x = x.astype('float64', casting='safe')
        x -= min_value
        x /= ((max_value - min_value) / 2.)
        x -= 1.
        x = np.sign(x) * (1 / u) * (((1 + u) ** np.abs(x)) - 1)
        x = self.float_to_uint8(x)
        return x

    def float_to_uint8(self, xx):
        x += 1.
        x /= 2.
        uint8_max_value = np.iinfo('uint8').max
        x *= uint8_max_value
        x = x.astype('uint8')
        return x
