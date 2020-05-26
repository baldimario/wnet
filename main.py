from datagenerators.datagenerator import DataGenerator
from models.modelconv import get_model
from models.wavenet import build_wavenet_model
from datasets.dataset import get_tedx, get_speech_like
from utils.util import receptive_field
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

config = {
    # architectural parameters
    'dilation_factor': 2,
    'input_shape': None,
    'channels': 64,
    'n_layers': 10,
    'n_dilation_depth': 12,
    'n_filters': 8,
    'filter_width': 2, #2,
    'residual': True,
    'use_ulaw': True,
    # training parameters
    'epochs': 50,
    'batch_size': 256
}


def main():
    rfield = receptive_field(config['n_layers'], config['n_dilation_depth'], config['dilation_factor'])

    print('Receptive field: {}'.format(rfield))


    config['input_shape'] = rfield + 128



    model = get_model(
        input_shape=config['input_shape'],
        channels=config['channels'],
        n_layers=config['n_layers'],
        n_dilation_depth=config['n_dilation_depth'],
        n_filters=config['n_filters'],
        residual=config['residual'],
        filter_width=config['filter_width']
    )

    #model = build_wavenet_model(config['input_shape'], config['n_filters'], config['filter_width'], config['n_layers'], config['channels'])
    model = get_model(
        input_shape=config['input_shape'],
        channels=config['channels'],
        n_layers=config['n_layers'],
        n_dilation_depth=config['n_dilation_depth'],
        n_filters=config['n_filters'],
        filter_width=config['filter_width'],
        residual=config['residual'],
        dilation_factor=config['dilation_factor']
    )
    #model = get_model(config['input_shape'], config['channels'], config['n_layers'], config['n_dilation_depth'], config['n_filters'], config['residual'], config['filter_width'], config['dilation_factor'])

    data = get_tedx()

    #data = get_speech_like(config['input_shape'])

    training_generator = DataGenerator(data,
                                       window=config['input_shape'],
                                       n_channels=config['channels'],
                                       batch_size=config['batch_size'],
                                       single_output=True,
                                       raw_input=False,
                                       use_ulaw=config['use_ulaw']
                                       )
    '''
    y = []
    for i in range(training_generator.__len__()):
        _, y_ = training_generator.__getitem__(i)
        y.append(y_)

    import numpy as np
    y = np.concatenate(y, axis=0)

    p = np.argmax(y, axis=-1).astype('float64')
    p -= np.min(p)
    p /= np.max(p)
    p -= np.mean(p)

    from scipy.io import wavfile
    wavfile.write('./orig.wav', 8000, p)
    '''

    predict_generator = DataGenerator(data,
                                      window=config['input_shape'],
                                      n_channels=config['channels'],
                                      batch_size=config['batch_size'],
                                      to_fit=False,
                                      single_output=True,
                                      raw_input=False
                                      )

    model.summary()

    model.fit(training_generator, epochs=config['epochs'])
    model.save('model.h5')

    exit()

    from tensorflow.keras.models import load_model
    model = load_model('model.h5')

    predicted = model.predict(predict_generator)

    import matplotlib.pyplot as plt
    plt.imshow(predicted[:, :].T)
    plt.show()
    print(predicted.shape)

    import numpy as np
    p = np.argmax(predicted, axis=-1).astype('float64')
    p -= np.min(p)
    p /= np.max(p)
    p -= np.mean(p)

    from scipy.io import wavfile
    wavfile.write('./pred.wav', 8000, p)

if __name__ == '__main__':
    main()
