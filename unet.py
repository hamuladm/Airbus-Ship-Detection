
import keras

class UnetCNN(object):
    def __init__(self, X, Y, init_dims: tuple) -> None:
        self.X = X
        self.Y = Y
        self.init_dims = init_dims


    def inpt(self):
        inpts = keras.layers.Input(self.init_dims)
        return inpts

    @staticmethod
    def maxpool(inpt):
        S = keras.layers.MaxPool2D(pool_size=(2,2))(inpt)
        return S
    
    def encode(self, inpt, layer_sz):
        S = keras.layers.Conv2D(layer_sz, (3,3), padding='same', activation='relu')(inpt)
        S = keras.layers.Conv2D(layer_sz, (3,3), padding='same', activation='relu')(S)
        P = self.maxpool(inpt)

        return P
    
    def decode(self, inpt, layer_sz):
        D = keras.layers.UpSampling2D(size=(2,2))(inpt)
        D =  keras.layers.Conv2D(layer_sz, (3,3), padding='same', activation='relu')(D)
        D =  keras.layers.Conv2D(layer_sz, (3,3), padding='same', activation='relu')(D)

        return D

    @staticmethod
    def concat(enc_layer, dec_layer):
        enc_layer = keras.layers.Concatenate([enc_layer, dec_layer])

        return enc_layer