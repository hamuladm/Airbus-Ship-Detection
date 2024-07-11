
import keras

class UnetCNN(object):
    def __init__(self, X, Y, init_dims: tuple) -> None:
        self.X = X
        self.Y = Y
        self.init_dims = init_dims

        self.inpt = self.inpt()

    def inpt(self):
        inpts = keras.layers.Input(self.init_dims)

        return inpts

    def encode(self, inpt, layer_sz):
        S = keras.layers.Conv2D(layer_sz, (3,3), padding='same', activation='relu')(inpt)
        S = keras.layers.Conv2D(layer_sz, (3,3), padding='same', activation='relu')(S)
        P = keras.layers.MaxPool2D(pool_size=(2,2))(S)

        return S, P
    
    def decode(self, inpt, prev_inpt, layer_sz):
        D = keras.layers.UpSampling2D(size=(2,2))(inpt)
        D =  keras.layers.Conv2D(layer_sz, (3,3), padding='same', activation='relu')(D)
        D =  keras.layers.Conv2D(layer_sz, (3,3), padding='same', activation='relu')(D)
        D = self.concat(prev_inpt, inpt)

        return D

    @staticmethod    
    def bridge(inpt, layer_sz):
        S = keras.layers.Conv2D(layer_sz, (3,3), padding='same', activation='relu')(inpt)
        S = keras.layers.Conv2D(layer_sz, (3,3), padding='same', activation='relu')(S)

        return S

    @staticmethod
    def concat(enc_layer, dec_layer):
        enc_layer = keras.layers.Concatenate()([enc_layer, dec_layer])

        return enc_layer
    
    def build_network(self):
        S1, P1 = self.encode(self.inpt, 64)
        S2, P2 = self.encode(P1, 128)
        S3, P3 = self.encode(P2, 256)
        S4, P4 = self.encode(P3, 512)

        B = self.bridge(P4)
        D1 = self.decode(B, S4, 512)
        D2 = self.decode(D1, S3, 256)
        D3 = self.decode(D2, S2, 128)
        D4 = self.decode(D3, S1, 64)

        output = keras.layers.Conv2D(1, 1, padding='same', activation='relu')(D4)
        model = keras.models.Model(self.inpt, output)

        return model
