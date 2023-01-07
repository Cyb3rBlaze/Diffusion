class Config():
    def __init__(self):
        self.epochs = 20

        self.batch_size = 16
        
        self.lr = 0.0001
        self.beta = 0.5

        self.img_dims = (28, 28)

        self.encoder_in_channels = 32
        self.decoder_in_channels = 256

        self.time_dim = 10