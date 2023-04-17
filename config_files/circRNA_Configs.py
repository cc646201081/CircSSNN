class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 101
        self.kernel_size = 7
        self.stride = 1
        self.final_out_channels = 128 #

        self.num_classes = 2
        self.dropout = 0.35
        self.features_len = 22


        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-3 #3e-4

        # data parameters
        self.drop_last = True
        self.batch_size = 64

