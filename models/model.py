from torch import nn

class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()
        hidden_channel = 128
        # data1
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, hidden_channel, kernel_size=1,
                      stride=configs.stride, bias=False, padding=0),  # (84-7+6)/1+1=84
            nn.BatchNorm1d(hidden_channel),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=1, stride=2, padding=1),  # (84-2+2)/2+1=43
            # nn.Dropout(configs.dropout)
        )

        # data2
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(101, hidden_channel, kernel_size=1, #92
                      stride=configs.stride, bias=False, padding=0),  # (30-7+4)/1+1=28
            nn.BatchNorm1d(hidden_channel),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=4, stride=1, padding=1),  # (28-4+2)/1+1=27
            # nn.Dropout(configs.dropout)
        )


        # data3
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(101, hidden_channel, kernel_size=1,  # 92
                      stride=configs.stride, bias=False, padding=0),  # (30-7+4)/1+1=28
            nn.BatchNorm1d(hidden_channel),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=4, stride=1, padding=1),  # (28-4+2)/1+1=27
            # nn.Dropout(configs.dropout)
        )

    def forward(self, x_in, tag):
        if tag==1:
            x = self.conv_block1(x_in)  # Conv1d: (128-8+2*4)/1 +1=129 MaxPool1d: (129-2+2*1)/2+1=65

        elif tag==2:
            x = self.conv_block2(x_in)  # Conv1d: (128-8+2*4)/1 +1=129 MaxPool1d: (129-2+2*1)/2+1=65

        elif tag==3:
            x = self.conv_block3(x_in)

        return  x
