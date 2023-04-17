import torch
import torch.nn as nn

from models.attention import Seq_Transformer



class TC(nn.Module):
    def __init__(self, configs, device):
        super(TC, self).__init__()
        self.num_channels = configs.final_out_channels
        self.lsoftmax = nn.LogSoftmax(1)
        self.device = device

        self.projection_head = nn.Sequential(
            nn.Conv1d(128, 1, kernel_size=1, stride=1, bias=False, padding=0),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(400, 100),

            nn.ReLU(),
            nn.Linear(100, 20),
            nn.ReLU(),
            nn.Linear(20, 2),
            nn.Dropout(configs.dropout)
        )

        self.seq_transformer = Seq_Transformer(patch_size=882, dim=400, depth=1,
                                               heads=8, mlp_dim=200)

    def forward(self, features_aug1, features_aug2, features_aug3):

        z = torch.concat([features_aug1, features_aug2, features_aug3], dim=-1)
        c_t = self.seq_transformer(z)
        yt = self.projection_head(c_t).squeeze(dim=1)

        return self.lsoftmax(yt)