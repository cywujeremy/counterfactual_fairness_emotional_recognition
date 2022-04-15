import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class ACRNN(nn.Module):
    def __init__(self, num_classes=4, is_training=True,
                 L1=128, L2=256, cell_units=128, num_linear=768,
                 p=10, time_step=150, F1=64, dropout_keep_prob=1):
        super(ACRNN, self).__init__()

        self.num_classes = num_classes
        self.is_training = is_training
        self.L1 = L1
        self.L2 = L2
        self.cell_units = cell_units
        self.num_linear = num_linear
        self.p = p
        self.time_step = time_step
        self.F1 = F1
        self.dropout_prob = 1 - dropout_keep_prob

        # tf filter : [filter_height, filter_width, in_channels, out_channels]
        self.conv1 = nn.Conv2d(3, self.L1, (5, 3), padding=(2, 1))       # [5, 3,   3, 128]  
        self.conv2 = nn.Conv2d(self.L1, self.L2, (5, 3), padding=(2, 1)) # [5, 3, 128, 256]
        self.conv3 = nn.Conv2d(self.L2, self.L2, (5, 3), padding=(2, 1)) # [5, 3, 256, 256]
        self.conv4 = nn.Conv2d(self.L2, self.L2, (5, 3), padding=(2, 1)) # [5, 3, 256, 256]
        self.conv5 = nn.Conv2d(self.L2, self.L2, (5, 3), padding=(2, 1)) # [5, 3, 128, 256]
        self.conv6 = nn.Conv2d(self.L2, self.L2, (5, 3), padding=(2, 1)) # [5, 3, 128, 256]

        self.linear1 = nn.Linear(self.p*self.L2, self.num_linear) # [10*256, 768]
        self.bn = nn.BatchNorm1d(self.num_linear)

        self.relu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout2d(p=self.dropout_prob)
        
        self.rnn = nn.LSTM(input_size=self.num_linear, hidden_size=self.cell_units, 
                            batch_first=True, num_layers=1, bidirectional=True) 

        # for attention
        self.a_fc1 = nn.Linear(2*self.cell_units, 1)  
        self.a_fc2 = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        # fully connected layers
        self.fc1 = nn.Linear(2*self.cell_units, self.F1) # [2*128, 64]
        self.fc2 = nn.Linear(self.F1, self.num_classes) # [num_classes]

    
    def forward(self, x):
        
        layer1 = self.relu(self.conv1(x))
        layer1 = F.max_pool2d(layer1, kernel_size=(2, 4), stride=(2, 4))   # [1,2,4,1], padding = 'valid'
        layer1 = self.dropout(layer1)

        layer2 = self.relu(self.conv2(layer1))
        layer2 = self.dropout(layer2)
        
        layer3 = self.relu(self.conv3(layer2))
        layer3 = self.dropout(layer3)

        layer4 = self.relu(self.conv4(layer3))
        layer4 = self.dropout(layer4)

        layer5 = self.relu(self.conv5(layer4))
        layer5 = self.dropout(layer5)

        layer6 = self.relu(self.conv6(layer5))
        layer6 = self.dropout(layer6)
        
        # lstm
        layer6 = layer6.permute(0, 2, 3, 1)
        layer6 = layer6.reshape(-1, self.time_step, self.L2*self.p)        # (-1, 150, 256*10)
        layer6 = layer6.reshape(-1, self.L2*self.p)                        # (1500, 2560)

        linear1 = self.relu(self.bn(self.linear1(layer6)))                 # [1500, 768]
        linear1 = linear1.reshape(-1, self.time_step, self.num_linear)     # [10, 150, 768]

        outputs1, output_states1 = self.rnn(linear1)                       # outputs1 : [10, 150, 128] (B,T,D)

        # # attention
        v = self.sigmoid(self.a_fc1(outputs1))                  # (10, 150, 1)
        alphas = self.softmax(self.a_fc2(v).squeeze())          # (B,T) shape, alphas are attention weights
        gru = (alphas.unsqueeze(2) * outputs1).sum(axis=1)      # (B,D)
        
        # # fc
        fully1 = self.relu(self.fc1(gru))
        fully1 = self.dropout(fully1)
        Ylogits = self.fc2(fully1)
        Ylogits = self.softmax(Ylogits)

        return Ylogits


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, num_feats):
        super(ResidualBlock, self).__init__()
        self.norm = nn.LayerNorm(num_feats)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.2)
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
    
    def forward(self, X):
        X = self.norm(X.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        return self.conv(self.dropout(self.gelu(X))) + X

class ACRNN2(nn.Module):

    def __init__(self):
        super(ACRNN2, self).__init__()

        self.embedding = nn.Sequential(*[
            nn.Conv2d(in_channels=3,   out_channels=128, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(128),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(256),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(256),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(256),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(256),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(256),
            nn.Dropout(0.3),
            nn.GELU(),
        ])

        self.linear = nn.Sequential(*[
            nn.Linear(10240, 768),
            nn.LeakyReLU(0.01),
            nn.Dropout2d(0.3)
        ])

        self.rnn = nn.LSTM(input_size=768, hidden_size=256, num_layers=4, dropout=0.5, bidirectional=True)

        # self.classifier = nn.Sequential(*[
        #     nn.Linear(512, 1024),
        #     nn.GELU(),
        #     nn.Linear(1024, 11),
        #     nn.Softmax(dim=2)
        # ])

        self.a_fc1 = nn.Linear(512, 1)
        self.a_fc2 = nn.Linear(1, 1)
        self.a_sigmoid = nn.Sigmoid()
        
        self.a_softmax = nn.Softmax(dim=1)

        self.classifier = nn.Sequential(*[
            nn.Linear(512, 2048),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 4),
            nn.Softmax(dim=1)
        ])


    def forward(self, X):

        X = X.permute((1, 2, 0, 3))
        out = self.embedding(X)
        out = out.permute((2, 0, 1, 3))

        out_shape = out.shape
        out = out.reshape((out_shape[0], out_shape[1], -1))
        out = self.linear(out)

        # packed_out = pack_padded_sequence(out, X_lengths, enforce_sorted=False)
        out, (_, _) = self.rnn(out)
        # out, lengths = pad_packed_sequence(out)
        # out = out.permute(1, 0, 2).reshape((out.shape[0], -1))
        # out = self.classifier(out)

        out = out.permute(1, 0, 2)
        v = self.a_sigmoid(self.a_fc1(out))
        alphas = self.a_softmax(self.a_fc2(v).squeeze())
        gru = (alphas.unsqueeze(2) * out).sum(axis=1)

        out = self.classifier(gru)

        # out = torch.stack([out[X_lengths[i] - 1, i, :] for i in range(out.shape[1])])

        return out