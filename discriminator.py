import torch
import torch.nn.functional as F
import torch.nn as nn

class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(50*27, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 32)
        )

        self.board_reduce = nn.Sequential(
            nn.Linear(1600, 1280),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128)
        )

        # self.conv = nn.Sequential(
        #     nn.Conv2d(27, 16, 5, padding='valid'),
        #     nn.MaxPool2d(2),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 8, 3, padding='valid'),
        #     nn.MaxPool2d(2),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(9248, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(512, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 32),
        #     nn.ReLU()
        # )

        self.final = nn.Sequential(
            nn.Linear(128 + 128 + 53, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.float()

    def forward(self, board, hand):
        batch_size, channels, height, width = board.shape
        h = self.board_reduce(self.linear(board.view(batch_size*height, channels, width)).view(batch_size, -1))
        v = self.board_reduce(self.linear(board.permute(0, 1, 3, 2).reshape(batch_size*width, channels, height)).view(batch_size, -1))
        # Was originally using these list comprehensions, but it's much faster to treat them like large batches
        # h = self.board_reduce(torch.hstack([self.linear(board[:, :, i, :]) for i in range(144)]))
        # v = self.board_reduce(torch.hstack([self.linear(board[..., i]) for i in range(144)]))
        x_hand = torch.hstack((board.sum(dim=(2, 3)), hand))
        # x_conv = self.conv(board)

        return self.final(torch.hstack((h, v, x_hand))).squeeze(1)


class Discriminator2(nn.Module):

    def __init__(self):
        super().__init__()
        # Convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(27, 64, kernel_size=(1, 7), stride=1, padding='valid'),
            nn.MaxPool2d((1, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=(1, 5), stride=1, padding='valid'),
            nn.MaxPool2d((1, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 24, kernel_size=(1, 3), stride=1, padding='valid'),
            nn.MaxPool2d((1, 2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(51840, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Additional convolutional layers
        self.conv2 = nn.Sequential(
            nn.Conv2d(27, 64, kernel_size=5, stride=1, padding='valid'),
            nn.MaxPool2d((2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding='valid'),
            nn.MaxPool2d((2, 2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(36992, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.linear = nn.Sequential(
            nn.Linear(128 + 128 + 128 + 53, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.float()

    def forward(self, board, hand):
        x_h = self.conv(board)
        x_v = self.conv(torch.flip(torch.rot90(board, k=1, dims=(2, 3)), (2, )))
        x_board = self.conv2(board)
        x_hand = torch.hstack((board.sum(dim=(2, 3)), hand))

        # Combine the convolutions
        x = torch.hstack((x_h, x_v, x_board, x_hand))

        return self.linear(x).squeeze(1)
