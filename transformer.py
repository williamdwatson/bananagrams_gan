import torch
import torch.nn as nn
from dataloader import BOARD_SIZE, EMBEDDING_SIZE, MAX_HAND_SIZE

class Seq2Seq(nn.Module):
    
    def __init__(self):
        super().__init__()
        # Start of sequence token
        self.sos = nn.Parameter(torch.randn(EMBEDDING_SIZE))
        # Positional encoding
        self.pos = nn.Parameter(torch.randn(BOARD_SIZE*BOARD_SIZE, EMBEDDING_SIZE))
        self.transformer = nn.Transformer(EMBEDDING_SIZE, nhead=1, dim_feedforward=256, num_decoder_layers=2, num_encoder_layers=2, batch_first=True)
        self.float()
    
    def forward(self, board: torch.Tensor, hand: torch.Tensor, hand_len: torch.Tensor):
        batch_size = board.shape[0]
        # Prepend the learnable SoS token and add the positional embedding
        board_seq = self.pos + torch.cat((self.sos.expand(batch_size, 1, EMBEDDING_SIZE), board[:, :-1, :]), dim=1)
        # Generate the target mask to prevent cheating
        tgt_mask = self.transformer.generate_square_subsequent_mask(board_seq.size(1)).to(board.device)
        # Generate the hand padding mask
        src_key_padding_mask = torch.arange(MAX_HAND_SIZE, device=board.device).expand(batch_size, -1) >= hand_len.unsqueeze(1)
        # Pass everything through the transformer
        output = self.transformer(src=hand, tgt=board_seq, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask)
        return output
