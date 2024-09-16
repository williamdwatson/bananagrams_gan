import pickle, torch
import matplotlib.pyplot as plt
from dataloader import get_dataloaders, to_string, BOARD_SIZE, EMBEDDING_SIZE, MAX_HAND_SIZE
from transformer import Seq2Seq
from tqdm import trange

BATCH_SIZE = 10

if __name__ == "__main__":
    DEVICE = torch.device("cuda")# if not torch.cuda.is_available() else "cuda")
    test_dl = get_dataloaders(batch_size=BATCH_SIZE, as_sequence=True, add_random=False)[1]

    model = Seq2Seq().to(DEVICE)
    model.load_state_dict(torch.load("checkpoints/0_2500.pt", map_location=DEVICE))
    # model.eval()

    with torch.no_grad():
        for boards, hands, hand_lens in test_dl:
            # Move everything to the proper device
            boards = boards.to(DEVICE)
            hands = hands.to(DEVICE)
            hand_lens = hand_lens.to(DEVICE)

            current_seq = model.sos.expand(BATCH_SIZE, 1, EMBEDDING_SIZE)  # Start with SoS
            generated = []
            for i in trange(BOARD_SIZE*BOARD_SIZE):
                # Add positional encoding to the current sequence
                current_seq_with_pos = model.pos[:current_seq.shape[1]] + current_seq
                # Generate the target mask
                tgt_mask = model.transformer.generate_square_subsequent_mask(current_seq.shape[1]).to(DEVICE)
                # Generate the source padding mask
                src_key_padding_mask = torch.arange(MAX_HAND_SIZE, device=DEVICE).expand(BATCH_SIZE, -1) >= hand_lens.unsqueeze(1)
                # Pass through the transformer
                output = model.transformer(src=hands, tgt=current_seq_with_pos, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask)
                # Get the most likely token
                next_token = output[:, -1, :].argmax(dim=-1)
                generated.append(next_token)
                next_token_one_hot = torch.nn.functional.one_hot(next_token, num_classes=EMBEDDING_SIZE).to(DEVICE)
                # Append that to the current sequence
                current_seq = torch.cat((current_seq, next_token_one_hot.unsqueeze(1)), dim=1)

            generated = torch.vstack(generated).transpose(1, 0).reshape(-1, BOARD_SIZE, BOARD_SIZE)
            for g in to_string(generated.cpu(), needs_argmax=False):
                print(g)
                print("-"*10)
            break
