import pickle, torch
import torch.nn as nn
import torch.optim as optim
from dataloader import get_dataloaders, EMBEDDING_SIZE, to_string
from tqdm import tqdm
from transformer import Seq2Seq

NUM_EPOCHS = 5

if __name__ == "__main__":
    DEVICE = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    train_dl, test_dl = get_dataloaders(batch_size=16, as_sequence=True, add_random=False)

    model = Seq2Seq().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    losses = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        with tqdm(total=NUM_EPOCHS*len(train_dl), desc=f"Epoch {epoch+1}") as pbar:
            for i, (boards, hands, hand_lens) in enumerate(train_dl):
                optimizer.zero_grad()
                # Move everything to the proper device
                boards = boards.to(DEVICE)
                hands = hands.to(DEVICE)
                hand_lens = hand_lens.to(DEVICE)
                tokens = boards.argmax(dim=-1)
                generated = tokens.reshape(-1, 50, 50)
                # Get the predicted output
                output = model(boards, hands, hand_lens)
                # Calculate the loss (removing the SoS token from the output)
                loss = criterion(output.reshape(-1, EMBEDDING_SIZE), boards.argmax(dim=-1).flatten())
                # Update the model
                loss.backward()
                optimizer.step()
                # Update the progress
                pbar.set_postfix(loss=loss.item())
                pbar.update()
                losses.append(loss.cpu().item())

                if i % 500 == 0 and i != 0:
                    torch.save(model.state_dict(), f"checkpoints/{epoch}_{i}.pt")
                    with open(f"checkpoints/loss_{epoch}_{i}.pkl", "wb") as f:
                        pickle.dump(losses, f)
