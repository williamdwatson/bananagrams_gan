import pickle, torch
import torch.nn as nn
import torch.optim as optim
from dataloader import get_dataloaders
from generator import Generator
from discriminator import Discriminator
from tqdm import tqdm

NUM_EPOCHS = 5

if __name__ == "__main__":
    DEVICE = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    train_dl, test_dl = get_dataloaders(batch_size=16)

    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizerD = optim.Adam(D.parameters(), lr=5e-5, betas=(0.5, 0.999))
    optimizerG = optim.Adam(G.parameters(), lr=5e-5, betas=(0.5, 0.999))

    d_loss = []
    g_loss = []
    current_d_loss = []
    current_g_loss = []
    
    for epoch in range(NUM_EPOCHS):
        G.train()
        D.train()
        with tqdm(total=NUM_EPOCHS*len(train_dl), desc=f"Epoch {epoch+1}") as pbar:
            for i, (boards, hands) in enumerate(train_dl):
                optimizerD.zero_grad()
                hands.requires_grad = False
                boards = boards.to(DEVICE)
                hands = hands.to(DEVICE)
                # First run the discriminator on the real boards
                predictions_real = D(boards, hands)
                lossD_real = criterion(predictions_real, torch.ones(boards.shape[0], dtype=torch.float32, device=DEVICE))
                lossD_real.backward()
                # Generate boards using the real hands
                generated = G(hands)
                predictions_gen = D(generated.detach(), hands)
                lossD_gen = criterion(predictions_gen, torch.zeros(boards.shape[0], dtype=torch.float32, device=DEVICE))
                lossD_gen.backward()
                lossD = lossD_real + lossD_gen
                optimizerD.step()

                optimizerG.zero_grad()
                predictions_gen = D(generated, hands)
                lossG = criterion(predictions_gen, torch.ones(boards.shape[0], dtype=torch.float32, device=DEVICE))
                lossG.backward()
                optimizerG.step()

                current_d_loss.append(lossD.item())
                current_g_loss.append(lossG.item())
                if len(current_d_loss) == 10:
                    d_loss.append(sum(current_d_loss)/10)
                    g_loss.append(sum(current_g_loss)/10)
                    current_g_loss.clear()
                    current_d_loss.clear()
                    pbar.set_postfix(lossD=d_loss[-1], lossG=g_loss[-1])
                pbar.update(1)

                if i % 500 == 0:
                    torch.save(G.state_dict(), f"checkpoints/generator_{epoch}_{i}.pt")
                    torch.save(D.state_dict(), f"checkpoints/discriminator_{epoch}_{i}.pt")
                    with open(f"checkpoints/loss_{epoch}_{i}.pkl", "wb") as f:
                        pickle.dump({"discriminator": d_loss, "generator": g_loss}, f)
        