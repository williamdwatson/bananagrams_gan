import pickle, torch
import matplotlib.pyplot as plt
from dataloader import get_dataloaders, to_string
from generator import Generator
from discriminator import Discriminator

if __name__ == "__main__":
    DEVICE = torch.device("cpu")
    test_dl = get_dataloaders(batch_size=100)[1]

    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)
    G.load_state_dict(torch.load("checkpoints/generator_0_15500.pt", map_location="cpu"))
    D.load_state_dict(torch.load("checkpoints/discriminator_0_15500.pt", map_location="cpu"))
    G.eval()
    D.eval()

    num_correct = 0
    num_fooled = 0
    num = 0
    real_boards = []
    fooled_boards = []
    not_fooled_boards = []
    with torch.no_grad():
        for boards, hands in test_dl:
            boards = boards.to(DEVICE)
            hands = hands.to(DEVICE)
            generated = G(hands)
            predictions_real = torch.sigmoid(D(boards, hands)).round()
            predictions_gen = torch.sigmoid(D(generated, hands)).round()
            num_fooled += torch.count_nonzero(predictions_gen == 1)
            num_correct += torch.count_nonzero(predictions_real == 1) + torch.count_nonzero(predictions_gen == 0)
            num += boards.shape[0]
            # real_boards.extend(to_string(boards))
            fooled_boards.extend(to_string(generated[predictions_gen == 1]))
            not_fooled_boards.extend(to_string(generated[predictions_gen == 0]))
            if num >= 100:
                break
    print(f"Discriminator accuracy: {100*num_correct/num:.2f}%")
    print(f"Percentage of generated boards that fooled: {100*num_fooled/(num/2):.2f}%")
    if fooled_boards:
        print("Examples of boards that fooled the discriminator:")
        for i in range(min(3, len(fooled_boards))):
            print(fooled_boards[i])
            print("-"*15)
    else:
        print("No generated boards fooled the discriminator!")
    if not_fooled_boards:
        print("Examples of boards that did not fool the discriminator")
        for i in range(min(3, len(not_fooled_boards))):
            print(not_fooled_boards[i])
            print("-"*15)
    else:
        print("Every generated board passed the discriminator!")

    with open("checkpoints/loss_0_15500.pkl", "rb") as f:
        loss = pickle.load(f)
    
    plt.plot([i*10 for i in range(len(loss["generator"]))], loss["generator"], label="Generator")
    plt.plot([i*10 for i in range(len(loss["discriminator"]))], loss["discriminator"], label="Discriminator")
    plt.legend()
    plt.ylabel("Loss")
    plt.xlabel("Minibatch number")
    plt.show()
