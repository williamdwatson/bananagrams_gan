import numpy as np
from numba import njit
from pathlib import Path
from scipy.special import softmax
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from typing import List

BOARD_SIZE = 50

@njit(cache=True)
def indices_to_array(indices: np.ndarray):
    """
    Converts saved `indices` to a 3D array representations

    Parameters
    ----------
    indices : np.ndarray
        1D array of the letter indices in the game
    
    Returns
    -------
    arr : np.ndarray
        3D arary representation of size (BOARD_SIZE, BOARD_SIZE, 27)
    hand : np.ndarray
        Length-26 1D array of the letters in the hand
    """
    arr = np.zeros((27, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    arr[26, ...] = 1    # All cells are empty to begin
    hand = np.zeros(26, dtype=np.float32)
    min_row = indices[::3].min()
    min_col = indices[1::3].min()
    # Data is in order: row, col, letter (0 for A, 1 for B, etc)
    for i in range(0, len(indices), 3):
        arr[indices[i+2], indices[i]-min_row, indices[i+1]-min_col] = 1
        arr[26, indices[i]-min_row, indices[i+1]-min_col] = 0       # This cell isn't empty
        hand[indices[i+2]] += 1
    return arr, hand

class BoardData(Dataset):

    def __init__(self, files: List[str]):
        super().__init__()
        data_arrays = []
        for file_name in files:
            with open(file_name, "rb") as f:
                data_arrays.append(np.frombuffer(f.read(), dtype=np.uint8))
        self.all_games = []
        for d in tqdm(data_arrays, desc='Reading data'):
            self.all_games.extend(arr if i == 0 else arr[1:] for i, arr in enumerate(np.split(d, np.where(d == 255)[0])) if len(arr) > 1)
    
    def __getitem__(self, idx):
        game, hand = indices_to_array(self.all_games[idx])
        game += np.random.random((27, BOARD_SIZE, BOARD_SIZE))
        return softmax(game, axis=0), hand

    def __len__(self):
        return len(self.all_games)

def get_dataloaders(batch_size=64):
    d = Path("training_data/data")
    training_files = []
    testing_files = []
    for bgb in (p for p in d.iterdir() if p.is_file() and p.suffix == ".bgb"):
        if bgb.stem.startswith("0"):
            testing_files.append(bgb)
        else:
            training_files.append(bgb)
    training_data = BoardData(training_files)
    testing_data = BoardData(testing_files)
    training_dl = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    testing_dl = DataLoader(testing_data, batch_size=batch_size, shuffle=True)
    return training_dl, testing_dl

def to_string(b):
    """
    Converts a `b`oard to its string representation

    Parameters
    ----------
    b :
        Tensor or Numpy array representing the board(s)
    
    Returns
    -------
    str or list :
        String representation of `b`; if `b` has a fourth batch dimension, this will be a list of strings of length `b.shape[0]`
    """
    b = np.array(b)
    if len(b.shape) == 4:
        letters = np.argmax(b, axis=1).astype(np.uint8)
        has_letters = np.where(letters != 26)
        letters[letters == 26] = ord(" ") - 65
        letters += 65
        board_strings = []
        for i in range(b.shape[0]):
            mask = has_letters[0] == i
            if not mask.any():
                board_strings.append("")
                continue
            min_col = has_letters[1][mask].min()
            max_col = has_letters[1][mask].max()
            min_row = has_letters[2][mask].min()
            max_row = has_letters[2][mask].max()
            board_strings.append("\n".join("".join(row) for row in letters[i, min_col:max_col+1, min_row:max_row+1].view("S1").astype(str)))
        return board_strings
    else:
        letters = np.argmax(b, axis=0).astype(np.uint8)
        has_letter = np.where(letters != 26)
        min_col = has_letter[0].min()
        max_col = has_letter[0].max()
        min_row = has_letter[1].min()
        max_row = has_letter[1].max()
        letters[letters == 26] = ord(" ") - 65
        letters += 65
        return "\n".join("".join(row) for row in letters[min_col:max_col+1, min_row:max_row+1].view("S1").astype(str))

if __name__ == "__main__":
    import torch
    train_dl, test_dl = get_dataloaders(batch_size=2)
    for b, hand in train_dl:
        b0 = b[0, ...]
        print(b0[:, 0, 0].sum())
        print(b0.sum(dim=(1, 2)))
        print(to_string(b0))
        print("-"*10)
        print(to_string(torch.flip(torch.rot90(b0, k=1, dims=(1, 2)), (1, ))))
        # for s in to_string(b):
        #     print(s)
        #     print("-"*10)
        break
