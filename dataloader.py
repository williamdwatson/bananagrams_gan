import torch
import numpy as np
from numba import njit
from pathlib import Path
from scipy.special import softmax
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from typing import List, Union

BOARD_SIZE = 50
EMBEDDING_SIZE = 27
MAX_HAND_SIZE = 144

@njit(cache=True)
def indices_to_sequence(indices: np.ndarray):
    """
    Converts saved `indices` to a 2D array representation

    Parameters
    ----------
    indices : np.ndarray
        1D array of the letter indices in the game
    
    Returns
    -------
    arr : np.ndarray
        2D arary representation of size (BOARD_SIZE*BOARD_SIZE, EMBEDDING_SIZE)
    hand : np.ndarray
        2D array representation of the hand in sequence form
    j : int
        Number of letters in `hand` (i.e. hand[j:] would be all zeros)
    """
    arr = np.zeros((BOARD_SIZE*BOARD_SIZE, EMBEDDING_SIZE), dtype=np.float32)
    arr[..., arr.shape[1]-1] = 1  # All cells are empty to begin
    hand = np.zeros((MAX_HAND_SIZE, EMBEDDING_SIZE), dtype=np.float32)
    min_row = indices[::3].min()
    min_col = indices[1::3].min()
    # Data is in order: row, col, letter (0 for A, 1 for B, etc)
    j = 0
    for i in range(0, len(indices), 3):
        row_idx = indices[i]-min_row
        col_idx = indices[i+1]-min_col
        arr[(row_idx*BOARD_SIZE)+col_idx, indices[i+2]] = 1
        arr[(row_idx*BOARD_SIZE)+col_idx, arr.shape[1]-1] = 0
        hand[j, indices[i+2]] = 1
        j += 1
    return arr, hand, j

@njit(cache=True)
def indices_to_array(indices: np.ndarray):
    """
    Converts saved `indices` to a 3D array representation

    Parameters
    ----------
    indices : np.ndarray
        1D array of the letter indices in the game
    
    Returns
    -------
    arr : np.ndarray
        3D arary representation of size (27, BOARD_SIZE, BOARD_SIZE)
    hand : np.ndarray
        Length-26 1D array of the letters in the hand
    """
    arr = np.zeros((EMBEDDING_SIZE, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    arr[arr.shape[0]-1, ...] = 1    # All cells are empty to begin
    hand = np.zeros(26, dtype=np.float32)
    min_row = indices[::3].min()
    min_col = indices[1::3].min()
    # Data is in order: row, col, letter (0 for A, 1 for B, etc)
    for i in range(0, len(indices), 3):
        arr[indices[i+2], indices[i]-min_row, indices[i+1]-min_col] = 1
        arr[arr.shape[0]-1, indices[i]-min_row, indices[i+1]-min_col] = 0       # This cell isn't empty
        hand[indices[i+2]] += 1
    return arr, hand

class BoardData(Dataset):

    def __init__(self, files: List[str], as_sequence: bool=False, add_random: bool=True):
        """
        Dataset of Bananagrams boards

        Parameters
        ----------
        files : list
            List of string filenames to load
        as_sequence : bool, default=False
            Whether to return the boards as 3D tensors or flattened tensors (i.e. sequences)
        add_random : bool, default=True
            Whether to add random noise to the data (followed by a softmax so information is conserved)
        """
        super().__init__()
        self.as_sequence = as_sequence
        self.add_random = add_random
        data_arrays = []
        for file_name in files:
            with open(file_name, "rb") as f:
                data_arrays.append(np.frombuffer(f.read(), dtype=np.uint8))
        self.all_games = []
        for d in tqdm(data_arrays, desc='Reading data'):
            self.all_games.extend(arr if i == 0 else arr[1:] for i, arr in enumerate(np.split(d, np.where(d == 255)[0])) if len(arr) > 1)
    
    def __getitem__(self, idx: int):
        if self.as_sequence:
            game, hand, j = indices_to_sequence(self.all_games[idx])
            if self.add_random:
                game += np.random.random(game.shape)
                return softmax(game, axis=1), hand, j
            else:
                return game, hand, j
        else:
            game, hand = indices_to_array(self.all_games[idx])
            if self.add_random:
                game += np.random.random(game.shape)
                return softmax(game, axis=0), hand
            else:
                return game, hand

    def __len__(self):
        return len(self.all_games)

def get_dataloaders(batch_size: int=64, as_sequence: bool=False, add_random: bool=True):
    d = Path("training_data/data")
    training_files = []
    testing_files = []
    for bgb in (p for p in d.iterdir() if p.is_file() and p.suffix == ".bgb"):
        if bgb.stem.startswith("0"):
            testing_files.append(bgb)
        else:
            training_files.append(bgb)
    training_data = BoardData(training_files, as_sequence=as_sequence, add_random=add_random)
    testing_data = BoardData(testing_files, as_sequence=as_sequence, add_random=add_random)
    training_dl = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    testing_dl = DataLoader(testing_data, batch_size=batch_size, shuffle=True)
    return training_dl, testing_dl

def to_string(b: Union[torch.Tensor, np.ndarray], needs_argmax: bool=True):
    """
    Converts a `b`oard to its string representation

    Parameters
    ----------
    b :
        Tensor or Numpy array representing the board(s)
    needs_argmax : bool, default=True
        Whether `b` needs to be argmax-ed, or whether it contains the letter indices already
    
    Returns
    -------
    str or list :
        String representation of `b`; if `b` has a batch dimension, this will be a list of strings of length `b.shape[0]`
    """
    b = np.array(b)
    if (needs_argmax and len(b.shape) == 4) or len(b.shape) == 3:
        letters = np.argmax(b, axis=1).astype(np.uint8) if needs_argmax else b.astype(np.uint8)
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
        letters = np.argmax(b, axis=0).astype(np.uint8) if needs_argmax else b.astype(np.uint8)
        has_letter = np.where(letters != 26)
        min_col = has_letter[0].min()
        max_col = has_letter[0].max()
        min_row = has_letter[1].min()
        max_row = has_letter[1].max()
        letters[letters == 26] = ord(" ") - 65
        letters += 65
        return "\n".join("".join(row) for row in letters[min_col:max_col+1, min_row:max_row+1].view("S1").astype(str))

if __name__ == "__main__":
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
