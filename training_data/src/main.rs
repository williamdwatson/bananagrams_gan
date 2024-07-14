use std::{cmp, f32::consts::E, fmt, fs, thread};
use hashbrown::HashSet;     // For faster default hash (ahash)
use rand::prelude::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// A numeric representation of a word
type Word = Vec<usize>;
/// Represents a hand of letters
type Letters = [usize; 26];

/// The maximum length of any word in the dictionary
const MAX_WORD_LENGTH: usize = 17;
/// Value of an empty cell on the board
const EMPTY_VALUE: usize = 30;
/// Number rows/columns in the board
const BOARD_SIZE: usize = 144;
/// Number of letters present on the board that can be used in a word (higher will result in fewer words being filtered out)
const FILTER_LETTERS_ON_BOARD: u8 = 2;
/// Maximum number of words to check before the solver stops trying a given word
const MAXIMUM_WORDS_CHECKED: usize = 500_000;
/// Minimum size of hand of letters to generate
const MINIMUM_HAND_SIZE: f32 = 11.0;
/// Maximum size of hand of letters to generate
const MAXIMUM_HAND_SIZE: f32 = 72.0;
/// Base to use when generating the 
const BASE: f32 = E;
/// All the letters present in standard Bananagrams as ASCII values
const TO_CHOOSE_FROM: [usize; 144] = [65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 66, 66, 66, 67, 67, 67, 68, 68, 68, 68, 68,
                                      68, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 70, 70, 70, 71, 71,
                                      71, 71, 72, 72, 72, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 74, 74, 75, 75, 76, 76, 76,
                                      76, 76, 77, 77, 77, 78, 78, 78, 78, 78, 78, 78, 78, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79,
                                      80, 80, 80, 81, 81, 82, 82, 82, 82, 82, 82, 82, 82, 82, 83, 83, 83, 83, 83, 83, 84, 84, 84, 84,
                                      84, 84, 84, 84, 84, 85, 85, 85, 85, 85, 85, 86, 86, 86, 87, 87, 87, 88, 88, 89, 89, 89, 90, 90];

/// A thin wrapper for handling the board
#[derive(Clone)]
struct Board {
    /// The underlying vector of the board (as in optimization level 0 the array overflows the stack)
    arr: Vec<usize>
}
impl Board {
    /// Creates a new board of dimensions `BOARD_SIZE`x`BOARD_SIZE` filled with the `EMPTY_VALUE`
    fn new() -> Board {
        return Board { arr: vec![EMPTY_VALUE; BOARD_SIZE*BOARD_SIZE] }
    }

    /// Unsafely gets a value from the board at the given index
    /// # Arguments
    /// * `row` - Row index of the value to get (must be less than `BOARD_SIZE`)
    /// * `col` - Column index of the value to get (must be less than `BOARD_SIZE`)
    /// # Returns
    /// `usize` - The value in the board at `(row, col)` (if either `row` or `col` are greater than `BOARD_SIZE` this will be undefined behavior)
    fn get_val(&self, row: usize, col: usize) -> usize {
        return unsafe { *self.arr.get_unchecked(row*BOARD_SIZE + col) };
    }

    /// Unsafely sets a value in the board at the given index
    /// # Arguments
    /// * `row` - Row index of the value to get (must be less than `BOARD_SIZE`)
    /// * `col` - Column index of the value to get (must be less than `BOARD_SIZE`)
    /// * `val` - Value to set at `(row, col)` in the board (if either `row` or `col` are greater than `BOARD_SIZE` this will be undefined behavior)
    fn set_val(&mut self, row: usize, col: usize, val: usize) {
        let v = unsafe { self.arr.get_unchecked_mut(row*BOARD_SIZE + col) };
        *v = val;
    }
}

/// Converts a `board` to a `String`
/// # Arguments
/// * `board` - Board to display
/// * `min_col` - Minimum occupied column index
/// * `max_col` - Maximum occupied column index
/// * `min_row` - Minimum occupied row index
/// * `max_row` - Maximum occupied row index
/// # Returns
/// * `String` - `board` in string form (with all numbers converted to letters)
fn board_to_string(board: &Board, min_col: usize, max_col: usize, min_row: usize, max_row: usize) -> String {
    let mut board_string: Vec<char> = Vec::with_capacity((max_row-min_row)*(max_col-min_col));
    for row in min_row..max_row+1 {
        for col in min_col..max_col+1 {
            if board.get_val(row, col) == EMPTY_VALUE {
                board_string.push(' ');
            }
            else {
                board_string.push((board.get_val(row, col) as u8+65) as char);
            }
        }
        board_string.push('\n');
    }
    let s: String = board_string.iter().collect();
    return s.trim_end().to_owned();
}


/// Converts a word into a numeric vector representation
/// # Arguments
/// * `word` - String word to convert
/// # Returns
/// `Word` - numeric representation of `word`, with each letter converted from 65 ('A') to 90 ('Z')
/// # See also
/// `convert_array_to_word`
fn convert_word_to_array(word: &str) -> Word {
    word.chars().filter(|c| c.is_ascii_uppercase()).map(|c| (c as usize - 65)).collect()
}

/// Checks whether a `word` can be made using the given `letters`
/// # Arguments
/// * `word` - The vector form of the word to check
/// * `letters` - Length-26 array of the number of each letter in the hand
/// # Returns
/// * `bool` - Whether `word` can be made using `letters`
fn is_makeable(word: &Word, letters: &Letters) -> bool {
    let mut available_letters = letters.clone();
    for letter in word.iter() {
        if unsafe { available_letters.get_unchecked(*letter) } == &0 {
            return false;
        }
        let elem = unsafe { available_letters.get_unchecked_mut(*letter) };
        *elem -= 1;
    }
    return true;
}

/// Checks that a `board` is valid after a word is played horizontally, given the specified list of `valid_word`s
/// Note that this does not check if all words are contiguous; this condition must be enforced elsewhere.
/// # Arguments
/// * `board` - `Board` being checked
/// * `min_col` - Minimum x (column) index of the subsection of the `board` to be checked
/// * `max_col` - Maximum x (column) index of the subsection of the `board` to be checked
/// * `min_row` - Minimum y (row) index of the subsection of the `board` to be checked
/// * `max_row` - Maximum y (row) index of the subsection of the `board` to be checked
/// * `row` - Row of the word played
/// * `start_col` - Starting column of the word played
/// * `end_col` - Ending column of the word played
/// * `valid_words` - HashSet of all valid words as `Vec<usize>`s
/// # Returns
/// `bool` - whether the given `board` is made only of valid words
fn is_board_valid_horizontal(board: &Board, min_col: usize, max_col: usize, min_row: usize, max_row: usize, row: usize, start_col: usize, end_col: usize, valid_words: &HashSet<Word>) -> bool {
    let mut current_letters: Vec<usize> = Vec::with_capacity(MAX_WORD_LENGTH);
    // Find the furtherest left column that the new play is connected to
    let mut minimum_col = start_col;
    while minimum_col > min_col {
        if board.get_val(row, minimum_col) == EMPTY_VALUE {
            minimum_col += 1;
            break;
        }
        minimum_col -= 1;
    }
    minimum_col = cmp::max(minimum_col, min_col);
    // Check across the row where the word was played
    for col_idx in minimum_col..max_col+1 {
        // If we're not at an empty square, add it to the current word we're looking at
        if board.get_val(row, col_idx) != EMPTY_VALUE {
            current_letters.push(board.get_val(row, col_idx));
        }
        else {
            // Turns out that checking with a set is faster than using a trie, at least for smaller hands
            if current_letters.len() > 1 && !valid_words.contains(&current_letters) {
                return false;
            }
            current_letters.clear();
            if col_idx > end_col {
                break;
            }
        }
    }
    if current_letters.len() > 1 && !valid_words.contains(&current_letters) {
        return false;
    }
    // Check down each column where a letter was played
    for col_idx in start_col..end_col+1 {
        current_letters.clear();
        // Find the furtherest up row that the word is connected to
        let mut minimum_row = row;
        while minimum_row > min_row {
            if board.get_val(minimum_row, col_idx) == EMPTY_VALUE {
                minimum_row += 1;
                break;
            }
            minimum_row -= 1;
        }
        minimum_row = cmp::max(minimum_row, min_row);
        for row_idx in minimum_row..max_row+1 {
            if board.get_val(row_idx, col_idx) != EMPTY_VALUE {
                current_letters.push(board.get_val(row_idx, col_idx));
            }
            else {
                if current_letters.len() > 1 && !valid_words.contains(&current_letters) {
                    return false;
                }
                current_letters.clear();
                if row_idx > row {
                    break;
                }
            }
        }
        if current_letters.len() > 1 && !valid_words.contains(&current_letters) {
            return false;
        }
    }
    return true;
}

/// Checks that a `board` is valid after a word is played vertically, given the specified list of `valid_word`s
/// Note that this does not check if all words are contiguous; this condition must be enforced elsewhere.
/// # Arguments
/// * `board` - `Board` being checked
/// * `min_col` - Minimum x (column) index of the subsection of the `board` to be checked
/// * `max_col` - Maximum x (column) index of the subsection of the `board` to be checked
/// * `min_row` - Minimum y (row) index of the subsection of the `board` to be checked
/// * `max_row` - Maximum y (row) index of the subsection of the `board` to be checked
/// * `start_row` - Starting row of the word played
/// * `end_row` - Ending row of the word played
/// * `col` - Column of the word played
/// * `valid_words` - HashSet of all valid words as `Vec<usize>`s
/// # Returns
/// `bool` - whether the given `board` is made only of valid words
fn is_board_valid_vertical(board: &Board, min_col: usize, max_col: usize, min_row: usize, max_row: usize, start_row: usize, end_row: usize, col: usize, valid_words: &HashSet<Word>) -> bool {
    let mut current_letters: Vec<usize> = Vec::with_capacity(MAX_WORD_LENGTH);
    // Find the furtherest up row that the new play is connected to
    let mut minimum_row = start_row;
    while minimum_row > min_row {
        if board.get_val(minimum_row, col) == EMPTY_VALUE {
            minimum_row += 1;
            break;
        }
        minimum_row -= 1;
    }
    minimum_row = cmp::max(minimum_row, min_row);
    // Check down the column where the word was played
    for row_idx in minimum_row..max_row+1 {
        // If it's not an empty value, add it to the current word
        if board.get_val(row_idx, col) != EMPTY_VALUE {
            current_letters.push(board.get_val(row_idx, col));
        }
        else {
            // Otherwise, check if we have more than one letter - if so, check if the word is valid
            if current_letters.len() > 1 && !valid_words.contains(&current_letters) {
                return false;
            }
            current_letters.clear();
            // If we're past the end of the played word, no need to check farther
            if row_idx > end_row {
                break;
            }
        }
    }
    // In case we don't hit the `else` in the previous loop
    if current_letters.len() > 1 {
        if !valid_words.contains(&current_letters) {
            return false;
        }
    }
    // Check across each row where a letter was played
    for row_idx in start_row..end_row+1 {
        current_letters.clear();
        // Find the furtherest left column that the word is connected to
        let mut minimum_col = col;
        while minimum_col > min_col {
            if board.get_val(row_idx, minimum_col) == EMPTY_VALUE {
                minimum_col += 1;
                break;
            }
            minimum_col -= 1;
        }
        minimum_col = cmp::max(minimum_col, min_col);
        for col_idx in minimum_col..max_col+1 {
            if board.get_val(row_idx, col_idx) != EMPTY_VALUE {
                current_letters.push(board.get_val(row_idx, col_idx));
            }
            else {
                if current_letters.len() > 1 && !valid_words.contains(&current_letters) {
                    return false;
                }
                current_letters.clear();
                if col_idx > col {
                    break;
                }
            }
        }
        if current_letters.len() > 1 && !valid_words.contains(&current_letters) {
            return false;
        }
    }
    return true;
}

/// Enumeration of how many letters have been used
#[derive(Copy, Clone)]
enum LetterUsage {
    /// There are still unused letters
    Remaining,
    /// More letters have been used than are available
    Overused,
    /// All letters have been used
    Finished
}
impl fmt::Display for LetterUsage {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
       match self {
            LetterUsage::Remaining => write!(f, "Remaining"),
            LetterUsage::Overused => write!(f, "Overused"),
            LetterUsage::Finished => write!(f, "Finished")
       }
    }
}
impl fmt::Debug for LetterUsage {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
             LetterUsage::Remaining => write!(f, "Remaining"),
             LetterUsage::Overused => write!(f, "Overused"),
             LetterUsage::Finished => write!(f, "Finished")
        }
     }
}

/// Enumeration of the direction a word is played
#[derive(Copy, Clone, PartialEq)]
enum Direction {
    /// The word was played horizontally
    Horizontal,
    /// The word was played vertically
    Vertical
}
impl fmt::Display for Direction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
       match self {
            Direction::Vertical => write!(f, "Horizontal"),
            Direction::Horizontal => write!(f, "Vertical")
       }
    }
}

/// Plays a word on the board
/// # Arguments
/// * `word` - The word to be played
/// * `row_idx` - The starting row at which to play the word
/// * `col_idx` - The starting column at which to play the word
/// * `board` - The current board (is modified in-place)
/// * `direction` - The `Direction` in which to play the word
/// * `letters` - The number of each letter currently in the hand
/// * `letters_on_board` - The number of each letter on the board (is modified in-place)
/// # Returns
/// *`Result` with:*
/// * `bool` - Whether the word could be validly played
/// * `Vec<(usize, usize)>` - Vector of the indices played in `board`
/// * `[usize; 26]`- The remaining letters
/// * `LetterUsage` - How many letters were used
/// 
/// *or empty `Err` if out-of-bounds*
fn play_word(word: &Word, row_idx: usize, col_idx: usize, board: &mut Board, direction: Direction, letters: &Letters, letters_on_board: &mut Letters) -> Result<(bool, Vec<(usize, usize)>, [usize; 26], LetterUsage), ()> {
    let mut played_indices: Vec<(usize, usize)> = Vec::with_capacity(MAX_WORD_LENGTH);
    match direction {
        Direction::Horizontal => {
            if col_idx + word.len() >= BOARD_SIZE {
                return Err(());
            }
            let mut remaining_letters = letters.clone();
            // Check if the word will start or end at a letter
            let mut valid_loc = (col_idx != 0 && board.get_val(row_idx, col_idx-1) != EMPTY_VALUE) || (BOARD_SIZE-col_idx <= word.len() && board.get_val(row_idx, col_idx+word.len()) != EMPTY_VALUE);
            // Check if the word will border any letters on the top or bottom
            valid_loc |= (col_idx..col_idx+word.len()).any(|c_idx| (row_idx < BOARD_SIZE-1 && board.get_val(row_idx+1, c_idx) != EMPTY_VALUE) || (row_idx > 0 && board.get_val(row_idx-1, c_idx) != EMPTY_VALUE));
            if !valid_loc {
                return Ok((false, played_indices, remaining_letters, LetterUsage::Remaining));
            }
            else {
                let mut entirely_overlaps = true;
                for i in 0..word.len() {
                    if board.get_val(row_idx, col_idx+i) == EMPTY_VALUE {
                        board.set_val(row_idx, col_idx+i, word[i]);
                        letters_on_board[word[i]] += 1;
                        played_indices.push((row_idx, col_idx+i));
                        entirely_overlaps = false;
                        let elem = unsafe { remaining_letters.get_unchecked_mut(word[i]) };
                        if *elem == 0 {
                            return Ok((false, played_indices, remaining_letters, LetterUsage::Overused));
                        }
                        *elem -= 1;
                    }
                    else if board.get_val(row_idx, col_idx+i) != word[i] {
                        return Ok((false, played_indices, remaining_letters, LetterUsage::Remaining));
                    }
                }
                if remaining_letters.iter().all(|count| *count == 0) && !entirely_overlaps {
                    return Ok((true, played_indices, remaining_letters, LetterUsage::Finished));
                }
                else {
                    return Ok((!entirely_overlaps, played_indices, remaining_letters, LetterUsage::Remaining));
                }
            }
        },
        Direction::Vertical => {
            if row_idx + word.len() >= BOARD_SIZE {
                return Err(());
            }
            let mut remaining_letters = letters.clone();
            // Check if the word will start or end at a letter
            let mut valid_loc = (row_idx != 0 && board.get_val(row_idx-1, col_idx) != EMPTY_VALUE) || (BOARD_SIZE-row_idx <= word.len() && board.get_val(row_idx+word.len(), col_idx) != EMPTY_VALUE);
            // Check if the word will border any letters on the right or left
            valid_loc |= (row_idx..row_idx+word.len()).any(|r_idx| (col_idx < BOARD_SIZE-1 && board.get_val(r_idx, col_idx+1) != EMPTY_VALUE) || (col_idx > 0 && board.get_val(r_idx, col_idx-1) != EMPTY_VALUE));
            if !valid_loc {
                return Ok((false, played_indices, remaining_letters, LetterUsage::Remaining));
            }
            else {
                let mut entirely_overlaps = true;
                for i in 0..word.len() {
                    if board.get_val(row_idx+i, col_idx) == EMPTY_VALUE {
                        board.set_val(row_idx+i, col_idx, word[i]);
                        letters_on_board[word[i]] += 1;
                        played_indices.push((row_idx+i, col_idx));
                        entirely_overlaps = false;
                        let elem = unsafe { remaining_letters.get_unchecked_mut(word[i]) };
                        if *elem == 0 {
                            return Ok((false, played_indices, remaining_letters, LetterUsage::Overused));
                        }
                        *elem -= 1;
                    }
                    else if board.get_val(row_idx+i, col_idx) != word[i] {
                        return Ok((false, played_indices, remaining_letters, LetterUsage::Remaining));
                    }
                }
                if remaining_letters.iter().all(|count| *count == 0) && !entirely_overlaps {
                    return Ok((true, played_indices, remaining_letters, LetterUsage::Finished));
                }
                else {
                    return Ok((!entirely_overlaps, played_indices, remaining_letters, LetterUsage::Remaining));
                }
            }
        }
    }
}

/// Checks which words can be played after the first
/// # Arguments
/// * `letters` - Length-26 array of originally available letters
/// * `word_being_checked` - Word that is being checked if playable
/// * `played_on_board` - Set of the letters played on the board
/// # Returns
/// * `bool` - Whether the `word_being_checked` is playable
fn check_filter_after_play(mut letters: Letters, word_being_checked: &Word, played_on_board: &HashSet<&usize>) -> bool {
    let mut already_seen_negative = false;
    for letter in word_being_checked.iter() {
        let elem = unsafe { letters.get_unchecked_mut(*letter) };
        if *elem == 0 && !played_on_board.contains(letter) {
            return false;
        }
        else if *elem <= 0 && already_seen_negative {
            return false;
        }
        else if *elem == 0 {
            already_seen_negative = true;
        }
        else {
            *elem -= 1;
        }
    }
    return true;
}

/// Removes words that can't be played with `current_letters` plus a set number of `board_letters`
/// # Arguments
/// * `current_letters` - Letters currently available in the hand
/// * `board_letters` - Letters played on the board
/// * `word_being_checked` - Word to check if it contains the appropriate number of letters
/// # Returns
/// * `bool` - Whether `word_being_checked` should pass the filter
fn check_filter_after_play_later(mut current_letters: Letters, mut board_letters: Letters, word_being_checked: &Word) -> bool {
    let mut num_from_board = 0u8;
    for letter in word_being_checked.iter() {
        let num_in_hand = unsafe { current_letters.get_unchecked_mut(*letter)};
        if *num_in_hand == 0 {
            if num_from_board == FILTER_LETTERS_ON_BOARD {
                return false;
            }
            let num_on_board = unsafe { board_letters.get_unchecked_mut(*letter)};
            if *num_on_board == 0 {
                return false;
            }
            *num_on_board -= 1;
            num_from_board += 1;
        }
        else {
            *num_in_hand -= 1;
        }
    }
    return true;
}

/// Undoes a play on the `board`
/// # Arguments
/// * `board` - `Board` being undone (is modified in-place)
/// * `played_indices` - Vector of the indices in `board` that need to be reset
/// * `letters_on_board` - Length-26 array of the number of each letter on the board (is modified in place)
fn undo_play(board: &mut Board, played_indices: &Vec<(usize, usize)>, letters_on_board: &mut Letters) {
    for index in played_indices.iter() {
        letters_on_board[board.get_val(index.0, index.1)] -= 1;
        board.set_val(index.0, index.1, EMPTY_VALUE);
    }
}

/// Recursively solves Bananagrams
/// # Arguments
/// * `board` - The `Board` to modify in-place
/// * `min_col` - Minimum occupied column index in `board`
/// * `max_col` - Maximum occupied column index in `board`
/// * `min_row` - Minimum occupied row index in `board`
/// * `max_row` - Maximum occupied row index in `board`
/// * `valid_words_vec` - Vector of vectors, each representing a word (see `convert_word_to_array`)
/// * `valid_words_set` - HashSet of vectors, each representing a word (a HashSet version of `valid_words_vec` for faster membership checking)
/// * `letters` - Length-26 array of the number of each letter in the hand
/// * `depth` - Depth of the current recursive call
/// * `words_checked` - The number of words checked in total
/// * `letters_on_board` - Length-26 array of the number of each letter currently present on the `board`
/// # Returns
/// *`Result` with:*
/// * `bool` - Whether the word could be validly played
/// * `usize` - Minimum occupied column index in `board`
/// * `usize` - Maximum occupied column index in `board`
/// * `usize` - Minimum occupied row index in `board`
/// * `usize` - Maximum occupied row index in `board`
/// 
/// *or empty `Err` on if out-of-bounds or past the maximum number of words to check*
fn play_further(board: &mut Board, min_col: usize, max_col: usize, min_row: usize, max_row: usize, valid_words_vec: Vec<&Word>, valid_words_set: &HashSet<Word>, letters: Letters, depth: usize, words_checked: &mut usize, letters_on_board: &mut Letters) -> Result<(bool, usize, usize, usize, usize), ()> {
    if *words_checked > MAXIMUM_WORDS_CHECKED {
        return Err(());
    }
    // If we're at an odd depth, play horizontally first (trying to alternate horizontal-vertical-horizontal as a heuristic to solve faster)
    if depth % 2 == 1 {
        for word in valid_words_vec.iter() {
            *words_checked += 1;
            // Try across all rows (starting from one before to one after)
            for row_idx in min_row-1..max_row+2 {
                // For each row, try across all columns (starting from the farthest out the word could be played)
                for col_idx in min_col-word.len()..max_col+2 {
                    // Using the ? because `play_word` can give an `Err` if the index is out of bounds
                    let res = play_word(word, row_idx, col_idx, board, Direction::Horizontal, &letters, letters_on_board)?;
                    if res.0 {
                        // If the word was played successfully (i.e. it's not a complete overlap and it borders at least one existing tile), then check the validity of the new words it forms
                        let new_min_col = cmp::min(min_col, col_idx);
                        let new_max_col = cmp::max(max_col, col_idx+word.len());
                        let new_min_row = cmp::min(min_row, row_idx);
                        let new_max_row = cmp::max(max_row, row_idx);
                        if is_board_valid_horizontal(board, new_min_col, new_max_col, new_min_row, new_max_row, row_idx, col_idx, col_idx+word.len()-1, valid_words_set) {
                            // If it's valid, go to the next recursive level (unless we've all the letters, at which point we're done)
                            match res.3 {
                                LetterUsage::Finished => {
                                    return Ok((true, new_min_col, new_max_col, new_min_row, new_max_row));
                                },
                                LetterUsage::Remaining => {
                                    // Another option: let new_valid_words_vec: Vec<&Word> = valid_words_vec.clone().into_iter().filter(|w| check_filter_after_play_later(letters.clone(), letters_on_board.clone(), w)).collect();
                                    // I think doing it that way might be less efficient however due to the `clone` of `valid_words_vec`
                                    let mut new_valid_words_vec: Vec<&Word> = Vec::with_capacity(valid_words_vec.len()/2);
                                    for i in 0..valid_words_vec.len() {
                                        if check_filter_after_play_later(letters.clone(), letters_on_board.clone(), valid_words_vec[i]) {
                                            new_valid_words_vec.push(valid_words_vec[i]);
                                        }
                                    }
                                    let res2 = play_further(board, new_min_col, new_max_col, new_min_row, new_max_row, new_valid_words_vec, valid_words_set, res.2, depth+1, words_checked, letters_on_board)?;
                                    if res2.0 {
                                        // If that recursive stack finishes successfully, we're done! (could have used another Result or Option rather than a bool in the returned tuple, but oh well)
                                        return Ok(res2);
                                    }
                                    else {
                                        // Otherwise, undo the previous play (cloning the board before each play so we don't have to undo is *way* slower)
                                        undo_play(board, &res.1, letters_on_board);
                                    }
                                },
                                LetterUsage::Overused => unreachable!()
                            }
                        }
                        else {
                            // If the play formed some invalid words, undo the previous play
                            undo_play(board, &res.1, letters_on_board);
                        }
                    }
                    else {
                        // If trying to play the board was invalid, undo the play
                        undo_play(board, &res.1, letters_on_board);
                    }
                }
            }
        }
        // If trying every word horizontally didn't work, try vertically instead
        for word in valid_words_vec.iter() {
            *words_checked += 1;
            // Try down all columns
            for col_idx in min_col-1..max_col+2 {
                // This is analgous to the above
                for row_idx in min_row-word.len()..max_row+2 {
                    let res = play_word(word, row_idx, col_idx, board, Direction::Vertical, &letters, letters_on_board)?;
                    if res.0 {
                        let new_min_col = cmp::min(min_col, col_idx);
                        let new_max_col = cmp::max(max_col, col_idx);
                        let new_min_row = cmp::min(min_row, row_idx);
                        let new_max_row = cmp::max(max_row, row_idx+word.len());
                        if is_board_valid_vertical(board, new_min_col, new_max_col, new_min_row, new_max_row, row_idx, row_idx+word.len()-1, col_idx, valid_words_set) {
                            match res.3 {
                                LetterUsage::Finished => {
                                    return Ok((true, new_min_col, new_max_col, new_min_row, new_max_row));
                                },
                                LetterUsage::Remaining => {
                                    let mut new_valid_words_vec: Vec<&Word> = Vec::with_capacity(valid_words_vec.len()/2);
                                    for i in 0..valid_words_vec.len() {
                                        if check_filter_after_play_later(letters.clone(), letters_on_board.clone(), valid_words_vec[i]) {
                                            new_valid_words_vec.push(valid_words_vec[i]);
                                        }
                                    }
                                    let res2 = play_further(board, new_min_col, new_max_col, new_min_row, new_max_row, new_valid_words_vec, valid_words_set, res.2, depth+1, words_checked, letters_on_board)?;
                                    if res2.0 {
                                        return Ok(res2);
                                    }
                                    else {
                                        undo_play(board, &res.1, letters_on_board);
                                    }
                                },
                                LetterUsage::Overused => unreachable!()
                            }
                        }
                        else {
                            undo_play(board, &res.1, letters_on_board);
                        }
                    }
                    else {
                        undo_play(board, &res.1, letters_on_board);
                    }
                }
            }
        }
        return Ok((false, min_col, max_col, min_row, max_row));
    }
    // If we're at an even depth, play vertically first. Otherwise this is analgous to the above.
    else {
        for word in valid_words_vec.iter() {
            *words_checked += 1;
            // Try down all columns
            for col_idx in min_col-1..max_col+2 {
                for row_idx in min_row-word.len()..max_row+2 {
                    let res = play_word(word, row_idx, col_idx, board, Direction::Vertical, &letters, letters_on_board)?;
                    if res.0 {
                        let new_min_col = cmp::min(min_col, col_idx);
                        let new_max_col = cmp::max(max_col, col_idx);
                        let new_min_row = cmp::min(min_row, row_idx);
                        let new_max_row = cmp::max(max_row, row_idx+word.len());
                        if is_board_valid_vertical(board, new_min_col, new_max_col, new_min_row, new_max_row, row_idx, row_idx+word.len()-1, col_idx, valid_words_set) {
                            match res.3 {
                                LetterUsage::Finished => {
                                    return Ok((true, new_min_col, new_max_col, new_min_row, new_max_row));
                                },
                                LetterUsage::Remaining => {
                                    let mut new_valid_words_vec: Vec<&Word> = Vec::with_capacity(valid_words_vec.len()/2);
                                    for i in 0..valid_words_vec.len() {
                                        if check_filter_after_play_later(letters.clone(), letters_on_board.clone(), valid_words_vec[i]) {
                                            new_valid_words_vec.push(valid_words_vec[i]);
                                        }
                                    }
                                    let res2 = play_further(board, new_min_col, new_max_col, new_min_row, new_max_row, new_valid_words_vec, valid_words_set, res.2, depth+1, words_checked, letters_on_board)?;
                                    if res2.0 {
                                        return Ok(res2);
                                    }
                                    else {
                                        undo_play(board, &res.1, letters_on_board);
                                    }
                                },
                                LetterUsage::Overused => unreachable!()
                            }
                        }
                        else {
                            undo_play(board, &res.1, letters_on_board);
                        }
                    }
                    else {
                        undo_play(board, &res.1, letters_on_board);
                    }
                }
            }
        }
        // No point in checking horizontally for the first depth, since it would have to form a vertical word that was already checked and failed
        if depth == 0 {
            return Ok((false, min_col, max_col, min_row, max_row));
        }
        for word in valid_words_vec.iter() {
            *words_checked += 1;
            // Try across all rows
            for row_idx in min_row-1..max_row+2 {
                for col_idx in min_col-word.len()..max_col+2 {
                    let res = play_word(word, row_idx, col_idx, board, Direction::Horizontal, &letters, letters_on_board)?;
                    if res.0 {
                        let new_min_col = cmp::min(min_col, col_idx);
                        let new_max_col = cmp::max(max_col, col_idx+word.len());
                        let new_min_row = cmp::min(min_row, row_idx);
                        let new_max_row = cmp::max(max_row, row_idx);
                        if is_board_valid_horizontal(board, new_min_col, new_max_col, new_min_row, new_max_row, row_idx, col_idx, col_idx+word.len()-1, valid_words_set) {
                            match res.3 {
                                LetterUsage::Finished => {
                                    return Ok((true, new_min_col, new_max_col, new_min_row, new_max_row));
                                },
                                LetterUsage::Remaining => {
                                    let mut new_valid_words_vec: Vec<&Word> = Vec::with_capacity(valid_words_vec.len()/2);
                                    for i in 0..valid_words_vec.len() {
                                        if check_filter_after_play_later(letters.clone(), letters_on_board.clone(), valid_words_vec[i]) {
                                            new_valid_words_vec.push(valid_words_vec[i]);
                                        }
                                    }
                                    let res2 = play_further(board, new_min_col, new_max_col, new_min_row, new_max_row, new_valid_words_vec, valid_words_set, res.2, depth+1, words_checked, letters_on_board)?;
                                    if res2.0 {
                                        return Ok(res2);
                                    }
                                    else {
                                        undo_play(board, &res.1, letters_on_board);
                                    }
                                },
                                LetterUsage::Overused => unreachable!()
                            }
                        }
                        else {
                            undo_play(board, &res.1, letters_on_board);
                        }
                    }
                    else {
                        undo_play(board, &res.1, letters_on_board);
                    }
                }
            }
        }
        return Ok((false, min_col, max_col, min_row, max_row));
    }
}

/// Plays a new bananagrams board using the given letters and dictionary
/// # Arguments
/// * `available_letters` - Array of the number of each letter to play with
/// * `dictionary` - Vector of vectors representing valid words
/// # Returns
/// * `Option`
///     * `None` - If no valid play was possible
///     * `Some` - If successful, a tuple of (the board solution, minimum column, maximum column, minimum row, maximum row)
fn play_bananagrams(available_letters: [usize; 26], dictionary: &Vec<Word>) -> Option<(Board, usize, usize, usize, usize)> {
    // Get a vector of all valid words
    let valid_words_vec: Vec<Word> = dictionary.iter().filter(|word| is_makeable(word, &available_letters)).map(|word| word.clone()).collect();
    if valid_words_vec.len() == 0 {
        return None;
    }
    let mut words_checked = 0;
    // Loop through each word and play it on a new board
    for (word_num, word) in valid_words_vec.iter().enumerate() {
        words_checked += 1;
        let mut board = Board::new();
        let col_start = BOARD_SIZE/2 - word.len()/2;
        let row = BOARD_SIZE/2;
        let mut use_letters: [usize; 26] = available_letters.clone();
        let mut letters_on_board = [0usize; 26];
        for i in 0..word.len() {
            board.set_val(row, col_start+i, word[i]);
            letters_on_board[word[i]] += 1;
            use_letters[word[i]] -= 1;  // Should never underflow because we've verified that every word is playable with these letters
        }
        let min_col = col_start;
        let min_row = row;
        let max_col = col_start + (word.len()-1);
        let max_row = row;
        if use_letters.iter().all(|count| *count == 0) {
            return Some((board.clone(), min_col, max_col, min_row, max_row));
        }
        else {
            // Reduce the set of remaining words to check to those that can be played with the letters not in the first word (plus only one of the tiles played in the first word)
            let word_letters: HashSet<&usize> = HashSet::from_iter(word.iter());
            let mut new_valid_words_vec: Vec<&Word> = Vec::with_capacity(valid_words_vec.len()-word_num);
            for i in word_num..valid_words_vec.len() {
                if check_filter_after_play(use_letters.clone(), &valid_words_vec[i], &word_letters) {
                    new_valid_words_vec.push(&valid_words_vec[i]);
                }
            }
            let valid_words_set: HashSet<Word> = HashSet::from_iter(valid_words_vec.iter().cloned());
            // Begin the recursive processing
            let result = play_further(&mut board, min_col, max_col, min_row, max_row, new_valid_words_vec, &valid_words_set, use_letters, 0, &mut words_checked, &mut letters_on_board);
            match result {
                // If the result was good, then we're done (otherwise we continue)
                Ok(res) => {
                    if res.0 {
                        return Some((board.clone(), res.1, res.2, res.3, res.4));
                    }
                },
                // If an error (we're out of bounds or we've reached the maximum number of iterations) then we continue
                Err(()) => {}
            }
        }
    }
    None
}

/// Generates a random hand of letters pulled from the entire set of Bananagrams tiles
/// # Arguments
/// * `rng` - Thread random number generator
/// # Returns
/// * `[usize; 26]` - Number of each letter present in the hand
fn generate_hand(rng: &mut ThreadRng) -> Letters {    
    // Calculate the logarithmic scaled value within [min, max]
    let scaled_value = (MAXIMUM_HAND_SIZE - MINIMUM_HAND_SIZE) * (BASE.powf(rng.gen()) - 1.0) / (BASE - 1.0) + MINIMUM_HAND_SIZE;
    
    // Convert to an integer
    let size = scaled_value.round() as usize;
    let mut letters = [0usize; 26];
    TO_CHOOSE_FROM.choose_multiple(rng, size).for_each(|c| {
        letters[(*c) - 65] += 1;
    });
    letters
}

/// Converts the `board` to a bytes representation for saving
/// # Arguments
/// * `board` - Board to save
/// * `min_col` - Minimum column with letters
/// * `max_col` - Maximum column with letters
/// * `min_row` - Minimum row with letters
/// * `max_row` - Maximum row with letters
/// # Returns
/// * `Vec<u8>` - Vector where each non-empty cell on the `board` is represented by the \[row index, column index, letter value\],
/// with all letters in succession. At the end will always be 255 (to serve as the demarcation between boards when saving).
fn board_to_bytes(board: &Board, min_col: usize, max_col: usize, min_row: usize, max_row: usize) -> Vec<u8> {
    let mut board_bytes: Vec<u8> = Vec::with_capacity((max_row-min_row)*(max_col-min_col));
    for row in min_row..max_row+1 {
        for col in min_col..max_col+1 {
            if board.get_val(row, col) != EMPTY_VALUE {
                board_bytes.push(row as u8);
                board_bytes.push(col as u8);
                board_bytes.push(board.get_val(row, col) as u8);
            }
        }
    }
    board_bytes.push(255);
    board_bytes
}

fn main() {
    let mut dictionary: Vec<Word> = include_str!("../../new_short_dictionary.txt").lines().map(convert_word_to_array).collect();
    dictionary.sort_by(|w1, w2| w2.len().cmp(&w1.len()));
    let mut default_parallelism_approx = 1usize;
    match thread::available_parallelism() {
        Ok(available_parallelism) => {
            default_parallelism_approx = available_parallelism.into();
        },
        _ => {}
    }
    const NUMBER_OF_BOARDS_TO_GENERATE: usize = 500;
    (0..default_parallelism_approx).into_par_iter().for_each(|thread_num| {
        let mut rng = thread_rng();
        let mut boards_generated: usize = 0;
        let mut all_board_bytes: Vec<u8> = Vec::new();
        while boards_generated < NUMBER_OF_BOARDS_TO_GENERATE {
            let letters = generate_hand(&mut rng);
            let res = play_bananagrams(letters, &dictionary);
            match res {
                Some(result) => {
                    all_board_bytes.extend(board_to_bytes(&result.0, result.1, result.2, result.3, result.4));
                    boards_generated += 1;
                    if boards_generated % 50 == 0 {
                        println!("Thread {} has generated {}", thread_num, boards_generated);
                    }
                },
                None => {/* Continue without incrementing since we failed to make a board */}
            }
        }
        fs::write(format!("data/{}_board4.bgb", thread_num), all_board_bytes).expect("Failed to write board data!");
    });
    
    // let letters = "EEEHILNNOOOQSTTTTUUWZ"; //"AAAACDEGIILLLNNNNNOSTTTUUVVWYZ"; //"CEEHHKLMMNOOOOSSTUVXZ"; //"CCEEEGHIIINNOOPRRSSSSSTTTTTWX"; //"CCEEEGHIIINNOOPRRSSTTTTWX";
    // let mut vals = [0usize; 26];
    // for c in letters.chars() {
    //     vals[c as usize - 65] += 1;
    // }
    // let now = std::time::Instant::now();
    // let res = play_bananagrams(vals, &dictionary);
    // match res {
    //     Some(result) => {
    //         println!("{}", board_to_string(&result.0, result.1, result.2, result.3, result.4));
    //     },
    //     None => println!("Failed!")
    // }
    // println!("{:?}", now.elapsed());
}
