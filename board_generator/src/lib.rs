use pyo3::prelude::*;
use rand::prelude::*;
use rand::distributions::Standard;
use array2d::Array2D;
use std::collections::{HashSet, VecDeque};

type Board = Array2D<usize>;

/// Dimensions of the board
const BOARD_SIZE: usize = 144;
/// The maximum length of any word in the dictionary
const MAX_WORD_LENGTH: usize = 17;

#[derive(Copy, Clone)]
enum Direction {
    Horizontal,
    Vertical,
}
impl Distribution<Direction> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Direction {
        match rng.gen_range(0..=1) {
            0 => Direction::Horizontal,
            1 => Direction::Vertical
        }
    }
}
impl Direction {
    /// Gets the opposite direction of the current (i.e. vertical -> horizontal or vice versa)
    fn opposite(self) -> Direction {
        match self {
            Direction::Horizontal => Direction::Vertical,
            Direction::Vertical => Direction::Horizontal
        }
    }
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
fn is_board_valid_horizontal(board: &Board, min_col: usize, max_col: usize, min_row: usize, max_row: usize, row: usize, start_col: usize, end_col: usize, valid_words: &HashSet<Vec<usize>>) -> bool {
    let mut current_letters: Vec<usize> = Vec::with_capacity(MAX_WORD_LENGTH);
    // Check across the row where the word was played
    for col_idx in min_col..max_col+1 {
        // If we're not at an empty square, add it to the current word we're looking at
        if board[(row, col_idx)] != 0 {
            current_letters.push(board[(row, col_idx)]);
        }
        else {
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
        for row_idx in min_row..max_row+1 {
            if board[(row_idx, col_idx)] != 0 {
                current_letters.push(board[(row_idx, col_idx)]);
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
fn is_board_valid_vertical(board: &Board, min_col: usize, max_col: usize, min_row: usize, max_row: usize, start_row: usize, end_row: usize, col: usize, valid_words: &HashSet<Vec<usize>>) -> bool {
    let mut current_letters: Vec<usize> = Vec::with_capacity(MAX_WORD_LENGTH);
    // Check down the column where the word was played
    for row_idx in min_row..max_row+1 {
        // If it's not an empty value, add it to the current word
        if board[(row_idx, col)] != 0 {
            current_letters.push(board[(row_idx, col)]);
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
        for col_idx in min_col..max_col+1 {
            if board[(row_idx, col_idx)] != 0 {
                current_letters.push(board[(row_idx, col_idx)]);
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

/// Plays a word on the `board` (modifying it in-place)
/// # Arguments
/// * `board` - Array2D board to change in-place
/// * `word` - Word to play represented as a vector of numbers
/// * `dir` - Direction to play the `word`
/// * `start_x` - x position of the `word`'s first letter
/// * `start_y` - y position of the `word`'s first letter
/// * `played_positions` - HashSet of previously played (x, y) positions; will be modified in-place to add newly played positions
fn play_word(board: &mut Board, word: &Vec<usize>, dir: Direction, start_x: usize, start_y: usize, played_positions: &mut HashSet<(usize, usize)>) {
    match dir {
        Direction::Horizontal => {
            for i in 0..word.len() {
                board[(i+start_x, start_y)] = word[i];
                played_positions.insert((i+start_x, start_y));
            }
        },
        Direction::Vertical => {
            for i in 0..word.len() {
                board[(start_x, start_y+i)] = word[i];
                played_positions.insert((i+start_x, start_y));
            }
        }
    }
}

fn generate_board(dictionary: &Vec<Vec<usize>>, target_size: usize) -> Option<Board> {
    let mut board: Board = Array2D::filled_with(0, BOARD_SIZE, BOARD_SIZE);
    let mut rng = thread_rng();
    if let Some(start_word) = dictionary.iter().filter(|w| w.len() <= target_size).choose(&mut rng) {
        // Play the first word in a random direction in the middle of the board
        let mut dir: Direction = rand::random();
        let mid = BOARD_SIZE/2;
        let mut played_positions = HashSet::new();
        let (start_x, start_y) = match dir {
            Direction::Horizontal => (BOARD_SIZE/2, BOARD_SIZE/2 - start_word.len()/2),
            Direction::Vertical => (BOARD_SIZE/2 - start_word.len()/2, BOARD_SIZE/2)
        };
        play_word(&mut board, &start_word, dir, start_x, start_y, &mut played_positions);
        // If the word chosen was the target length, we're done
        if start_word.len() == target_size {
            return Some(board);
        }
        // Otherwise, play the second word at a random location in the opposite direction
        dir = dir.opposite();
        let second_pos = played_positions.iter().choose(&mut rng).unwrap();
        let second_pos_letter = board[*second_pos];
        // Choose a random word that overlaps
        let word = dictionary.iter().filter(|w| w.contains(&second_pos_letter)).choose(&mut rng).unwrap();
        // Choose a random position of overlapping
        let pos = word.iter().enumerate().filter_map(|(idx, c)| if *c == second_pos_letter { Some(idx) } else { None }).choose(&mut rng).unwrap();
        // Play the word
        match dir {
            Direction::Horizontal => play_word(&mut board, &word, dir, second_pos.0-pos, second_pos.1, &mut played_positions),
            Direction::Vertical => play_word(&mut board, &word, dir, second_pos.0, second_pos.1-pos, &mut played_positions)
        }
        // If we've reached the target size, we're done
        if played_positions.len() >= target_size {
            return Some(board);
        }
        // Otherwise, keep trying until we hit the proper size
        while played_positions.len() < target_size {
            'outer: loop {
                dir = rand::random();
                let play_pos = played_positions.iter().choose(&mut rng).unwrap();
                let play_letter = board[*play_pos];
                // Choose a random word that overlaps
                let word = dictionary.iter().filter(|w| w.contains(&play_letter)).choose(&mut rng).unwrap();
                // Choose a random position of overlapping 
                let mut possible_positions: Vec<usize> = word.iter().enumerate().filter_map(|(idx, c)| if *c == play_letter { Some(idx) } else { None }).collect();
                possible_positions.shuffle(&mut rng);
                for pos in possible_positions {
                    let success = match dir {
                        Direction::Horizontal => play_word(&mut board, &word, dir, second_pos.0-pos, second_pos.1, &mut played_positions),
                        Direction::Vertical => play_word(&mut board, &word, dir, second_pos.0, second_pos.1-pos, &mut played_positions)
                    };
                    if success {
                        break 'outer;
                    }
                }
            }
            let played_vec: Vec<&(usize, usize)> = played_positions.iter().collect();
            let second_pos = played_positions.iter().choose(&mut rng).unwrap();
            let second_pos_letter = board[*second_pos];
            // Choose a random word that overlaps
            let word = dictionary.iter().filter(|w| w.contains(&second_pos_letter)).choose(&mut rng).unwrap();
        }
        Some(board)
    }
    else {
        None
    }
}

fn board_to_string(board: &Array2D<char>) -> String {
    let mut s = "".to_string();
    for i in 0..board.num_rows() {
        for j in 0..board.num_columns() {
            if board[(i, j)] == '.' {
                s += " ";
            }
            else {
                s += &board[(i, j)].to_string();
            }
        }
        s += "\n";
    }
    s
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    let dictionary = vec![
        "apple".to_string(),
        "banana".to_string(),
        "orange".to_string(),
        "grape".to_string(),
        "peach".to_string(),
        // Add more words as needed
    ];

    let target_size = 21;
    let board = generate_board(&dictionary, target_size);
    println!("{}", board_to_string(&board));
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn board_generator(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}
