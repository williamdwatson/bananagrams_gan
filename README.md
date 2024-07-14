# Bananagrams GAN
This project explores the use of a generative adversarial network to produce Bananagrams boards given a starting hand.

## Sources
Several English dictionaries of "common" words were combined to generate a dictionary of acceptable words. These include:
* Lists from https://people.sc.fsu.edu/~jburkardt/datasets/words/words.html under the [LGPL license](https://www.gnu.org/licenses/lgpl-3.0.en.html#license-text):
    * basic_english_850.txt
    * basic_english_2000.txt
    * doublet_words.txt
    * globish.txt
    * simplified_english.txt
    * special_english.txt
    * unique_grams.txt
* Lists from https://github.com/MichaelWehar/Public-Domain-Word-Lists in the public domain:
    * 200-less-common.txt
    * 5000-more-common.txt
* MIT's 10000 word list (https://www.mit.edu/~ecprice/wordlist.10000)
    * wordlist.10000.txt
* 10000 word list derived from Google (https://github.com/first20hours/google-10000-english)
    * google-10000-english.txt

The base dictionary was [short_dictionary.txt](https://github.com/williamdwatson/bananagrams_solver/blob/d0adc0d61f1de7c3fc9d41047f560f1473bc4def/src-tauri/src/short_dictionary.txt) from a previous Bananagrams solver project of mine. I combined that dictionary with the ones listed here using `combine_dictionary.ipynb` to form `new_short_dictionary.txt`; some manual editing was performed as well.

All source dictionary text files are stored in `dictionaries.tar.gz`, and the derived one is present in the repository root.
