If you grab a screenshot of a wordle board (and only the wordle board), returns words most likely to be next.

Best way to do this on windows is to use Win + Shift + S

*browser plugins that change the webpage colors will break it

This script uses logic cut and refined from my babble script, with the logic steps as follows:
1. Acquire image from clipboard
2. Find board grid lines
  - gets a black and white representation of the board of colors that are the background or not
  - gets the total of pixels for all rows and columns for x and y
  - there is a greater number of pixels on each gridline, gets, separately, the x and y positions for these
  - gets the average position of each grouping, discards the edges/dead space
  - gets the linear (mx + b) relationship for each x and y
  - returns arrays, for x and y directions, of each grid line's position, of those that surround the game squares
3. Analyze the game board to return tiles and tile types in format relevant to the game/puzzle
  - takes the prior generated arrays as inputs to find and do logic on each game square
  - checks to see how many background pixels are in square, too many means it's empty
  - if not empty, creates a copy of the square as a black and white representation, with threshold to only white the text
  - b/w square is normalized to only contain the text (stretched so that white pixels touch each edge of a 25x25 square)
  - opencv logic with redundancy compares b/w square to existing normalized alphabet images, returning the letter of the most similar
  - seeing the color of each square, puts the letter result into the relevant output arrays
4. Takes the prior output arrays as input, generating a regex string to filter wordset
  - a sorting pass is done, using a pre-generated dataset to score the value of each letter in the word (avoiding lower-value repeats)
    - generated dataset is the frequency of each letter of the alphabet in each specific indexed position in all 5 letter words in the word dataset
5. Prints the result, which is a list of words, with the most likely words to come next as first in the list
