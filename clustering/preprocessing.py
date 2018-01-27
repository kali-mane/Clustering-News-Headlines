# Cleaning scrapped data to remove punctuation, numbers and make it to lower case
# Write the data to a file to be used further

import re
from string import punctuation
import os


def main():

    path = os.path.abspath(os.path.dirname(__file__))
    f = open(os.path.join(path, 'data\scrapped_headlines.txt'), 'rt', encoding='utf-8')
    text_file = f.read().split('\n')

    text_lower = [text.lower() for text in text_file]
    text_letters = [''.join(c for c in s if c not in punctuation) for s in text_lower]

    text_final = [re.sub(r'[^A-Za-z]+', ' ', x) for x in text_letters]

    with open(os.path.join(path, 'data\headlines_cleaned.txt'), 'w') as fw:
        for text in text_final:
            fw.write(text)
            fw.write('\n')
    fw.close()


if __name__ == '__main__':
    main()
