Download the dataset from https://www.kaggle.com/datasets/aliabedimadiseh/grch38-human-genome-dna/data
or use any dataset thats a .fa file and pertains to base pairs of DNA, code is set to run on Colab with a google drive or you can just change file paths and run it on a computer
Idea of this is to find errors in base pairs, for example deletions would make it so that A(1) and C(3) would connect, which shouldn't happen, and it would be marked to be fixed based on what type of error it is, such as a deltion and adjusted and rewritten in the updated .fa file
