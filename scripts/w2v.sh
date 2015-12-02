time ./word2vec -train ../data/forW2V.txt -output ../data/vectors.bin -cbow 1 -size 50 -window 20 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 1 -iter 15
python vec2txt.py
