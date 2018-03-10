from lm_train import *
from align_ibm1 import *
from BLEU_score import *
from log_prob import *
from preprocess import *
from math import log
from decode import *
import pickle
import os
import csv

def evalAlign(decode_file, LM, AM, *eval_files):
    


    with open(decode_file) as f:
        french_lines = f.readlines()

    
    for file in eval_files:
        
        print("\n")
        print(file + " sentences")
        print("\n")

        with open(file) as f:
            eng_file = f.readlines()
            
        for i in range(len(french_lines)):
            if i < 25:
                french = preprocess(french_lines[i], "french")
                eng_ref = preprocess(eng_file[i], "french")
                english = decode(french, LM, AM)

                for n in [1,2,3]:
                    bs = BLEU_score(eng_ref, english, n)

                    # Sentence number, n-value, bleu score
                    print(i + 1, n, bs)
                    
       


decode_file = "/u/cs401/A2_SMT/data/Hansard/Testing/Task5.f"
file1 = "/u/cs401/A2_SMT/data/Hansard/Testing/Task5.e"
file2 = "/u/cs401/A2_SMT/data/Hansard/Testing/Task5.google.e"
data_dir = "/u/cs401/A2_SMT/data/Hansard/Training/"


if os.path.exists("e_temp.pickle"):
    with open('e_temp.pickle', 'rb') as handle:
        LM = pickle.load(handle)
else:
    LM = lm_train(data_dir, 'e', 'e_temp')

for num in [1000, 10000, 15000, 30000]:
    max_iter = 5

    print("\n")
    print(str(num) + " sentences")
    print("\n")

    filename = "AM_ibm1_" + str(num) + "_" + str(max_iter)
    if os.path.exists(filename + ".pickle"):
        with open(filename + '.pickle', 'rb') as handle:
            AM = pickle.load(handle)
    else:
        AM = align_ibm1(data_dir, num, max_iter, filename)

    evalAlign(decode_file, LM, AM, file1, file2)








