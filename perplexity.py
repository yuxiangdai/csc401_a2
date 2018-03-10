from log_prob import *
from preprocess import *
import os

def preplexity(LM, test_dir, language, smoothing = False, delta = 0):
    """
        Computes the preplexity of language model given a test corpus
        
        INPUT:
        
        LM : 		(dictionary) the language model trained by lm_train
        test_dir : 	(string) The top-level directory name containing data
                    e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
        language : `(string) either 'e' (English) or 'f' (French)
        smoothing : (boolean) True for add-delta smoothing, False for no smoothing
        delta : 	(float) smoothing parameter where 0<delta<=1
    """
    files = os.listdir(test_dir)
    pp = 0
    N = 0
    vocab_size = len(LM["uni"])
    
    for ffile in files:
        if ffile.split(".")[-1] != language:
            continue
        
        opened_file = open(test_dir+ffile, "r")
        for line in opened_file:
            processed_line = preprocess(line, language)
            tpp = log_prob(processed_line, LM, smoothing, delta, vocab_size)
            
            if tpp > float("-inf"):
                pp = pp + tpp
                N += len(processed_line.split())

            print(processed_line, tpp)
        opened_file.close()
    if N > 0:
        pp = 2**(-pp/N)
    return pp

#test

data_dir = "/Users/yuxiangdai/Documents/A2_SMT/data/Hansard/Training/"
test_data_dir = "/Users/yuxiangdai/Documents/A2_SMT/data/Hansard/Testing/"

language = "e"
fn_LM = language + "_temp"
smoothing = True

test_LM = lm_train(data_dir, language, fn_LM)
# with open(fn_LM + '.pickle', 'rb') as handle:
#     test_LM = pickle.load(handle)
print(preplexity(test_LM, test_data_dir, language, False, 0))
# print(preplexity(test_LM, test_data_dir, language, smoothing, 0.01))
# print(preplexity(test_LM, test_data_dir, language, smoothing, 0.05))
# print(preplexity(test_LM, test_data_dir, language, smoothing, 0.1))
# print(preplexity(test_LM, test_data_dir, language, smoothing, 0.2))
# print(preplexity(test_LM, test_data_dir, language, smoothing, 0.5))