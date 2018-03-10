from lm_train import *
from log_prob import *
from preprocess import *
from math import log
import os
import time



def align_ibm1(train_dir, num_sentences, max_iter, fn_AM):
    """
    Implements the training of IBM-1 word alignment algoirthm. 
    We assume that we are implemented P(foreign|english)
    
    INPUTS:
    train_dir : 	(string) The top-level directory name containing data
                    e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
    num_sentences : (int) the maximum number of training sentences to consider
    max_iter : 		(int) the maximum number of iterations of the EM algorithm
    fn_AM : 		(string) the location to save the alignment model
    
    OUTPUT:
    AM :			(dictionary) alignment model structure
    
    The dictionary AM is a dictionary of dictionaries where AM['english_word']['foreign_word'] 
    is the computed expectation that the foreign_word is produced by english_word.
    
            LM['house']['maison'] = 0.5
    """
    AM = {}
    
    # Read training data
    AM, sentence_pairs = read_hansard(train_dir, num_sentences)
    
    # Initialize AM uniformly     
    AM["SENTSTART"] = {"SENTSTART": 1}
    AM["SENTEND"] = {"SENTEND": 1}

    tcount = {}
    total = {}
    for eng_word in AM.keys():
        if checkef(eng_word, eng_word):
            total[eng_word] = 0
            for fre_word in AM[eng_word].keys(): 
                initialize(eng_word, fre_word, AM)
                if eng_word not in tcount.keys():
                
                    # tcount[fre_word] = {eng_word: 0}
                    tcount[eng_word] = {fre_word: 0}
                else:
                    tcount[eng_word][fre_word] = 0

    # Iterate between E and M steps
    #   for a number of iterations:
    for iteration in range(max_iter):
        print("iter:", iteration)
        #   set tcount(f, e) to 0 for all f, e
        for eng in tcount.keys(): 
            for fre in tcount[eng]:
                tcount[eng][fre] = 0
        #   set total(e) to 0 for all e
        for eng in total.keys():
            total[eng] = 0

        em_step(tcount, total, AM, sentence_pairs)

    with open(fn_AM + '.pickle','wb') as handle:
        pickle.dump(AM, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return AM
    
# ------------ Support functions --------------
def read_hansard(train_dir, num_sentences):
    """
    Read up to num_sentences from train_dir.
    
    INPUTS:
    train_dir : 	(string) The top-level directory name containing data
                    e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
    num_sentences : (int) the maximum number of training sentences to consider
    
    
    Make sure to preprocess!
    Remember that the i^th line in fubar.e corresponds to the i^th line in fubar.f.
    
    Make sure to read the files in an aligned manner.
    """
    # TODO
    AM = {}
    sentence_pairs = []
    num_lines = 0
    for root, dirs, files in os.walk(train_dir):
        for file in files:
            if file.endswith(".e") and file.startswith("hansard"):
                filename = file[:-2]
                if num_lines < num_sentences:
                    english_file = open(os.path.join(train_dir, filename + ".e"))
                    french_file = open(os.path.join(train_dir, filename + ".f"))
                    line = english_file.readline()
                    line_french = french_file.readline()
                    
                    while line and num_lines < num_sentences:
                        processed_line = preprocess(line, "e")
                        processed_line_french = preprocess(line_french, "f")
                        sentence_pairs.append([processed_line, processed_line_french])
                        
                        for eng_word in processed_line.split():
                            for fre_word in processed_line_french.split():
                                if checkef(eng_word, fre_word):
                                    if eng_word in AM.keys():
                                        if fre_word not in AM[eng_word].keys():
                                            AM[eng_word][fre_word] = 0    
                                    else:
                                        AM[eng_word] = {fre_word: 0}
                        line = english_file.readline()
                        line_french = french_file.readline()
                        num_lines += 1
                        print(filename, num_lines)

    return AM, sentence_pairs              


def initialize(eng, fre, AM):
    """
    Initialize alignment model uniformly.
    Only set non-zero probabilities where word pairs appear in corresponding sentences.
    """

    AM[eng][fre] = 1 / len(AM[eng])  # uniformly distribute eng_word prob



    
def em_step(tcount, total, AM, sentence_pairs):
    """
    One step in the EM algorithm.
    Follows the pseudo-code given in the tutorial slides.
    """
    # TODO



#    initialize P(f|e)
#   for a number of iterations:
#   set tcount(f, e) to 0 for all f, e
#   set total(e) to 0 for all e
#   for each sentence pair (F, E) in training corpus:


    # Start time
    start = time.time()
        
    
    for E, F in sentence_pairs:
        # convert into lists for set() and count()
        Fsplit = F.split()[1:-1]
        Esplit = E.split()[1:-1]
        for f in set(Fsplit):
            denom_c = 0
            for e in set(Esplit):
                #  denom_c += P(f|e) * F.count(f)
                denom_c += AM[e][f] * F.count(f)
                # denom_c += AM[e][f] * F.count(f)
            for e in set(Esplit):
                tcount[e][f] += AM[e][f] * F.count(f) * E.count(e) / denom_c
                total[e] += AM[e][f] * F.count(f) * E.count(e) / denom_c 
    
    end = time.time()
    seconds = end - start
    print("Time taken E : {0} seconds".format(seconds))
    start = time.time()

    for e in total.keys():     
        for f in tcount[e]:  
            AM[e][f] = tcount[e][f] / total[e]

    end = time.time()
    seconds = end - start
    print("Time taken M : {0} seconds".format(seconds))   

        #   for each e in domain(total(:)):
#     for each f in domain(tcount(:,e)):
#       P(f|e) = tcount(f, e) / total(e)

#     for each unique word f in F:
#       denom_c = 0
#       for each unique word e in E:

#       
#         tcount(f, e) += P(f|e) * F.count(f) * E.count(e) / denom_c
#         total(e) += P(f|e) * F.count(f) * E.count(e) / denom_c


def checkef(e, f):
    if e != 'SENTSTART' and e != 'SENTEND' and f != 'SENTSTART' and f != 'SENTEND':
        return True
    else:
        return False


# Testing the AM on the Toy data 
# train_dir = "/Users/yuxiangdai/Documents/A2_SMT/data/Toy/"
# num_sentences = 5
# max_iter = 5
# fn_AM = "toyIBM"
# AM = align_ibm1(train_dir, num_sentences, max_iter, fn_AM)


num_sentences = 100

train_dir = "/Users/yuxiangdai/Documents/A2_SMT/data/Hansard/Training/"
AM = align_ibm1(train_dir, num_sentences, 5, "daniel")
print("len:", len(AM))
