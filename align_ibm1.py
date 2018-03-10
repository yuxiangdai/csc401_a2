from lm_train import *
from log_prob import *
from preprocess import *
from math import log
import os




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
    AM['SENTSTART'] = {}
    AM['SENTEND'] = {}
    AM['SENTSTART']['SENTSTART'] = 1 
    AM['SENTEND']['SENTEND'] = 1

    tcount = {}
    total = {}
    for eng_word in AM.keys():
        total[eng_word] = 0
        for fre_word in AM[eng_word].keys():
            initialize(eng_word, fre_word, AM)
            if eng_word != 'SENTSTART' and eng_word != 'SENTEND' and fre_word != 'SENTSTART' and fre_word != 'SENTEND':
                if fre_word not in tcount.keys():
                    tcount[fre_word] = {eng_word: 0}
                else:
                    tcount[fre_word][eng_word] = 0
    
    # for eng in AM.keys():
    #     for fre in AM[eng].keys():
            
    

    # Iterate between E and M steps
    #   for a number of iterations:
    for iteration in range(max_iter):
        #   set tcount(f, e) to 0 for all f, e
        for fre in tcount.keys(): 
            for eng in tcount[fre].keys():
                tcount[fre][eng] = 0
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
    q = 0
    for root, dirs, files in os.walk(train_dir):
        for file in files:
            # if file.endswith(".e") and file.startswith("hansard"):
            if file.endswith(".e"):
                filename = file[:-2]
                q += 1
                # print(q)
                with open(os.path.join(train_dir, filename + ".e")) as f:
                    english_file = f.readlines()
                with open(os.path.join(train_dir, filename + ".f")) as f:
                    french_file = f.readlines()

                if(len(english_file) != len(french_file)):
                    print(q)

                for i in range(min(len(english_file), len(french_file), num_sentences)):
                    line = english_file[i]
                    line_french = french_file[i]
                    processed_line = preprocess(line, "e")
                    processed_line_french = preprocess(line_french, "f")
                    sentence_pairs.append([processed_line, processed_line_french])
                        
                ## Initialize AM

                    for eng_word in processed_line.split():
                        
                        for fre_word in processed_line_french.split():
                            if eng_word != 'SENTSTART' and eng_word != 'SENTEND' and fre_word != 'SENTSTART' and fre_word != 'SENTEND':
                                if eng_word in AM.keys():
                                    if fre_word not in AM[eng_word].keys():
                                        AM[eng_word][fre_word] = 0    
                                else:
                                    AM[eng_word] = {fre_word: 0}

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

    for E, F in sentence_pairs:
        # convert into lists for set() and count()
        Fsplit = F.split()
        Esplit = E.split()
        for f in set(Fsplit):
            denom_c = 0
            for e in set(Esplit):
                if checkef(e, f):
                #  denom_c += P(f|e) * F.count(f)
                    denom_c += AM[e][f] * F.count(f)
                # denom_c += AM[e][f] * F.count(f)
            for e in set(Esplit):
                if checkef(e, f):
                    tcount[f][e] += AM[e][f] * F.count(f) * E.count(e) / denom_c
                    total[e] += AM[e][f] * F.count(f) * E.count(e) / denom_c 
    for e in total.keys():
        for f in tcount.keys():
            if e in tcount[f].keys():
                AM[e][f] = tcount[f][e] / total[e]
                

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

train_dir = "/Users/yuxiangdai/Documents/A2_SMT/data/Toy/"
num_sentences = 5
max_iter = 5
fn_AM = "toyIBM"
AM = align_ibm1(train_dir, num_sentences, max_iter, fn_AM)
print("done")