from preprocess import *
from lm_train import *
from math import log

def log_prob(sentence, LM, smoothing=False, delta=0, vocabSize=0):
    """
    Compute the LOG probability of a sentence, given a language model and whether or not to
    apply add-delta smoothing
    
    INPUTS:
    sentence :	(string) The PROCESSED sentence whose probability we wish to compute
    LM :		(dictionary) The LM structure (not the filename)
    smoothing : (boolean) True for add-delta smoothing, False for no smoothing
    delta : 	(float) smoothing parameter where 0<delta<=1
    vocabSize :	(int) the number of words in the vocabulary
    
    OUTPUT:
    log_prob :	(float) log probability of sentence
    """
    
    #TODO: Implement by student.
    words = sentence.split()
    
    prob = 1
    if not smoothing:
        """ delta smoothing """
        delta = 0
        vocabSize = 0
    
    for index, word in enumerate(words[:-1]):
        
        count_w1w2 = 0
        count_w = 0
        if word in LM['bi'].keys():    
            next_word = word[index + 1]   
            if next_word in LM['bi'][word].keys():
                count_w1w2 = LM['bi'][word][next_word]
        
        if word in LM['uni'].keys():  
            count_w = LM['uni'][word]
        
        prob = prob * (count_w1w2 + delta) / (count_w + delta * vocabSize)
            
    log_prob = 0
    if prob != 0:
        log_prob = log(prob, base=2)
    else:
        log_prob = float('-inf')

    return log_prob