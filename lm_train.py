from preprocess import *
import pickle
import os

def lm_train(data_dir, language, fn_LM):
    """
    This function reads data from data_dir, computes unigram and bigram counts,
    and writes the result to fn_LM
    
    INPUTS:
    
    data_dir	: (string) The top-level directory continaing the data from which
                    to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
    language	: (string) either 'e' (English) or 'f' (French)
    fn_LM		: (string) the location to save the language model once trained
    
    OUTPUT
    
    LM			: (dictionary) a specialized language model
    
    The file fn_LM must contain the data structured called "LM", which is a dictionary
    having two fields: 'uni' and 'bi', each of which holds sub-structures which 
    incorporate unigram or bigram counts
    
    e.g., LM['uni']['word'] = 5 		# The word 'word' appears 5 times
          LM['bi']['word']['bird'] = 2 	# The bigram 'word bird' appears 2 times.
    """
    # Remove for actual
    
    uni = {}
    bi = {}
    # all of the data ﬁles in data dir that end in either ‘e’ for English or ‘f’ for French

    

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith("." + language):
                with open(os.path.join(data_dir, file)) as f:
                    for line in f:
                        if line.endswith('\n'):
                            line = line[:-1]
                        ## Use these two lines once preprocessing is working
                        edited_line = preprocess(line, language)
                        words = edited_line.split()
                        # words = line.split()
                        for index, word in enumerate(words[:-1]):
                            if word in uni.keys():
                                uni[word] += 1
                            else:
                                uni[word] = 1
                            next_word = words[index + 1]
                            if word in bi.keys():
                                if next_word in bi[word].keys():
                                    bi[word][next_word] += 1
                                else:
                                    bi[word][next_word] = 1
                            else:
                                bi[word] = {next_word: 1}
                        last_word = words[-1]
                        if last_word in uni.keys():
                            uni[last_word] += 1
                        else:
                            uni[last_word] = 1 
                        print(words)

    #Save Model
    language_model = {'uni':uni, 'bi':bi}
    with open(fn_LM+'.pickle', 'wb') as handle:
    	pickle.dump(language_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return language_model

data_dir = "/Users/yuxiangdai/Documents/A2_SMT/data/Hansard/Training/"
language = "e"
fn_LM = "test"
lm_train(data_dir, language, fn_LM)