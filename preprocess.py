import re

def preprocess(in_sentence, language):
    """ 
    This function preprocesses the input text according to language-specific rules.
    Specifically, we separate contractions according to the source language, convert
    all tokens to lower-case, and separate end-of-sentence punctuation 
	
	INPUTS:
	in_sentence : (string) the original sentence to be processed
	language	: (string) either 'e' (English) or 'f' (French)
				   Language of in_sentence
				   
	OUTPUT:
	out_sentence: (string) the modified sentence
    """
    # TODO: Implement Function


    def daposrepl(matchobj):
        if matchobj.group(0) in ["d'abord", "d'accord", "d'ailleurs", "d'habitude", "d’abord", "d’accord", "d’ailleurs", "d’habitude"]: 
            print(matchobj.group(0))
            return matchobj.group(0)
        else: 
            return matchobj.group(0)[:2] + " " + matchobj.group(0)[2:]


    # sentence-final punctuation

    in_sentence = re.sub(r'(),()', r"\1 , \2", in_sentence) ## commas
    in_sentence = re.sub(r'():()', r"\1 : \2", in_sentence) ## colons
    in_sentence = re.sub(r'();()', r"\1 ; \2", in_sentence) ## semicolons
    in_sentence = re.sub(r'()(\(|\)|\[|\]|\}|\{)()', r"\1 \2 \3", in_sentence) ##  parentheses
    in_sentence = re.sub(r'()\-()', r"\1 - \2", in_sentence) ## dash btw parentheses
    in_sentence = re.sub(r'()([+-<>=])()', r"\1 \2 \3", in_sentence) ## math
    in_sentence = re.sub(r'()(\"+|\'{2,}|\’{2,})()', r"\1 \2 \3", in_sentence) # quotes

    # add more sentence final puncs, or just add all puncs
    in_sentence = re.sub(r'()([`~!@#$%^&*\(\)_+{}|:\"<>\?\-\=\[\]\;\'.\/,]+$)', r"\1 \2", in_sentence) # sentence final punctuation


    if language == 'french':
        in_sentence = re.sub(r'(l)(\'|\’)(\w+)', r"\1\2 \3", in_sentence) # 1: separate l'
        in_sentence = re.sub(r'([^aeioud]+)(\'|\’)(\w+)', r"\1\2 \3", in_sentence) # 2: seperate all constonants except d
        in_sentence = re.sub(r'(qu)(\'|\’)(\w+)', r"\1\2 \3", in_sentence) # 3: separate qu'
        in_sentence = re.sub(r'(puisqu|lorsqu)(\'|\’)(\w+)', r"\1\2 \3", in_sentence) # 4: puisqu', lorsqu'
        
        in_sentence = re.sub(r'd(\'|\’)\w+', daposrepl, in_sentence)

    out_sentence = re.sub(r'()\s+()', r"\1 \2", in_sentence) ## really dumb way to stop repeat spaces
    return out_sentence

# in_sentence = "Right Hon. Jean Chretien (Prime Minister, Lib."
# preprocess(in_sentence, 'english')