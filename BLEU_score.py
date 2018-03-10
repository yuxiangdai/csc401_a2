import math

def BLEU_score(candidate, references, n):
    """
    Compute the LOG probability of a sentence, given a language model and whether or not to
    apply add-delta smoothing
    
    INPUTS:
    sentence :	(string) Candidate sentence.  "SENTSTART i am hungry SENTEND"
    references:	(list) List containing reference sentences. ["SENTSTART je suis faim SENTEND", "SENTSTART nous sommes faime SENTEND"]
    n :			(int) one of 1,2,3. N-Gram level.

    
    OUTPUT:
    bleu_score :	(float) The BLEU score
    """
    
    #TODO: Implement by student.
    p = 1
    for i in range(1, n + 1):
        words = candidate.split()
        if i == 1:
            N = len(words)
            C = 0
            for j in range(N):
                uni = words[j]
                
                C += check_1(words, references, uni)

            prec = C / N

        if i == 2:
            N = len(words) - 1
            C = 0
            for j in range(N):
                bi = words[j:j+2]
                C += check_2(words, references, bi)

            prec = C / N  
            
        
        if i == 3:
            N = len(words) - 2
            C = 0
            for j in range(N):
                tri = words[j: j + 3]
                
                C += check_3(words, references, tri)

            prec = C / N 
            
        p = p * prec
    
    words = candidate.split()
    N = len(words)
    smallest_abs_diff = float("inf")
    for ref in references:
        r_words = ref.split()
        r_N = len(r_words)
        if(abs(N - r_N) < smallest_abs_diff):
            nearest_len = r_N
            smallest_abs_diff = abs(N - r_N)

    brevity = nearest_len / N

    if brevity < 1:
        brev_pen = 1
    else:
        brev_pen = math.exp(1 - brevity)

    bleu_score = brev_pen * pow(p, 1/n)

    return bleu_score

def check_1(words, references, uni):
    for ref in references:
        r_words = ref.split()
        r_N = len(r_words)
        for k in range(r_N):
            if r_words[k] == uni:
                return 1
    return 0

def check_2(words, references, bi):
    for ref in references:
        r_words = ref.split()
        r_N = len(r_words) - 1
        for k in range(r_N):
            if r_words[k:k+2] == bi:
                return 1
    return 0

def check_3(words, references, tri):
    for ref in references:
        r_words = ref.split()
        r_N = len(r_words) - 2
        for k in range(r_N):
            if r_words[k: k + 3] == tri:
                return 1
    return 0