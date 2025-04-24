import math
import numpy as np
from loguru import logger

def decode(seqFile, I, B, A, baseChar):
    N = I.shape[1]
    T = len(seqFile)
    
    VP = np.full((N, T), -np.inf)
    backpointer = np.zeros((N, T), dtype=int)

    logger.info("Starting Viterbi decoding for sequence of length {}", T)

    # Initialization
    cInd = ord(seqFile[0]) - ord(baseChar)
    logger.debug("Initial character: '{}' (index {})", seqFile[0], cInd)
    for i in range(N):
        if B[cInd][i] > 0 and I[0][i] > 0:
            VP[i][0] = math.log10(I[0][i]) + math.log10(B[cInd][i])
        logger.debug("Init VP[{}][0] = {}", i, VP[i][0])

    # Recursion
    for t in range(1, T):
        cInd = ord(seqFile[t]) - ord(baseChar)
        logger.debug("Processing timestep {}: character '{}' (index {})", t, seqFile[t], cInd)
        for i in range(N):
            max_prob = -np.inf
            best_prev_state = 0
            for j in range(N):
                if A[j][i] > 0 and B[cInd][i] > 0:
                    prob = VP[j][t-1] + math.log10(A[j][i]) + math.log10(B[cInd][i])
                    if prob > max_prob:
                        max_prob = prob
                        best_prev_state = j
            VP[i][t] = max_prob
            backpointer[i][t] = best_prev_state
            logger.debug("VP[{}][{}] = {}, backpointer = {}", i, t, max_prob, best_prev_state)

    # Backtracking
    best_path = np.zeros(T, dtype=int)
    best_path[T-1] = np.argmax(VP[:, T-1])
    logger.info("Starting backtrace from final state {}", best_path[T-1])
    for t in range(T-2, -1, -1):
        best_path[t] = backpointer[best_path[t+1]][t+1]
        logger.debug("Backtrace t={}: state={}", t, best_path[t])

    # Output
    result = zip(seqFile, best_path)
    logger.info("Final sequence and state mapping:")
    for char, state in result:
        logger.info("{} -> {}", char, state)

if __name__ == '__main__':
    from pathlib import Path
    from loguru import logger

    logger.add("viterbi_debug.log", level="DEBUG")

    r = Path("samplemod2").read_text().splitlines()
    seqFile = Path("sampleseq2").read_text().strip()

    N = int(r[0])
    baseChar = '!'
    SYMNUM = 94
    I = np.zeros((1, N))
    B = np.zeros((SYMNUM, N))
    A = np.zeros((N, N))

    for idx, keywrd in enumerate(r):
        if keywrd.startswith("InitPr"):
            count = int(keywrd.split()[1])
            for line in r[idx+1:idx+1+count]:
                index, val = line.split()
                I[0][int(index)] = float(val)
        elif keywrd.startswith("OutputPr"):
            count = int(keywrd.split()[1])
            for line in r[idx+1:idx+1+count]:
                s, sym, pr = line.split()
                B[ord(sym)-ord(baseChar)][int(s)] = float(pr)
        elif keywrd.startswith("TransPr"):
            count = int(keywrd.split()[1])
            for line in r[idx+1:idx+1+count]:
                s, s1, pr = line.split()
                A[int(s)][int(s1)] = float(pr)

    decode(seqFile, I, B, A, baseChar)
