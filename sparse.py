
# 实现 sparseHMM 算法
from utils import *
from datetime import datetime
def sparse(hmm):
    ############## niml-sparse method ###########
    hmm.logger.info("Run sparse algorithm")
    start = datetime.now()
    right = 0
    wrong = 0
    for i in range(1, len(hmm.tests)):
        P_t_1 = dict()
        for j, pb in hmm.B[hmm.tests['Col_sum'].iloc[i - 1]]:
            P_t_1[j] = hmm.bigPi.get(j, 0.00000001) * pb
        P_t = dict()
        y = hmm.tests['Col_sum'].iloc[i]
        for j, pb in hmm.B[y]:
            if j in hmm.bigA:
                P_t[j] = max(P_t_1.get(a, 0.000000001) * pa * pb for a, pa in hmm.bigA.get(j))
        if not P_t:
            from copy import deepcopy
            P_t = deepcopy(hmm.bigPi)
        normalize(P_t)
        r,w = hmm.evaluate(P_t,i)
        right+=r
        wrong+=w
    hmm.logger.info( 'Right:%d, Wrong:%d' % (right, wrong))
    hmm.logger.info('Testing time %s and testing point %d' % (str(datetime.now() - start),len(hmm.tests)))
    hmm.logger.info(right / (right + wrong))