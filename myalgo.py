from collections import defaultdict
import numpy as np
from scipy.stats import norm
from datetime import  datetime

def normalize(P):
    s = sum(P.values())
    for state,p in P.items():
        P[state] = p/s

def statistic(hmm):
    hmm.datas['Date'] = [datetime.strftime(x,"%Y-%m-%d") for x in hmm.datas.index]
    res = defaultdict(list)
    for name,g in hmm.datas.groupby('Date'):
        for state,number in g['Col_state'].value_counts().items():
            if len(g) < 1400:continue
            res[state].append(number)
    for k in res:
        res[k] = (np.mean(res[k]),np.std(res[k]))
    return res

def getAddition(j,M,hour,addition):
    if not j in M:return 0.0000001
    a,b = addition[hour][j]
    return 1.0 - norm.cdf((float(M[j]) - a)/ b)


def getOffset(state,records,addition):
    if not state in records or not state in addition:return 0.00000001
    a,b = addition[state]
    return 1.0 - norm.cdf((float(records[state]) - a)/ b)

def myAlgoAll(hmm):
    ############## niml-sparse method ###########
    hmm.logger.info("Run my algorithm")
    start = datetime.now()
    right = 0
    wrong = 0
    delta = 0.0
    allpower = 0.0
    lastday = hmm.tests.index[0].day
    addition = statistic2(hmm)
    record = defaultdict(int)

    for i in range(1, len(hmm.tests)):
        if hmm.tests.index[i].day != lastday:
            lastday = hmm.tests.index[i].day
            record = defaultdict(int)
        P_t_1 = dict()
        for j, pb in hmm.B[hmm.tests['Col_sum'].iloc[i - 1]]:
            P_t_1[j] = hmm.Pi[hmm.tests.index[i-1].hour].get(j, 0.00000001) * pb

        P_t = dict()
        y = hmm.tests['Col_sum'].iloc[i]
        for j, pb in hmm.B[y]:
            if j in hmm.A[hmm.tests.index[i].hour]:
                P_t[j] = max(P_t_1.get(a, 0.000000001) * pa * pb * getOffset(j,record,addition) for a, pa in hmm.A[hmm.tests.index[i].hour].get(j))
        if not P_t:
            from copy import deepcopy
            P_t = deepcopy(hmm.Pi[hmm.tests.index[i].hour])
        state, maxp = -1, -1
        for s, p in P_t.items():
            if p > maxp:
                state = s
        record[state] += 1

        r,w,d,a = hmm.evaluatePower(i,state)
        right+=r
        wrong+=w
        delta += d
        allpower += a
    #hmm.logger.info( 'Right:%d, Wrong:%d' % (right, wrong))
    hmm.logger.info('Testing time %s and testing point %d' % (str(datetime.now() - start),len(hmm.tests)))
    hmm.logger.info(right / (right + wrong))
    hmm.logger.info('Power: %f' % (1.0-delta/(2*allpower)))
