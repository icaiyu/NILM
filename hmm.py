import pandas as pd
import numpy as np
from collections import defaultdict
import pickle
import logging

pd.set_option('display.max_columns', None)  # 显示所有列
class NilmHmm():
    def __init__(self, logpath,logname="result.log"):
        self.state_weights = dict()
        self.state_num = []
        self.A = [defaultdict(set) for _ in range(24)]
        self.bigA = defaultdict(set)
        self.B = defaultdict(set)
        self.Pi = [defaultdict(float) for _ in range(24)]
        self.bigPi = dict()
        self.setlogger(logpath,logname)

    def readDataFromRedd(self,devices, datapath='house1/',folds = 20):
        self.devices = devices
        labels = [dev+'_label' for dev in devices]
        self.state_num = dict()
        datas = pd.read_csv(datapath + devices[0] + '.csv', parse_dates=True, index_col='0')
        self.state_num[devices[0]] = len(np.unique(datas['label']))
        datas.columns = [devices[0], devices[0] + '_label']

        for dev in devices[1:]:
            fd = pd.read_csv(datapath + '/' + dev + '.csv', parse_dates=True, index_col='0')
            fd.index = pd.to_datetime(fd.index, unit='s')
            self.state_num[dev] = len(np.unique(fd['label']))
            fd.columns = [dev, dev + '_label']
            datas = pd.merge(datas, fd, left_index=True, right_index=True, how='outer')

        self.state_weights = dict()
        last = 1
        for dev in self.devices:
            self.state_weights[dev] = last
            last  *= self.state_num[dev]

        datas['Col_sum'] = datas[devices].apply(lambda x: x.sum(), axis=1)
        datas['Col_state'] = (datas[labels] * pd.Series(self.state_weights)).sum(axis=1).astype(int)
        trains_index = int(len(datas) * (folds - 1) / folds)
        self.tests = datas[trains_index:]
        self.datas = datas[:trains_index]

    def readDataFromData(self,datas,state_num,devices,folds=20):
        self.devices = devices
        labels = [dev+'_label' for dev in devices]
        devices_labels = devices + labels
        self.datas = datas[devices_labels]
        self.state_num = state_num
        self.state_weights = dict()
        last = 1
        for dev in devices:
            self.state_weights[dev] = last
            last *= self.state_num[dev]
        self.datas['Col_sum'] = self.datas[devices].apply(lambda x: x.sum(), axis=1)
        self.datas['Col_state'] = (self.datas[labels] * pd.Series(self.state_weights)).sum(axis=1).astype(int)

        trains_index = int(len(datas) * (folds - 1) / folds)
        self.tests = self.datas[trains_index:]
        self.datas = self.datas[:trains_index]

    def readDataFromAmp(self,devices, datapath='house1/', folds=20):
        self.devices = devices


        datas = pd.read_csv(datapath + devices[0] + '.csv', parse_dates=True, index_col='0')
        datas.index = pd.to_datetime(datas.index,unit='s')
        self.state_weights[devices[0]] = 1
        last = len(datas['label'].unique())
        self.state_num.append(last)
        datas.columns = [devices[0], devices[0] + '_label']

        for dev in devices[1:]:
            fd = pd.read_csv(datapath + '/' + dev +'.csv' , parse_dates=True, index_col='0')
            fd.index = pd.to_datetime(fd.index,unit='s')
            self.state_weights[dev] = int(last)
            last *= len(np.unique(fd['label']))
            self.state_num.append(len(np.unique(fd['label'])))
            fd.columns = [dev, dev + '_label']
            datas = pd.merge(datas, fd, left_index=True, right_index=True, how='outer')

        datas.dropna(inplace=True)
        datas['Col_sum'] = datas[devices].apply(lambda x: x.sum(), axis=1)
        datas['Col_state'] = datas[[label + '_label' for label in devices]].apply(encode, axis=1)
        trains_index = int(len(datas) * (folds - 1) / folds)
        self.tests = datas[trains_index:]
        self.datas = datas[:trains_index]

    def calculateA(self):
        for i in range(24):
            start = str(i) + ":00"
            end = str(i) + ":59"
            tmp = self.datas.between_time(start, end)

            d = defaultdict(int)
            counts = defaultdict(int)
            for j in range(len(tmp) - 1):
                d[(tmp["Col_state"][j], tmp['Col_state'][j + 1])] += 1
                counts[tmp["Col_state"][j]] += 1

            for (a, b) in d.keys():
                d[(a, b)] = float(d[(a, b)]) / float(counts[a])
            for (a, b), p in d.items():
                self.A[i][b].add((a, p))

    # 计算全部的A矩阵
    def calculatebigA(self):
        d = defaultdict(int)
        counts = defaultdict(int)
        for j in range(len(self.datas) - 1):
            d[(self.datas["Col_state"][j], self.datas['Col_state'][j + 1])] += 1
            counts[self.datas["Col_state"][j]] += 1

        for (a, b) in d.keys():
            d[(a, b)] = float(d[(a, b)]) / float(counts[a])
        for (a, b), p in d.items():
            self.bigA[b].add((a, p))

    def calculateB(self):
        groups = self.datas.groupby('Col_state')
        for name, g in groups:
            l = len(g)
            for a, b in g['Col_sum'].value_counts().items():
                self.B[a].add((name, float(b) / float(l)))

    def calculatePi(self):
        for i in range(24):
            start = str(i) + ":00"
            end = str(i) + ":59"
            tmp = self.datas.between_time(start, end)
            for a, b in tmp['Col_state'].value_counts().items():
                self.Pi[i][a] = float(b) / len(tmp)

    def calculatebigPi(self):
        for a, b in self.datas['Col_state'].value_counts().items():
            self.bigPi[a] = float(b) / len(self.datas)

    def fit(self):
        self.calculatebigA()
        self.calculatebigPi()
        self.calculateB()
        self.calculateA()
        self.calculatePi()

    def savemodel(self, path = 'modesl/', modelname="model1"):
        hmm = dict()
        hmm['bigA'] = self.bigA
        hmm['B'] = self.B
        hmm['bigPi'] = self.bigPi
        hmm['A'] = self.A
        hmm['Pi'] = self.Pi
        with open(path + modelname, 'wb') as jsonfile:
            pickle.dump(hmm, jsonfile)

    def loadmodel(self, path = 'models/' ,modelname="model1"):
        with open(path + modelname, "rb") as jfile:
            hmm = pickle.load(jfile)
        self.bigA = hmm['bigA']
        self.B = hmm['B']
        self.bigPi = hmm['bigPi']
        self.A = hmm['A']
        self.Pi = hmm['Pi']

    def compare(self, i, state):
        groundtruth = []
        for dev in self.devices:
            groundtruth.append(self.tests[dev + "_label"][i])
        right = self.accuracy(groundtruth, self.decode(state))
        wrong = len(self.devices) - right
        return right, wrong

    def decode(self, state):
        res = []
        for dev in self.devices:
            tmp = state // self.state_weights[dev]
            res.append(tmp % self.state_num[dev])
        return res

    def accuracy(self, state1, state2):
        assert len(state1) == len(state2)
        c = 0
        for i in range(len(state1)):
            if state1[i] == state2[i]:
                c += 1
        return c

    def fitpower(self):
        self.power = defaultdict(dict)
        for dev in self.devices:
            for name,g in self.datas[[dev,dev+'_label']].groupby(dev+'label'):
                self.power[dev][name] = g[dev].mean()

    def evaluatePower(self,i,state):
        #groundtruth = []
        delta = 0
        allpower = 0
        right = 0
        estimate = self.decode(state)
        for index,dev in enumerate(self.devices):
            truthpower = self.tests[dev][i]
            estimatepower =  self.power[dev][estimate[index]]
            delta += np.abs(truthpower-estimatepower)
            allpower += truthpower

            if self.tests[dev + "_label"][i] == estimate[index]:right += 1

        wrong = len(self.devices) - right
        return right,wrong,delta,allpower
    def allevaluate(self,P,i):
        state, maxp = -1, -1
        for s, p in P.items():
            if p > maxp:
                state = s

    def evaluate(self, P, i):
        state, maxp = -1, -1
        for s, p in P.items():
            if p > maxp:
                state = s
        return self.compare(i, state)

    def setlogger(self, path,logname):
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        logging.basicConfig(level=logging.INFO, filename=path + logname, filemode='w')
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        self.logger = logging.getLogger('nilm')
        self.logger.addHandler(console)