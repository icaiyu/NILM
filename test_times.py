
# 不同电器参与分解所需要的时间，进行对比
from hmm import NilmHmm
from myalgo import myAlgo
from itertools import  combinations
from datetime import datetime
import pandas as pd

DATA = {
'house1': [
    'bathroom_gfi11',
    'electric_heat12',
    'kitchen_outlets14',
    'kitchen_outlets15',
    'lighting16',
    'refrigerator4',
    'washer_dryer19'],

'house2' :[
    'kitchen_outlets',
    'lighting',
    'stove',
    'microwave',
    'washer_dryer',
    'kitchen_outlets2',
    'refrigerator',
    'dishwaser',
    'disposal',
],


'house3':[
    'outlets_unknown1',
    'lighting1',
    'electronics',
    'refrigerator',
    'disposal',
    'dishwaser',
    'furance',
    'washer_dryer1',
    'microwave',
    'smoke_alarms',
    'bathroom_gfi',
],

'house4':[
    'lighting1',
    'furance',
    'washer_dryer',
    'stove',
    'air_conditioning1',
    'miscellaeneous',
    'smoke_alarms',
    'dishwaser',
    'bathroom_gfi1',
],
'house5':[
    'microwave',
    'lighting1',
    'outlets_unknown1',
    'furance',
    'outlets_unknown2',
    'washer_dryer1',
    'washer_dryer2',
    'subpanel1',
    'subpanel2',
    'electric_heat1',
    'electric_heat2',
    'lighting2',
    'outlets_unknown3',
    'bathroom_gfi',
    'lighting3',
    'refrigerator',
    'lighting4',
    'dishwaser',
    'disposal',
    'electronics',
    'lighting5',
    'kitchen_outlets1',
    'kitchen_outlets2',
    'outdoor_outlets3',
],
'house6':[
    'kitchen_outlets1',
    'washer_dryer',
    'stove',
    'electronics',
    'bathroom_gfi',
    'refrigerator',
    'dishwaser',
    'outlets_unknown1',
    'outlets_unknown2',
    'electric_heat',
    'kitchen_outlets2',
    'lighting',
    'air_conditioning1',
    'air_conditioning2',
    'air_conditioning3',
]


}

#
import  numpy as np
def readDataFromRedd(devices, datapath='house1/'):

    state_num = dict()
    datas = pd.read_csv(datapath + devices[0] + '.csv', parse_dates=True, index_col='0')
    state_num[devices[0]] = len(np.unique(datas['label']))
    datas.columns = [devices[0], devices[0] + '_label']

    for dev in devices[1:]:
        fd = pd.read_csv(datapath + '/' + dev + '.csv', parse_dates=True, index_col='0')
        fd.index = pd.to_datetime(fd.index, unit='s')
        state_num[dev] = len(np.unique(fd['label']))
        fd.columns = [dev, dev + '_label']
        datas = pd.merge(datas, fd, left_index=True, right_index=True, how='outer')
    return datas,state_num

if __name__ == '__main__':

    # 这里表示选取家庭6， 计算该家庭中不同电器数量参与分解所需要的时间
    housename = 'house6'
    # nums = int(sys.argv[1]) if len(sys.argv)>=2 else 7

    for nums in range(2,len(alldevices)):
        alldevices = DATA[housename]
        folds = 40
        c = 0
        for i in combinations(range(len(alldevices)),nums):
            c += 1
            if c == 4:break
            devices = [alldevices[x] for x in i]
            hmm = NilmHmm(logpath = '%s_logs/' % housename ,logname='time_devices_in_REDD.log')
            print('Start the %s devices (%d): %s' % (housename,nums,'_'.join(devices)))

            print('Loading data ...')

            start = datetime.now()
            hmm.readDataFromRedd(devices,datapath='house6/',folds=folds)
            hmm.logger.info("Loading data time: %s" % str(datetime.now() - start))

            modelname = '_'.join(devices) + '_' +str(folds)

            print("Creating the model....")
            start = datetime.now()
            hmm.fit()
            hmm.logger.info("Training length: %d ;Fitting time: %s" % (len(hmm.datas),str(datetime.now()-start)))
            hmm.savemodel('%s_models/'%housename,modelname)
            hmm.logger.info('Start the testing the devices: %s' % '_'.join(devices))
            myAlgo(hmm)

