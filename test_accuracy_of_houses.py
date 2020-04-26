from hmm_REDD import NilmHmm
from myalgo import myAlgo
from sparse import sparse
from nilm import vertibi
import os
from itertools import  combinations
from datetime import datetime
import sys
import pandas as pd
import  numpy as np

DATA = {
    "house1":[
        "oven1",
        "oven2",
        "dishwaser",
        "washer_dryer1",
        "lighting2",
        "washer_dryer2"
    ],
    "house2":[
        "kitchen_outlets",
        "stove",
        "microwave",
        "kitchen_outlets2",
        "refrigerator",
        "dishwaser",
        "disposal"
    ],
    "house3":[
        "outlets_unknown1",
        "lighting1",
        "refrigerator",
        "disposal",
        "dishwaser",
        "furance",
        "microwave",
        "bathroom_gfi"
    ],
    "house4":[
        "lighting1",
        "furance",
        "washer_dryer",
        "stove",
        "air_conditioning1",
        "dishwaser",
        "bathroom_gfi1"
    ],
    "house5":[
        "microwave",
        "lighting1",
        "outlets_unknown1",
        "outlets_unknown2",
        "subpanel1",
        "subpanel2",
        "electric_heat1",
        "kitchen_outlets2"
    ],
    "house6":[
        "kitchen_outlets1",
        "washer_dryer",
        "stove",
        "refrigerator",
        "outlets_unknown1",
        "air_conditioning2",
        "air_conditioning3"
    ]
}


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
    #housename = 'house1'
    for housename in DATA.keys():
        print("housename",housename)
        nums = int(sys.argv[1]) if len(sys.argv)>=2 else len(DATA[housename])
        div = int(sys.argv[2]) if len(sys.argv)>=3 else 1
        devices = DATA[housename]
        datas,state_num = readDataFromRedd(devices,datapath=housename+'/')
        folds = 20
        hmm = NilmHmm(logpath = 'logs/' ,logname='%s_devices_in_REDD.log' %housename)
        print('Start the %s devices (%d): %s' % (housename,nums,'_'.join(devices)))
        hmm.readDataFromData(datas,state_num,devices,folds)
        print("Creating the model....")
        hmm.fit()
        hmm.logger.info('Start the testing the devices: %s' % '_'.join(devices))
        myAlgo(hmm)

