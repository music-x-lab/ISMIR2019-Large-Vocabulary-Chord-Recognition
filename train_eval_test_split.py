import numpy as np

f=open('data/all_1217.csv','r')
lines=[line.strip() for line in f.readlines()]
f.close()
names={line:i for i,line in enumerate(lines)}
total_songs=len(names)
TRAIN_IDS=[]
VAL_IDS=[]
TEST_IDS=[]
TEST_FOLD_LOOKUP_TABLE={}
np.random.seed(20190326)
for fold in range(5):
    f=open('data/train%02d.csv'%fold,'r')
    result=[names[line.strip()] for line in f.readlines()]
    result_length=len(result)
    val_set_count=result_length//4
    perm=np.random.permutation(result_length)
    result=[result[i] for i in perm]
    TRAIN_IDS.append(result[:-val_set_count])
    VAL_IDS.append(result[-val_set_count:])
    f=open('data/test%02d.csv'%fold,'r')
    data=[line.strip() for line in f.readlines()]
    TEST_IDS.append([names[i] for i in data])
    for name in data:
        TEST_FOLD_LOOKUP_TABLE[name]=fold
    f.close()

def get_train_set_ids(fold):
    return np.array(TRAIN_IDS[fold])

def get_val_set_ids(fold):
    return np.array(VAL_IDS[fold])

def get_test_set_ids(fold):
    return np.array(TEST_IDS[fold])

def get_test_fold_by_name(entry_name):
    if(entry_name.startswith('jam/')):
        keyword=entry_name[4:]
        if(keyword in TEST_FOLD_LOOKUP_TABLE):
            return TEST_FOLD_LOOKUP_TABLE[keyword]
    return -1
