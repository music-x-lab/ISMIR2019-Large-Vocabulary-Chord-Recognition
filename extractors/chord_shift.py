import numpy as np
CHORD_SHIFT_TABLE_SIZE=20
CHORD_SHIFT_TABLE=np.zeros((CHORD_SHIFT_TABLE_SIZE*12+1,24),dtype=np.int64)

for c in range(CHORD_SHIFT_TABLE_SIZE):
    for i in range(12):
        for j in range(24):
            CHORD_SHIFT_TABLE[c*12+i+1,j]=c*12+(i+j)%12+1

if __name__=='__main__':
    print(CHORD_SHIFT_TABLE)