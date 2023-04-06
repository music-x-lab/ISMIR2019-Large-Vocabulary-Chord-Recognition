import numpy as np
import os
from mir.music_base import get_scale_and_suffix
from settings import DEFAULT_SR,DEFAULT_HOP_LENGTH,JAM_DATASET_PATH
import sklearn.metrics as skm
import mir_eval.chord
from pumpp.task.chord import ChordTagTransformer

# some codes are from https://github.com/bmcfee/ismir2017_chords
# without modification for fair comparison
tag_transformer=ChordTagTransformer('3567s')
ISMIR2017_CHORD_VOCAB=np.array(['A#:7', 'A#:aug', 'A#:dim', 'A#:dim7', 'A#:hdim7', 'A#:maj',
       'A#:maj6', 'A#:maj7', 'A#:min', 'A#:min6', 'A#:min7', 'A#:minmaj7',
       'A#:sus2', 'A#:sus4', 'A:7', 'A:aug', 'A:dim', 'A:dim7', 'A:hdim7',
       'A:maj', 'A:maj6', 'A:maj7', 'A:min', 'A:min6', 'A:min7',
       'A:minmaj7', 'A:sus2', 'A:sus4', 'B:7', 'B:aug', 'B:dim', 'B:dim7',
       'B:hdim7', 'B:maj', 'B:maj6', 'B:maj7', 'B:min', 'B:min6',
       'B:min7', 'B:minmaj7', 'B:sus2', 'B:sus4', 'C#:7', 'C#:aug',
       'C#:dim', 'C#:dim7', 'C#:hdim7', 'C#:maj', 'C#:maj6', 'C#:maj7',
       'C#:min', 'C#:min6', 'C#:min7', 'C#:minmaj7', 'C#:sus2', 'C#:sus4',
       'C:7', 'C:aug', 'C:dim', 'C:dim7', 'C:hdim7', 'C:maj', 'C:maj6',
       'C:maj7', 'C:min', 'C:min6', 'C:min7', 'C:minmaj7', 'C:sus2',
       'C:sus4', 'D#:7', 'D#:aug', 'D#:dim', 'D#:dim7', 'D#:hdim7',
       'D#:maj', 'D#:maj6', 'D#:maj7', 'D#:min', 'D#:min6', 'D#:min7',
       'D#:minmaj7', 'D#:sus2', 'D#:sus4', 'D:7', 'D:aug', 'D:dim',
       'D:dim7', 'D:hdim7', 'D:maj', 'D:maj6', 'D:maj7', 'D:min',
       'D:min6', 'D:min7', 'D:minmaj7', 'D:sus2', 'D:sus4', 'E:7',
       'E:aug', 'E:dim', 'E:dim7', 'E:hdim7', 'E:maj', 'E:maj6', 'E:maj7',
       'E:min', 'E:min6', 'E:min7', 'E:minmaj7', 'E:sus2', 'E:sus4',
       'F#:7', 'F#:aug', 'F#:dim', 'F#:dim7', 'F#:hdim7', 'F#:maj',
       'F#:maj6', 'F#:maj7', 'F#:min', 'F#:min6', 'F#:min7', 'F#:minmaj7',
       'F#:sus2', 'F#:sus4', 'F:7', 'F:aug', 'F:dim', 'F:dim7', 'F:hdim7',
       'F:maj', 'F:maj6', 'F:maj7', 'F:min', 'F:min6', 'F:min7',
       'F:minmaj7', 'F:sus2', 'F:sus4', 'G#:7', 'G#:aug', 'G#:dim',
       'G#:dim7', 'G#:hdim7', 'G#:maj', 'G#:maj6', 'G#:maj7', 'G#:min',
       'G#:min6', 'G#:min7', 'G#:minmaj7', 'G#:sus2', 'G#:sus4', 'G:7',
       'G:aug', 'G:dim', 'G:dim7', 'G:hdim7', 'G:maj', 'G:maj6', 'G:maj7',
       'G:min', 'G:min6', 'G:min7', 'G:minmaj7', 'G:sus2', 'G:sus4', 'N',
       'X'], dtype='<U10')

ISMIR2017_CHORD_VOCAB_LOOKUP={str:i for i,str in enumerate(ISMIR2017_CHORD_VOCAB)}

ISMIR2017_ROOT_VOCAB=['A#', 'A', 'B', 'C#', 'C', 'D#', 'D', 'E', 'F#', 'F', 'G#', 'G', 'N', 'X']

def get_vocab_id_by_string(str):
    str=tag_transformer.simplify(str)
    if(str=='N'):
        return ISMIR2017_CHORD_VOCAB_LOOKUP['N']
    elif(str=='X'):
        return ISMIR2017_CHORD_VOCAB_LOOKUP['X']
    scale,suffix=get_scale_and_suffix(str)
    name='%s%s'%(ISMIR2017_ROOT_VOCAB[scale],suffix)
    if(name in ISMIR2017_CHORD_VOCAB_LOOKUP):
        return ISMIR2017_CHORD_VOCAB_LOOKUP[name]
    else:
        return ISMIR2017_CHORD_VOCAB_LOOKUP['X']

def chordlab_to_ismir2017_array(chordlab_est,chordlab_ref,sr=DEFAULT_SR,hop_length=DEFAULT_HOP_LENGTH):

    n_frames=min(
        int(np.round(chordlab_ref[-1][1]*sr/hop_length)),
        int(np.round(chordlab_est[-1][1]*sr/hop_length)))
    arr_est=np.zeros((n_frames,),dtype=np.int)
    for token in chordlab_est:
        start=max(0,int(np.round(token[0]*sr/hop_length)))
        end=min(n_frames,int(np.round(token[1]*sr/hop_length)))
        id=get_vocab_id_by_string(token[2])
        arr_est[start:end]=id
    return arr_est


def conf_to_xconf(C):

    # Qualities = 14 + 2

    Q = np.zeros((16, 16), dtype=C.dtype)

    for j in range(0, 12):
        for i in range(0, 12):
            if j == i:
                continue
            Q[:14, :14] += C[j * 14: j * 14 + 14,
                           i * 14: i * 14 + 14]

            Q[:14, 14:] += C[j * 14:j * 14 + 14, -2:]
            Q[14:, :14] += C[-2:, i * 14:i * 14 + 14]

    Q[14:, 14:] = C[-2:, -2:]

    return Q

def conf_to_rconf(C):

    Q = np.zeros((14, 14), dtype=C.dtype)

    # C[i*12:i*12+14] == truth has root i
    # C[:, j*12:j*12+14] == prediction has root j
    for i in range(0, 12):
        for j in range(0, 12):
            Q[i, j] = np.sum(C[i*14:i*14+14, j*14:j*14+14])

        Q[i, -2:] = np.sum(C[i*14:i*14+14, -2:], axis=0)

        Q[-2:, i] = np.sum(C[-2:, i*14:i*14+14], axis=1)

    Q[-2:, -2:] = C[-2:, -2:]
    return Q

def conf_to_qconf(C):

    # Qualities = 14 + 2

    Q = np.zeros((16, 16), dtype=C.dtype)

    for i in range(0, 12):
        Q[:14, :14] += C[i * 14: (i +1)* 14,
                       i * 14: (i+1) * 14]

        Q[:14, 14:] += C[i * 14:(i+1) * 14, -2:]
        Q[14:, :14] += C[-2:, i * 14:(i+1) * 14]

    Q[14:, 14:] = C[-2:, -2:]

    return Q

def confusion_single(chordlab_est,chordlab_ref):
    est=chordlab_to_ismir2017_array(chordlab_est,chordlab_ref)
    ref=chordlab_to_ismir2017_array(chordlab_ref,chordlab_est)
    return skm.confusion_matrix(est,ref,labels=np.arange(len(ISMIR2017_CHORD_VOCAB)))

def confusion(pool):
    result=np.zeros((len(ISMIR2017_CHORD_VOCAB),len(ISMIR2017_CHORD_VOCAB)))
    for (chordlab_est,chordlab_ref) in pool:
        result+=confusion_single(chordlab_est,chordlab_ref)
    return result

def split_chordlab(chordlab):
    return (np.array([[data[0],data[1]] for data in chordlab],dtype=np.float64),[data[2] for data in chordlab])

def lib_eval_single(chordlab_est,chordlab_ref):
    (gd_intervals,gd_labels) = split_chordlab(chordlab_ref)
    (est_intervals,est_labels) = split_chordlab(chordlab_est)
    est_intervals,est_labels = mir_eval.util.adjust_intervals(est_intervals,est_labels,gd_intervals.min(),gd_intervals.max(),start_label='X',end_label='X')
    (intervals,gd_labels,est_labels)=mir_eval.util.merge_labeled_intervals(gd_intervals,gd_labels,est_intervals,est_labels)
    #durations = mir_eval.util.intervals_to_durations(intervals)
    result=mir_eval.chord.evaluate(intervals,gd_labels,intervals,est_labels)
    #print(result)
    return result

def lib_eval_all(pool,output_file_name):
    metrics="root,thirds,triads,tetrads,mirex,majmin,sevenths".split(',')
    f=open(output_file_name,'w')
    f.write(','.join(['']+metrics)+'\n')
    for i,(chordlab_est,chordlab_ref) in enumerate(pool):
        result=lib_eval_single(chordlab_est,chordlab_ref)
        f.write(','.join([str(i)]+[str(result[metric]) for metric in metrics])+'\n')
    f.close()

def norm(x, axis=1):
    return x / x.sum(axis=axis, keepdims=True)

if __name__ == '__main__':
    from results import process_folder
    pool=process_folder("output/output_joint_chord_net_ismir_v1.0_triad_only_reweight(1.0,1.0)_s%d.best_hmm_full/jam/",
                              os.path.join(JAM_DATASET_PATH,'chordlab')+'/')
    lib_eval_all(pool,"output/output_joint_chord_net_ismir_v1.0_triad_only_reweight(1.0,1.0)_s%d.best_hmm_full.test.csv")


    conf=confusion(pool)
    qconf=conf_to_qconf(conf)

    print(conf.diagonal().sum() / conf.sum())
    print(norm(conf).diagonal().sum() / norm(conf).sum())