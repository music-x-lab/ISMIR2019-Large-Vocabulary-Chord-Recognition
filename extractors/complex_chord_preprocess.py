from mir.extractors import ExtractorBase
from mir import io
from io_new.complex_chord_io import ComplexChordIO
import numpy as np
import complex_chord
from mir.music_base import get_scale_and_suffix


def chordlab_to_complex_chord(entry,chordlab):
    n_frame=entry.n_frame
    tags = np.ones((n_frame,6))*-2
    for tokens in chordlab:
        sr=entry.prop.sr
        win_shift=entry.prop.hop_length
        begin=int(round(float(tokens[0])*sr/win_shift))
        end = int(round(float(tokens[1])*sr/win_shift))
        if (end > n_frame):
            end = n_frame
        if(begin<0):
            begin=0
        tags[begin:end,:]=complex_chord.Chord(tokens[2]).to_numpy().reshape((1,6))
    return tags

def get_flat_chord_vocab(vocab_filename):
    f=open(vocab_filename,'r')
    lines=[s.strip() for s in f.readlines() if s.strip()!='']
    f.close()
    chord_dict={'N':0}
    n_chord=1
    for line in lines:
        if(line.startswith('C:')):
            quality=line[1:]
            chord_dict[quality]=n_chord
            n_chord+=12
    return chord_dict

def get_flat_chord_vocab_size(vocab_filename):
    f=open(vocab_filename,'r')
    lines=[s.strip() for s in f.readlines() if s.strip()!='']
    f.close()
    chord_dict={'N':0}
    n_chord=1
    for line in lines:
        if(line.startswith('C:')):
            quality=line[1:]
            chord_dict[quality]=n_chord
            n_chord+=12
    return n_chord

def chordlab_to_flat_vocab(entry,chordlab,chord_dict):
    n_frame=entry.n_frame
    tags = np.ones((n_frame),dtype=np.int32)*-1
    sr=entry.prop.sr
    win_shift=entry.prop.hop_length
    for tokens in chordlab:
        begin=int(round(float(tokens[0])*sr/win_shift))
        end = int(round(float(tokens[1])*sr/win_shift))
        if (end > n_frame):
            end = n_frame
        if(begin<0):
            begin=0
        if(tokens[2][0] not in 'NX'):
            scale,suffix=get_scale_and_suffix(tokens[2])
        else:
            scale=0
            suffix=tokens[2]
        if(suffix in chord_dict):
            result=chord_dict[suffix]+scale
        else:
            result=-1
        tags[begin:end]=result
    return tags

