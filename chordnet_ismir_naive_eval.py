from chordnet_ismir_naive import ChordNet,chord_limit,ChordNetCNN
from mir.nn.train import NetworkInterface
import mir.io as io
import datasets
from extractors.cqt import CQTV2,SimpleChordToID
from mir import io,DataEntry
from io_new.chordlab_io import ChordLabIO
from extractors.xhmm_ismir import XHMMDecoder
from complex_chord import Chord,ChordTypeLimit,shift_complex_chord_array_list,complex_chord_chop,enum_to_dict,\
    TriadTypes,SeventhTypes,NinthTypes,EleventhTypes,ThirteenthTypes,create_tag_list
from mir.music_base import NUM_TO_ABS_SCALE
from extractors.complex_chord_preprocess import chordlab_to_complex_chord
from mir import cache
import os
import numpy as np
from joblib import Parallel,delayed
from train_eval_test_split import get_test_fold_by_name
from mir.cache import mkdir_for_file
from mir.extractors.misc import FrameCount
from settings import DEFAULT_SR,DEFAULT_HOP_LENGTH
from mir.data_file import TextureBuilder

def visualize_dataset(net,dataset,chord_dict_name='ismir2017',music='music'):
    hmm=XHMMDecoder(template_file='data/%s_chord_list.txt'%chord_dict_name)
    dataset.append_extractor(CQTV2,'cqt',source=music)
    for entry in dataset.entries:
        print('lstm working on',entry.name)
        probs=net.inference(entry.cqt)
        chordlab=hmm.decode_to_chordlab(entry,probs,False)
        prob_vis=np.concatenate(probs,axis=1)
        entry.append_data(prob_vis,io.SpectrogramIO,'prob')
        entry.append_data(chordlab,ChordLabIO,'hmm_chord')
        entry.visualize(['prob','hmm_chord'],music=music)

def visualize_any(net,file,chord_dict_name='ismir2017'):
    hmm=XHMMDecoder(template_file='data/%s_chord_list.txt'%chord_dict_name)
    entry=DataEntry()
    entry.prop.set('sr',DEFAULT_SR)
    entry.prop.set('hop_length',DEFAULT_HOP_LENGTH)
    entry.append_file(file,io.MusicIO,'music')
    entry.append_extractor(CQTV2,'cqt')
    print('lstm working on',file)
    probs=net.inference(entry.cqt)
    chordlab=hmm.decode_to_chordlab(entry,probs,False)
    prob_vis=np.concatenate(probs,axis=1)
    entry.append_data(prob_vis,io.SpectrogramIO,'prob')
    entry.append_data(chordlab,ChordLabIO,'hmm_chord')
    #exporter=SonicVisualizerExporter()
    #exporter.create('temp/output.svl',sr=entry.prop.sr,win_shift=entry.prop.hop_length)
    #SimplerTexture().generate(entry.hmm_chord,None,exporter)
    #exporter.close()
    entry.visualize(['prob','hmm_chord'])

def eval_jam(cross_net_name,chord_dict_name='ismir2017',show=False,use_bass=True,use_7=True,use_extended=True):
    nets=[NetworkInterface(ChordNet(None),cross_net_name%i,load_checkpoint=True) for i in range(5)]
    jam=datasets.create_jam_dataset()
    jam.append_extractor(CQTV2,'cqt')
    jam.append_extractor(FrameCount,'n_frame',source='cqt')
    #jam.activate_proxy('cqt',thread_number=-1,free=True)
    tag_list=create_tag_list(chord_limit)
    hmm=XHMMDecoder(template_file='data/%s_chord_list.txt'%chord_dict_name,use_bass=use_bass,use_7=use_7,use_extended=use_extended)
    for entry in jam.entries:
        save_name='output/output_%s_hmm_%s/%s.lab'%(cross_net_name,chord_dict_name,entry.name)
        if(os.path.exists(save_name)):
            continue
        print(entry.name)
        fold_id=get_test_fold_by_name(entry.name)
        probs=nets[fold_id].inference(entry.cqt)
        chordlab=hmm.decode_to_chordlab(entry,probs,False)
        prob_vis=np.concatenate(probs,axis=1)
        if(show):
            entry.append_data((tag_list,prob_vis),io.SpectrogramIO,'prob')
            entry.append_data(chordlab,ChordLabIO,'hmm_chord')
            entry.visualize(['prob','hmm_chord'])
        entry.append_data(chordlab,ChordLabIO,'hmm_chord')
        entry.save('hmm_chord',save_name,True)
        entry.free()

if __name__ == '__main__':
    eval_jam('joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s%d.best',chord_dict_name='submission')
    eval_jam('joint_chord_net_ismir_naive_v1.0_reweight(0.3,10.0)_s%d.best',chord_dict_name='submission')
    eval_jam('joint_chord_net_ismir_naive_v1.0_reweight(0.5,10.0)_s%d.best',chord_dict_name='submission')
    eval_jam('joint_chord_net_ismir_naive_v1.0_reweight(0.7,20.0)_s%d.best',chord_dict_name='submission')
    eval_jam('joint_chord_net_ismir_naive_v1.0_reweight(1.0,20.0)_s%d.best',chord_dict_name='submission')


