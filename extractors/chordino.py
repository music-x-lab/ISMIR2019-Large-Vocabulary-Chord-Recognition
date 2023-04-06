from mir.extractors.extractor_base import *
from mir.common import WORKING_PATH,SONIC_ANNOTATOR_PATH,PACKAGE_PATH
from mir.cache import hasher
import numpy as np
import subprocess
from io_new.chordlab_io import ChordLabIO
from mir.extractors.vamp_extractor import rewrite_extract_n3
from mir.music_base import get_scale_and_suffix,NUM_TO_ABS_SCALE
from mir.data_file import FileProxy

class ChordinoLab(ExtractorBase):

    def __init__(self):
        super(ChordinoLab, self).__init__()
        self.suffix_remap_dict={
            '':'maj',
            'm':'min',
            'dim7':'dim7',
            '6':'maj6',
            '7':'7',
            'maj7':'maj7',
            'm7':'min7',
            'm6':'min6',
            'dim':'dim',
            'aug':'aug',
            'm7b5':'hdim7'
        }
        self.delta_dict=['1','b2','2','b3','3','4','b5','5','b6','6','b7','7']

    def get_feature_class(self):
        return ChordLabIO

    def chord_name_fix(self,name):
        if(name=='N'):
            return name
        scale,suffix=get_scale_and_suffix(name)
        bass=scale
        if('/' in suffix):
            suffix,bass_str=suffix.split('/')
            bass,_=get_scale_and_suffix(bass_str)
        if(suffix not in self.suffix_remap_dict):
            raise Exception('Unknown suffix: %s'%suffix)
        trans_suffix=self.suffix_remap_dict[suffix]
        delta=(bass-scale+12)%12
        if(delta==0):
            return '%s:%s'%(NUM_TO_ABS_SCALE[scale],trans_suffix)
        else:
            return '%s:%s/%s'%(NUM_TO_ABS_SCALE[scale],trans_suffix,self.delta_dict[delta])

    def extract(self,entry,**kwargs):
        print('Chordino working on entry '+entry.name)
        length=entry.n_frame*(entry.prop.hop_length/entry.prop.sr)
        if(isinstance(entry.dict['music'],FileProxy)):
            temp_path=entry.dict['music'].filepath
        else:
            music_io = io.MusicIO()
            temp_path=os.path.join(WORKING_PATH,'temp/chordino_extractor_%s.wav'%hasher(entry.name))
            music_io.write(entry.music,temp_path,entry)
            entry.free('music')
        temp_n3_path=os.path.join(WORKING_PATH,'temp/chordino_extractor_%s.n3'%hasher(entry.name))
        rewrite_extract_n3(entry,os.path.join(WORKING_PATH,'data/chordino.n3'),temp_n3_path)
        proc=subprocess.Popen([SONIC_ANNOTATOR_PATH,
                               '-t',temp_n3_path,
                               temp_path,
                               '-w','lab','--lab-stdout'
                               ],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
        # print('Begin processing')
        result=[]
        last_begin=0.0
        last_chord=''
        for line in proc.stdout:
            line=bytes.decode(line).strip()
            tokens=line.split('\t')
            begin = float(tokens[0])
            #todo: check if time needs shift
            if (begin < 0.0):
                begin = 0.0
            if(last_chord!=''):
                converted_chord=self.chord_name_fix(last_chord)
                result.append([last_begin,begin,converted_chord])
            last_begin=begin
            last_chord=tokens[1][1:-1]
        if(last_begin<length and last_chord!=''):
            result.append([last_begin,length,last_chord])
        try:
            if(not isinstance(entry.dict['music'],FileProxy)):
                os.unlink(temp_path)

            os.unlink(temp_n3_path)
        except:
            pass
        return result
