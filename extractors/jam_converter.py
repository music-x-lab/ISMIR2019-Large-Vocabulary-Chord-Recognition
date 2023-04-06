from mir.extractors import ExtractorBase
from io_new.chordlab_io import ChordLabIO
from mir import io
import librosa
import numpy as np
from extractors.chord_name_fix import fix_mirex_chord_name

class JamsToChordLabs(ExtractorBase):

    def get_feature_class(self):
        return ChordLabIO

    def extract(self,entry,**kwargs):
        result=[]
        last_time=0.0
        for annotation in entry.jam.annotations:
            if(annotation.sandbox.key=='skipped'):
                continue
            for obs in annotation.data:
                result.append([float(obs.time),float(obs.time+obs.duration),fix_mirex_chord_name(obs.value)])
                if(obs.time+1e-3<last_time):
                    print(entry.name,obs.time,last_time)
                assert(obs.time+1e-3>=last_time)
                last_time=obs.time+obs.duration
        return result