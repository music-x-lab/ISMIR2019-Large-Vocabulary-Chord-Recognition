from mir.extractors import ExtractorBase
from mir import io
import numpy as np
from mir.music_base import get_scale_and_suffix

class FramedKey(ExtractorBase):

    def get_feature_class(self):
        return io.SpectrogramIO

    def extract(self,entry,**kwargs):
        n_frame=entry.n_frame
        sr=entry.prop.sr
        win_shift=entry.prop.hop_length
        result=np.zeros((n_frame))
        keys=entry.key
        for i in range(len(keys)):
            tokens=keys[i]
            begin=int(round(float(tokens[0])*sr/win_shift))
            end = int(round(float(tokens[1])*sr/win_shift))
            if (end > n_frame):
                end = n_frame
            if(begin<0):
                begin=0
            if(i==0):
                begin=0
            if(i==len(keys)-1):
                end=n_frame
            try:
                scale,suffix=get_scale_and_suffix(tokens[2])
                suffix=suffix.strip().lower()
                assert(suffix in ['maj','min'])
                result[begin:end]=scale+(0 if suffix=='maj' else 12)
            except:
                print('Warning: key annotation is %s'%tokens[2])
                result[begin:end]=-1
        return result
