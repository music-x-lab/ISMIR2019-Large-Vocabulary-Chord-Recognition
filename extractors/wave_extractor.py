from mir.extractors import ExtractorBase
from mir import io
import librosa
import numpy as np


class ChromaAlignedFramedWave(ExtractorBase):

    def get_feature_class(self):
        return io.SpectrogramIO

    def extract(self,entry,**kwargs):
        hop_length=entry.prop.hop_length
        n_frame=entry.prop.n_frame
        win_size=entry.prop.win_size
        padding=(win_size-hop_length)//2
        y=entry.music#[padding:]
        assert(len(y)>=n_frame*hop_length)
        return y[:n_frame*hop_length].reshape((n_frame,hop_length))

class CQTAlignedFramedWave(ExtractorBase):

    def get_feature_class(self):
        return io.SpectrogramIO

    def extract(self,entry,**kwargs):
        hop_length=entry.prop.hop_length
        n_frame=entry.cqt.shape[0]
        win_size=entry.prop.win_size
        padding=(win_size-hop_length)//2
        y=np.hstack((entry.music,np.zeros(hop_length,dtype=entry.music.dtype)))#[padding:]
        assert(len(y)>=n_frame*hop_length)
        return y[:n_frame*hop_length].reshape((n_frame,hop_length))
