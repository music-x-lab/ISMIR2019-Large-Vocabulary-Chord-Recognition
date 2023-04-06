from mir.extractors import ExtractorBase
from mir import io
import librosa
import numpy as np


class BeatAlignCQT(ExtractorBase):


    def get_feature_class(self):
        return io.SpectrogramIO

    def extract(self,entry,**kwargs):
        if('source' in kwargs):
            music=entry.dict[kwargs['source']].get(entry)
        else:
            music=entry.music
        assert(entry.prop.hop_length==256)

        result=librosa.core.hybrid_cqt(music,
                                bins_per_octave=12,
                                fmin=librosa.note_to_hz('F#0'),
                                n_bins=96,
                                tuning=None,
                                hop_length=entry.prop.hop_length).T
        return abs(result).astype(np.float32)

