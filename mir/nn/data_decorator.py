import numpy as np
from abc import ABC,abstractmethod

def data_type_fix(data):

    if(data.dtype==np.float16):
        return data.astype(np.float32)
    elif(data.dtype==np.float64):
        return data.astype(np.float32)
    elif(data.dtype==np.int32):
        return data.astype(np.int64)
    elif(data.dtype==np.int16):
        return data.astype(np.int64)
    elif(data.dtype==np.int8):
        return data.astype(np.int64)
    else:
        return data


class AbstractPitchShifter(ABC):

    @abstractmethod
    def pitch_shift(self,data,shift):
        raise NotImplementedError()

class NoPitchShifter(AbstractPitchShifter):

    def pitch_shift(self,data,shift):
        return data

class CQTPitchShifter(AbstractPitchShifter):

    def __init__(self,spec_dim,shift_low,shift_high,shift_step=3):
        self.shift_low=shift_low
        self.shift_high=shift_high
        self.spec_dim=spec_dim
        self.shift_step=shift_step
        self.min_input_dim=(-self.shift_low+self.shift_high)*self.shift_step+self.spec_dim

    def pitch_shift(self,data,shift):
        if(data.shape[1]<self.min_input_dim):
            raise Exception('CQTPitchShifter excepted spectrogram with dim >= %d, got %d'%
                            (self.min_input_dim,data.shape[1]))
        start_dim=(-shift+self.shift_high)*self.shift_step
        return data[:,start_dim:start_dim+self.spec_dim]

class ChromaShifter(AbstractPitchShifter):

    def pitch_shift(self,data,shift):
        raise NotImplementedError()

class NotePitchShifter(AbstractPitchShifter):

    def pitch_shift(self,data,shift):
        new_data=data.copy()
        new_data[new_data>0]+=shift
        return new_data
