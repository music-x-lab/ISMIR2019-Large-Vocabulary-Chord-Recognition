from mir.extractors import ExtractorBase
from mir import io
from io_new.chordlab_io import ChordLabIO
import pickle
import os
import pumpp

from extractors.ismir2017.train_deep import construct_model
from train_eval_test_split import get_test_fold_by_name

MODEL_PATH=R'extractors\ismir2017\data'

class ISMIR2017ChordExtractor(ExtractorBase):

    def get_feature_class(self):
        return ChordLabIO

    def extract(self,entry,**kwargs):
        source='music' if 'source' not in kwargs else kwargs['source']
        fold=kwargs['fold'] if 'fold' in kwargs else get_test_fold_by_name(entry.name)
        if(fold==-1):
            raise Exception('Unspecific argument: fold')

        with open(os.path.join(MODEL_PATH,'pump.pkl'), 'rb') as fd:
            pump = pickle.load(fd)
        chord_dict=pump['chord_tag'].encoder.classes_
        model=construct_model(pump,structured=True)[0]
        weight_path=os.path.join(MODEL_PATH,'fold%02d_weights.pkl'%fold)
        model.load_weights(weight_path)
        sr = 44100
        hop_length = 4096
        p_feature = pumpp.feature.CQTMag(name='cqt', sr=sr, hop_length=hop_length, log=True, conv='tf', n_octaves=6)
        pump = pumpp.Pump(p_feature)
        data=pump.transform(entry.dict[source].filepath)
        result=model.predict(data)[0][0].argmax(axis=1)
        result=chord_dict[result]
        chordlab=[]
        last_time=None
        last_chord='N'
        for i in range(len(result)+1):
            time=i*hop_length/sr
            if(i==len(result) or result[i]!=result[i-1]):
                if(last_time is not None):
                    chordlab.append([last_time,time,last_chord])
            if(i<len(result)):
                if(i==0 or result[i]!=result[i-1]):
                    last_time=time
                    last_chord=result[i]
        return chordlab
