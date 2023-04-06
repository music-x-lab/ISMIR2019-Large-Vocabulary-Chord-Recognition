import datasets
from mir.nn.data_storage import FramedRAMDataStorage,FramedH5DataStorage
import numpy as np
import datasets
from extractors.cqt import CQTV2
from mir.extractors.misc import FrameCount
from extractors.key_preprocess import FramedKey
from extractors.beat_preprocess import SimpleFramedDownbeatAnnotation,BasicStructureAnnotationFromBillboard
from mir import io

def create_jams_storage():
    jam=datasets.create_jam_dataset()
    jam.append_extractor(CQTV2,'cqt')
    jam.activate_proxy('cqt',thread_number=8,free=True)
    jam.append_extractor(FrameCount,'n_frame',source='cqt')
    jam.activate_proxy('n_frame',thread_number=8,free=True)
    storage_jam_xchord=FramedH5DataStorage('d:/jams_xchord',dtype=np.int16)
    if(not storage_jam_xchord.created):
        storage_jam_xchord.create_and_cache(jam.entries,'xchord')
    storage_jam_xchord=FramedH5DataStorage('d:/jams_cqt',dtype=np.int16)
    if(not storage_jam_xchord.created):
        storage_jam_xchord.create_and_cache(jam.entries,'cqt')

if __name__ == '__main__':
    create_jams_storage()
    #create_jamosu_chord_storage()