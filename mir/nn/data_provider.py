import torch.utils.data as torch_data
import torch
from torch.utils.data.dataloader import default_collate
from .data_storage import FramedStorage
from .data_decorator import AbstractPitchShifter,NoPitchShifter,data_type_fix
import numpy as np

class DataProvider(torch_data.Dataset):

    def __init__(self,need_shuffle,collate_fn):
        super(DataProvider, self).__init__()
        self.need_shuffle=need_shuffle
        self.collate_fn=collate_fn

    def _NetworkInterface__init_training_worker(self,worker_id):
        return self.init_worker(worker_id,is_training_set=True)

    def _NetworkInterface__init_validation_worker(self,worker_id):
        return self.init_worker(worker_id,is_training_set=False)

    def init_worker(self,worker_id,is_training_set):
        raise NotImplementedError()

    def get_length(self):
        raise NotImplementedError()

    def get_sample(self,id):
        raise NotImplementedError()

    def __len__(self):
        return self.get_length()

    def __getitem__(self, item):
        return self.get_sample(item)



class FramedDataProvider(DataProvider):

    def __init__(self,train_sample_length,average_samples_per_song=10,shift_low=0,shift_high=0,num_workers=0,
                 allow_truncate=False,need_shuffle=True,collate_fn=default_collate,sample_step=1):
        super(FramedDataProvider, self).__init__(need_shuffle,collate_fn)
        self.train_sample_length=train_sample_length
        self.allow_truncate=allow_truncate
        self.average_samples_per_song=average_samples_per_song
        self.shift_low=shift_low
        self.shift_high=shift_high
        self.num_workers=num_workers
        self.start=None
        self.length=None
        self.storage=[]
        self.valid_song_count=-1
        self.sample_step=sample_step

    def init_worker(self,worker_id,is_training_set):
        if(worker_id>0):
            print('Init worker',worker_id)
            if(is_training_set):
                    np.random.seed(torch.utils.data.get_worker_info().seed%(2**32))
            else:
                np.random.seed(worker_id)
        for (storage,valid_indices,pitch_shifter) in self.storage:
            storage.load()

    def link(self,storage:FramedStorage,pitch_shifter:AbstractPitchShifter=None,subrange=None):
        if(pitch_shifter is None):
            pitch_shifter=NoPitchShifter()
        total_song_count=storage.get_length()
        if(subrange is None):
            subrange=np.arange(total_song_count)
        if(len(self.storage)==0):
            valid_indices=subrange if self.allow_truncate else subrange[storage.length[subrange]>=self.train_sample_length]
            self.valid_song_count=len(valid_indices)
            self.start=storage.start[valid_indices]
            self.length=storage.length[valid_indices]
            if(self.valid_song_count==0):
                print('Warning: No valid song was detected in %s'%self.__class__.__name__)
        valid_indices=subrange if self.allow_truncate else subrange[storage.length[subrange]>=self.train_sample_length]
        new_start=storage.start[valid_indices]
        if(len(new_start)!=self.valid_song_count):
            raise Exception('Inconsistent song count encountered in %s'%self.__class__.__name__)

        if(np.any(self.start!=new_start)):
            raise Exception('Inconsistent data lengths encountered in %s'%self.__class__.__name__)
        self.storage.append((storage,valid_indices,pitch_shifter))

    def get_length(self):
        return self.valid_song_count*self.average_samples_per_song*(self.shift_high-self.shift_low+1)

    def get_sample(self,id):
        shift=id%(self.shift_high-self.shift_low+1)
        raw_id=id//(self.shift_high-self.shift_low+1)%self.valid_song_count
        # print('Getting',raw_id,shift)
        if(self.length[raw_id]<=self.train_sample_length or self.train_sample_length==-1): #truncated
            return [
                data_type_fix(pitch_shifter.pitch_shift(
                    storage.locate(valid_indices[raw_id],0,self.length[raw_id]),shift+self.shift_low))
                for (storage,valid_indices,pitch_shifter) in self.storage
            ]
        else:

            sample_id=np.random.randint((self.length[raw_id]-self.train_sample_length)//self.sample_step+1)*self.sample_step
            return [
                data_type_fix(pitch_shifter.pitch_shift(
                    storage.locate(valid_indices[raw_id],sample_id,self.train_sample_length),shift+self.shift_low))
                for (storage,valid_indices,pitch_shifter) in self.storage
            ]
