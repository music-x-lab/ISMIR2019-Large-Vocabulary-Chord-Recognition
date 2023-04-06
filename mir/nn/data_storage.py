from abc import ABC,abstractmethod
from mir.common import DEFAULT_DATA_STORAGE_PATH
import h5py
import os
import numpy as np


def mkdir_for_file(path):
    folder_path=os.path.dirname(path)
    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)
    return path

class AbstractDataStorage(ABC):
    def __init__(self):
        self.created=False
        self.loaded=False

    @abstractmethod
    def load(self):
        raise NotImplementedError()

    @abstractmethod
    def unload(self):
        raise NotImplementedError()

    @abstractmethod
    def create_and_cache(self,entries,proxy_name,allow_truncate=False,n_frames=None):
        raise NotImplementedError()

    @abstractmethod
    def get_length(self):
        raise NotImplementedError()

class BasicH5DataStorage(AbstractDataStorage):
    def __init__(self,name):
        super(BasicH5DataStorage, self).__init__()
        self.filename=mkdir_for_file(os.path.join(DEFAULT_DATA_STORAGE_PATH,name+'.h5d'))
        if(os.path.exists(self.filename)):
            self.created=True
            self.load_meta()
        self.root=None

    def load_meta(self):
        raise NotImplementedError()

    def delete(self):
        if(self.created):
            self.root.close()
            self.root=None
            os.unlink(self.filename)
            self.loaded=False
            self.created=False

    def create_h5(self,root,entries,proxy_name,allow_truncate=False,n_frames=None):
        raise NotImplementedError()

    def create_and_cache(self,entries,proxy_name,allow_truncate=False,n_frames=None):
        with h5py.File(self.filename+'.h5part','w') as root:
            self.create_h5(root,entries,proxy_name,allow_truncate,n_frames)
        os.rename(self.filename+'.h5part',self.filename)
        self.created=True
        self.load_meta()

class BasicRAMDataStorage(AbstractDataStorage):
    def __init__(self,name):
        super(BasicRAMDataStorage, self).__init__()
        self.filename=mkdir_for_file(os.path.join(DEFAULT_DATA_STORAGE_PATH,name+'.npy'))
        if(os.path.exists(self.filename)):
            self.created=True
            self.load_meta()
        self.data=None

    def load_meta(self):
        raise NotImplementedError()

    def load(self):
        if(not self.created):
            raise Exception('Data storage not created yet')
        if(not self.loaded):
            self.data=np.load(self.filename)
            self.data.flags.writeable=False
            self.loaded=True

    def unload(self):
        if(self.loaded):
            self.data=None
            self.loaded=False

    def delete(self):
        if(self.created):
            os.unlink(self.filename)
            self.data=None
            self.loaded=False
            self.created=False

    def create_ram(self,entries,proxy_name,allow_truncate=False,n_frames=None):
        raise NotImplementedError()

    def create_and_cache(self,entries,proxy_name,allow_truncate=False,n_frames=None):
        self.create_ram(entries,proxy_name,allow_truncate,n_frames)
        np.save(self.filename,self.data)
        self.created=True
        self.load_meta()

class FramedStorage(AbstractDataStorage):
    def __init__(self):
        super(FramedStorage, self).__init__()
        self.length=None
        self.start=None

    def load(self):
        raise NotImplementedError()

    def locate(self,song_id,sample_id,length):
        raise NotImplementedError()

class FramedH5DataStorage(BasicH5DataStorage,FramedStorage):
    def __init__(self,name,dtype=None):
        super(FramedH5DataStorage, self).__init__(name)
        self.feature=None
        self.dtype=dtype

    def load_meta(self):
        root=h5py.File(self.filename,'r')
        self.length=np.array(root['length'])
        self.start=np.array(root['start'])
        self.total_song_count=self.length.shape[0]
        root.close()

    def load(self):
        if(not self.created):
            raise Exception('Data storage not created yet')
        if(not self.loaded):
            self.root=h5py.File(self.filename,'r')
            self.feature=self.root['feature']
            self.loaded=True

    def unload(self):
        if(self.loaded):
            print('Unload me')
            self.feature=None
            self.root.close()
            self.root=None
            self.loaded=False

    def create_h5(self,root,entries,proxy_name,allow_truncate=False,n_frames=None):
        if(self.dtype is None):
            raise Exception('Unspecific dtype when creating.')
        total_length=0
        spec_shape=entries[0].dict[proxy_name].get(entries[0]).shape
        if(len(spec_shape)==1):
            spec_dim=1
        else:
            assert(len(spec_shape)==2)
            spec_dim=spec_shape[1]
        print('Spectrogram dim =',spec_dim)
        if(n_frames is None):
            print('Info: Using interior n_frame information of entries')
            n_frames=[0]*len(entries)
            for (i,entry) in enumerate(entries):
                n_frames[i]=entry.n_frame
                total_length+=entry.n_frame
                print('%d/%d Appending length :'%(i,len(entries)),entry.name,'(Total: %d)'%total_length)
        else:
            total_length=np.sum(n_frames)
        print('Total length =',total_length)
        dataset=root.create_dataset('feature',(total_length,spec_dim),dtype=self.dtype)
        start=root.create_dataset('start',(len(entries),),dtype=np.int64)
        length=root.create_dataset('length',(len(entries),),dtype=np.int64)
        p=0
        for(i,entry) in enumerate(entries):
            print('%d/%d Appending %s of %s'%(i,len(entries),proxy_name,entry.name))
            data=entry.dict[proxy_name].get(entry)
            if(len(data.shape)==1):
                data=data.reshape((-1,1))
            if(allow_truncate):
                assert(n_frames[i]<=data.shape[0])
                data=data[:n_frames[i],:]
            else:
                assert(n_frames[i]==data.shape[0])
            start[i]=p
            length[i]=n_frames[i]
            dataset[p:p+n_frames[i],:]=data
            p+=n_frames[i]
            entry.free()

    def get_length(self):
        return self.total_song_count

    def locate(self,song_id,sample_id,length):
        if(sample_id+length>self.length[song_id]):
            print('Warning: Cross boundary sampling at %s'%self.__class__.__name__)
        sample=self.feature[self.start[song_id]+sample_id:self.start[song_id]+sample_id+length,:]
        return sample

class FramedRAMDataStorage(BasicRAMDataStorage,FramedStorage):
    def __init__(self,name,dtype=None):
        super(FramedRAMDataStorage, self).__init__(name)
        self.dtype=dtype

    def load_meta(self):
        self.length=np.load(self.filename[:-4]+'.length.npy')
        self.length.flags.writeable=False
        self.start=np.cumsum(self.length)-self.length
        self.total_song_count=self.length.shape[0]

    def create_ram(self,entries,proxy_name,allow_truncate=False,n_frames=None):
        if(allow_truncate==True or n_frames is not None):
            print('allow_truncate & n_frames parameters are not supported by RAM datasets.')
        if(self.dtype is None):
            raise Exception('Unspecific dtype when creating.')
        spec_shape=entries[0].dict[proxy_name].get(entries[0]).shape
        if(len(spec_shape)==1):
            spec_dim=1
        else:
            assert(len(spec_shape)==2)
            spec_dim=spec_shape[1]
        print('Spectrogram dim =',spec_dim)
        length=np.zeros(len(entries),dtype=np.int64)
        def get_entry_feature(i,entry,proxy_name):
            print('%d/%d Appending %s of %s'%(i,len(entries),proxy_name,entry.name))
            data=entry.dict[proxy_name].get(entry).astype(self.dtype)
            length[i]=data.shape[0]
            return data.reshape((-1,1)) if len(data.shape)==1 else data
        result=np.concatenate([get_entry_feature(i,entry,proxy_name) for (i,entry) in enumerate(entries)],axis=0)
        np.save(self.filename[:-4]+'.length.npy',length)
        self.data=result
        self.length=length
        self.start=np.cumsum(self.length)-self.length
        self.loaded=True

    def get_length(self):
        return self.total_song_count

    def locate(self,song_id,sample_id,length):
        if(sample_id+length>self.length[song_id]):
            print('Warning: Cross boundary sampling at %s'%self.__class__.__name__)
        sample=self.data[self.start[song_id]+sample_id:self.start[song_id]+sample_id+length,:]
        return sample
