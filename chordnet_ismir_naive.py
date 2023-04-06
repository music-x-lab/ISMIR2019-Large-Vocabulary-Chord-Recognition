import torch.nn as nn
import torch.nn.functional as F
from mir.nn.train import NetworkBehavior,NetworkInterface
from mir.nn.data_storage import FramedRAMDataStorage,FramedH5DataStorage
from mir.nn.data_decorator import CQTPitchShifter,AbstractPitchShifter,NoPitchShifter
from mir.nn.data_provider import FramedDataProvider
import torch
import numpy as np
from complex_chord import Chord,ChordTypeLimit,shift_complex_chord_array_list,complex_chord_chop,enum_to_dict,\
    TriadTypes,SeventhTypes,NinthTypes,EleventhTypes,ThirteenthTypes,complex_chord_chop_list
from train_eval_test_split import get_train_set_ids,get_test_set_ids,get_val_set_ids

SHIFT_LOW=-5
SHIFT_HIGH=6
SHIFT_STEP=3
SPEC_DIM=252
LSTM_TRAIN_LENGTH=1000

chord_limit=ChordTypeLimit(
    triad_limit=6,
    seventh_limit=3,
    ninth_limit=3,
    eleventh_limit=2,
    thirteenth_limit=2
)


class ReweightedLoss(nn.Module):

    def __init__(self,counter,power=1.0,max_clip=10.0,gpu=False,triad_only=False):
        super(ReweightedLoss, self).__init__()
        self.weight=[None]*6
        for i in range(6):
            if(i==0 or i==1):
                self.weight[i]=torch.tensor([counter[i][(j+11)//12] for j in range(len(counter[i])*12-11)],dtype=torch.float32)
            else:
                self.weight[i]=torch.tensor(counter[i],dtype=torch.float32)
            self.weight[i]=torch.pow(self.weight[i].max()/self.weight[i],power)
            self.weight[i][self.weight[i]>max_clip]=max_clip
            if(gpu==True):
                self.weight[i]=self.weight[i].cuda()
        self.triad_only=triad_only


    def forward(self, output, tag):
        def conditional_classifier_loss(a,b,weight=None):
            if((b<0).all()):
                return torch.tensor(0,device=b.device)
            loss=F.cross_entropy(a[b>=0],b[b>=0],weight=weight[:a.shape[1]])
            #loss_term=self.loss_calc(a[b>=0],b[b>=0])
            return loss
        if(self.triad_only):
            result=conditional_classifier_loss(output[0],tag[:,0],weight=self.weight[0])
        else:
            result=conditional_classifier_loss(output[0],tag[:,0],weight=self.weight[0])+\
                conditional_classifier_loss(output[1],tag[:,1]+1,weight=self.weight[1])+\
                conditional_classifier_loss(output[2],tag[:,2],weight=self.weight[2])+\
                conditional_classifier_loss(output[3],tag[:,3],weight=self.weight[3])+\
                conditional_classifier_loss(output[4],tag[:,4],weight=self.weight[4])+\
                conditional_classifier_loss(output[5],tag[:,5],weight=self.weight[5])
        return result


class CNNFeatureExtractor(nn.Module):

    def norm_layer(self,channels):
        return nn.InstanceNorm2d(channels)

    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()

        self.cdim1=16
        self.cdim2=32
        self.cdim3=64
        self.cdim4=80

        self.conv1a=nn.Conv2d(1,self.cdim1,(3,3),padding=(1,1))
        self.norm1a=self.norm_layer(self.cdim1)
        self.conv1b=nn.Conv2d(self.cdim1,self.cdim1,(3,3),padding=(1,1))
        self.norm1b=self.norm_layer(self.cdim1)
        self.conv1c=nn.Conv2d(self.cdim1,self.cdim1,(3,3),padding=(1,1))
        self.norm1c=self.norm_layer(self.cdim1)
        self.pool1=nn.MaxPool2d((1,3))
        self.conv2a=nn.Conv2d(self.cdim1,self.cdim2,(3,3),padding=(1,1))
        self.norm2a=self.norm_layer(self.cdim2)
        self.conv2b=nn.Conv2d(self.cdim2,self.cdim2,(3,3),padding=(1,1))
        self.norm2b=self.norm_layer(self.cdim2)
        self.conv2c=nn.Conv2d(self.cdim2,self.cdim2,(3,3),padding=(1,1))
        self.norm2c=self.norm_layer(self.cdim2)
        self.pool2=nn.MaxPool2d((1,3))
        self.conv3a=nn.Conv2d(self.cdim2,self.cdim3,(3,3),padding=(1,1))
        self.norm3a=self.norm_layer(self.cdim3)
        self.conv3b=nn.Conv2d(self.cdim3,self.cdim3,(3,3),padding=(1,1))
        self.norm3b=self.norm_layer(self.cdim3)
        self.pool3=nn.MaxPool2d((1,4))
        self.conv4a=nn.Conv2d(self.cdim3,self.cdim4,(3,3),padding=(1,0))
        self.norm4a=self.norm_layer(self.cdim4)
        self.conv4b=nn.Conv2d(self.cdim4,self.cdim4,(3,3),padding=(1,0))
        self.norm4b=self.norm_layer(self.cdim4)
        self.output_size=3*self.cdim4

    def forward(self, x):
        assert(len(x.shape)==3)
        batch_size=x.shape[0]
        seq_length=x.shape[1]
        x=x.view((batch_size,1,seq_length,SPEC_DIM))
        x=F.selu(self.norm1a(self.conv1a(x)))
        x=F.selu(self.norm1b(self.conv1b(x)))
        x=F.selu(self.norm1c(self.conv1c(x)))
        x=self.pool1(x)
        x=F.selu(self.norm2a(self.conv2a(x)))
        x=F.selu(self.norm2b(self.conv2b(x)))
        x=F.selu(self.norm2c(self.conv2c(x)))
        x=self.pool2(x)
        x=F.selu(self.norm3a(self.conv3a(x)))
        x=F.selu(self.norm3b(self.conv3b(x)))
        x=self.pool3(x)
        x=F.selu(self.norm4a(self.conv4a(x)))
        x=F.selu(self.norm4b(self.conv4b(x)))
        x=x.transpose(1,2).contiguous().view((batch_size,seq_length,self.output_size))
        return x

class ChordNet(NetworkBehavior):

    def __init__(self,cross_subpart_counter,triad_only=False):
        super(ChordNet, self).__init__()
        self.triad_only=triad_only
        self.audio_feature_block=CNNFeatureExtractor()

        self.condition_linear=nn.Linear(self.audio_feature_block.output_size+12+chord_limit.triad_limit+12,128)

        self.hidden_dim1=192
        self.lstm1=nn.LSTM(
            input_size=self.audio_feature_block.output_size,
            hidden_size=self.hidden_dim1//2,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.output_dim1=chord_limit.triad_limit*12+2+12
        self.output_dim2=chord_limit.seventh_limit+chord_limit.ninth_limit+chord_limit.eleventh_limit+chord_limit.thirteenth_limit+4
        self.final_fc1=nn.Linear(self.hidden_dim1,self.output_dim1+self.output_dim2)

        #self.loss_calc=FocalLoss(gamma=2.0)
        if(cross_subpart_counter is not None):
            self.loss_reweight=ReweightedLoss(cross_subpart_counter,power=1.0,max_clip=1.0,gpu=self.use_gpu,triad_only=triad_only)
    def init_hidden(self,batch_size,hidden_dim):
        c_0=torch.zeros(2,batch_size,hidden_dim//2)
        h_0=torch.zeros(2,batch_size,hidden_dim//2)
        if(self.use_gpu):
            c_0=c_0.cuda()
            h_0=h_0.cuda()
        return (c_0,h_0)

    def forward(self, x):
        batch_size=x.shape[0]
        seq_length=x.shape[1]
        x=self.audio_feature_block(x)
        x1=self.lstm1(x,self.init_hidden(batch_size,self.hidden_dim1))[0]
        x1=self.final_fc1(x1).reshape((batch_size*seq_length,self.output_dim1+self.output_dim2))

        bass_del=chord_limit.bass_slice_begin+12+1
        seventh_del=bass_del+chord_limit.seventh_limit+1
        ninth_del=seventh_del+chord_limit.ninth_limit+1
        eleventh_del=ninth_del+chord_limit.eleventh_limit+1
        thirteenth_del=eleventh_del+chord_limit.thirteenth_limit+1
        return x1[:,:chord_limit.bass_slice_begin],\
            x1[:,chord_limit.bass_slice_begin:bass_del],\
            x1[:,bass_del:seventh_del],\
            x1[:,seventh_del:ninth_del],\
            x1[:,ninth_del:eleventh_del],\
            x1[:,eleventh_del:thirteenth_del]

    def loss(self, x, y):
        output=self.feed(x)
        tag=y.view((-1,6))
        return self.loss_reweight(output,tag)

    def inference(self, x):
        seq_length=x.shape[0]
        output=self.feed(x[:,SHIFT_HIGH*SHIFT_STEP:SHIFT_HIGH*SHIFT_STEP+SPEC_DIM].view((1,seq_length,SPEC_DIM)))
        result_triad=F.softmax(output[0],dim=1).cpu().numpy()
        result_bass=F.softmax(output[1],dim=1).cpu().numpy()
        result_7=F.softmax(output[2],dim=1).cpu().numpy()
        result_9=F.softmax(output[3],dim=1).cpu().numpy()
        result_11=F.softmax(output[4],dim=1).cpu().numpy()
        result_13=F.softmax(output[5],dim=1).cpu().numpy()
        return result_triad,result_bass,result_7,result_9,result_11,result_13

class ChordNetCNN(NetworkBehavior):

    def __init__(self,cross_subpart_counter):
        super(ChordNetCNN, self).__init__()
        self.audio_feature_block=CNNFeatureExtractor()

        self.hidden_dim1=192
        self.output_dim1=chord_limit.triad_limit*12+2+12
        self.output_dim2=chord_limit.seventh_limit+chord_limit.ninth_limit+chord_limit.eleventh_limit+chord_limit.thirteenth_limit+4
        self.final_fc1=nn.Linear(self.audio_feature_block.output_size,self.output_dim1+self.output_dim2)

        #self.loss_calc=FocalLoss(gamma=2.0)
        if(cross_subpart_counter is not None):
            self.loss_reweight=ReweightedLoss(cross_subpart_counter,power=1.0,max_clip=1.0,gpu=self.use_gpu)
    def init_hidden(self,batch_size,hidden_dim):
        c_0=torch.zeros(2,batch_size,hidden_dim//2)
        h_0=torch.zeros(2,batch_size,hidden_dim//2)
        if(self.use_gpu):
            c_0=c_0.cuda()
            h_0=h_0.cuda()
        return (c_0,h_0)

    def forward(self, x):
        batch_size=x.shape[0]
        seq_length=x.shape[1]
        x=self.audio_feature_block(x)
        x1=self.final_fc1(x).reshape((batch_size*seq_length,self.output_dim1+self.output_dim2))

        bass_del=chord_limit.bass_slice_begin+12+1
        seventh_del=bass_del+chord_limit.seventh_limit+1
        ninth_del=seventh_del+chord_limit.ninth_limit+1
        eleventh_del=ninth_del+chord_limit.eleventh_limit+1
        thirteenth_del=eleventh_del+chord_limit.thirteenth_limit+1
        return x1[:,:chord_limit.bass_slice_begin],\
            x1[:,chord_limit.bass_slice_begin:bass_del],\
            x1[:,bass_del:seventh_del],\
            x1[:,seventh_del:ninth_del],\
            x1[:,ninth_del:eleventh_del],\
            x1[:,eleventh_del:thirteenth_del]

    def loss(self, x, y):
        output=self.feed(x)
        tag=y.view((-1,6))
        return self.loss_reweight(output,tag)

    def inference(self, x):
        seq_length=x.shape[0]
        output=self.feed(x[:,SHIFT_HIGH*SHIFT_STEP:SHIFT_HIGH*SHIFT_STEP+SPEC_DIM].view((1,seq_length,SPEC_DIM)))
        result_triad=F.softmax(output[0],dim=1).cpu().numpy()
        result_bass=F.softmax(output[1],dim=1).cpu().numpy()
        result_7=F.softmax(output[2],dim=1).cpu().numpy()
        result_9=F.softmax(output[3],dim=1).cpu().numpy()
        result_11=F.softmax(output[4],dim=1).cpu().numpy()
        result_13=F.softmax(output[5],dim=1).cpu().numpy()
        return result_triad,result_bass,result_7,result_9,result_11,result_13

class FocalLoss(nn.Module):

    def __init__(self,gamma=0.0):
        super(FocalLoss, self).__init__()
        self.gamma=gamma

    def forward(self, input, target):
        logpt=F.log_softmax(input,dim=1)
        logpt=logpt.gather(1,target[:,None]).view((-1))
        pt=torch.tensor(logpt.data.exp())
        loss=-1*(1-pt)**self.gamma*logpt
        return loss.mean()

class ComplexChordShifter(AbstractPitchShifter):

    def pitch_shift(self,data,shift):
        return shift_complex_chord_array_list(complex_chord_chop_list(data,chord_limit),shift)

if __name__ == '__main__':
    TOTAL_SLICE_COUNT=5
    import sys,pickle
    slice_id=int(sys.argv[1])
    if(slice_id>=5 or slice_id<-1):
        raise Exception('Invalid input')
    storage_x=FramedH5DataStorage('D:/jams_cqt')
    storage_y=FramedH5DataStorage('D:/jams_xchord')
    storage_x.load_meta()
    song_count=storage_x.total_song_count
    if(0<=slice_id and slice_id<=5):
        print('Train on slice %d'%slice_id)
        f=open('data/cross_subpart_weight%d.pkl'%slice_id,'rb')
        cross_subpart_counter=pickle.load(f)
        f.close()
        train_indices=get_train_set_ids(slice_id)
        val_indices=get_val_set_ids(slice_id)
    else:
        train_indices=np.arange(song_count)
        val_indices=np.arange(0,1) # fake validation here
        # todo: weight calculation for full dataset
        f=open('data/cross_subpart_weight%d.pkl'%0,'rb')
        cross_subpart_counter=pickle.load(f)
        f.close()
    train_provider=FramedDataProvider(train_sample_length=LSTM_TRAIN_LENGTH,shift_low=SHIFT_LOW,shift_high=SHIFT_HIGH,num_workers=1,average_samples_per_song=1)
    train_provider.link(storage_x,CQTPitchShifter(SPEC_DIM,SHIFT_LOW,SHIFT_HIGH),subrange=train_indices)
    train_provider.link(storage_y,ComplexChordShifter(),subrange=train_indices)

    val_provider=FramedDataProvider(train_sample_length=-1,shift_low=0,shift_high=0,num_workers=1,average_samples_per_song=1,need_shuffle=False)
    val_provider.link(storage_x,CQTPitchShifter(SPEC_DIM,SHIFT_LOW,SHIFT_HIGH),subrange=val_indices)
    val_provider.link(storage_y,ComplexChordShifter(),subrange=val_indices)



    trainer=NetworkInterface(ChordNet(cross_subpart_counter,triad_only=True),
                             'joint_chord_net_ismir_v1.0_triad_only_reweight(1.0,1.0)_s%d'%slice_id,load_checkpoint=True)
    if(slice_id==-1):
        trainer.train_supervised(train_provider,val_provider,batch_size=12,
                             learning_rates_dict={1e-3:35,1e-4:25,1e-5:15,1e-6:10},round_per_print=10,round_per_save=500,
                             round_per_val=-1,early_end_epochs=100,val_batch_size=1)
    else:
        trainer.train_supervised(train_provider,val_provider,batch_size=12,
                                 learning_rates_dict={1e-3:60,1e-4:30,1e-5:30,1e-6:10},round_per_print=10,round_per_save=500,
                                 round_per_val=-1,early_end_epochs=5,val_batch_size=1)