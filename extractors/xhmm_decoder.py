import numpy as np
from complex_chord import shift_complex_chord_array,Chord,NUM_TO_ABS_SCALE


class XHMMDecoder():

    def __init__(self,diff_trans_penalty=30.0,no_chord_penalty=np.log(24.0),beat_trans_penalty=(15.0,45.0,100.0),
                 template_file='data/full_chord_list.txt'):
        self.diff_trans_penalty=diff_trans_penalty
        self.beat_trans_penalty=beat_trans_penalty
        self.no_chord_penalty=no_chord_penalty
        self.__init_known_chord_names(template_file)

    def __init_known_chord_names(self,template_file):
        known_chord_array_pool={}
        known_triad_bass=set()
        f=open(template_file,'r')
        test_chord_names=f.readlines()
        for chord_name in test_chord_names:
            chord_name=chord_name.strip()
            if(':' in chord_name):
                tokens=chord_name.split(':')
                assert(tokens[0]=='C')
                c=Chord(chord_name)
                array=c.to_numpy()
                if(-2 in array):
                    continue
                for shift in range(12):
                    shift_name='%s:%s'%(NUM_TO_ABS_SCALE[shift],tokens[1])
                    shift_array=tuple(shift_complex_chord_array(array,shift))
                    if(shift_array in known_chord_array_pool):
                        continue
                    known_chord_array_pool[shift_array]=shift_name
                    if(shift_array[0:2] not in known_triad_bass):
                        known_triad_bass.add(shift_array[0:2])
        f.close()
        self.known_chord_array=[((0,-1,-1,-1,-1,-1),'N')]+list(known_chord_array_pool.items())
        self.known_triad_bass=[(0,-1)]+list(known_triad_bass)

    def get_chord_tag_obs(self,prob_list,triad_restriction=None):
        suffix_probs=[None]*4
        (prob_triad,prob_bass,suffix_probs[0],suffix_probs[1],suffix_probs[2],suffix_probs[3])=prob_list
        n_frame=prob_triad.shape[0]
        result_names=[]
        result_array=[]
        for (array,name) in self.known_chord_array:
            is_in_range=True
            for i in range(6):
                if(array[i]>=prob_list[i].shape[-1]):
                    is_in_range=False
                    break
            if(is_in_range):
                assert(array[0]>=0)
                result_names.append(name)
                result_array.append(list(array))
        result_array=np.array(result_array,dtype=np.int)
        result_logprob=np.log(prob_triad[:,result_array[:,0]])
        bass_collect=result_array[:,1]>=0
        result_logprob[:,np.logical_not(bass_collect)]-=self.no_chord_penalty
        result_logprob[:,bass_collect]+=np.log(prob_bass[:,result_array[bass_collect,1]])

        for i in range(4):
            suffix_collect=result_array[:,i+2]>=0
            roots=(result_array[:,0]-1)%12
            if(len(suffix_probs[i].shape)==3):
                result_logprob[:,suffix_collect]+=\
                    np.log(suffix_probs[i][:,roots[suffix_collect],result_array[suffix_collect,i+2]])
            else:
                result_logprob[:,suffix_collect]+=\
                    np.log(suffix_probs[i][:,result_array[suffix_collect,i+2]])

        if(triad_restriction is not None):
            triad_restriction=np.array(triad_restriction)
            result_logprob[result_array[None,:,0]!=triad_restriction[:,0,None]]=-np.inf
            result_logprob[result_array[None,:,1]!=triad_restriction[:,1,None]]=-np.inf

        return result_names,result_logprob

    def get_triad_bass_obs(self,prob_list):
        (prob_triad,prob_bass,_,_,_,_)=prob_list
        n_frame=prob_triad.shape[0]
        result_array=[]
        for array in self.known_triad_bass:
            is_in_range=True
            for i in range(2):
                if(array[i]>=prob_list[i].shape[-1]):
                    is_in_range=False
                    break
            if(is_in_range):
                assert(array[0]>=0)
                result_array.append(list(array))
        result_array=np.array(result_array,dtype=np.int)
        result_logprob=np.log(prob_triad[:,result_array[:,0]])
        bass_collect=result_array[:,1]>=0
        result_logprob[:,np.logical_not(bass_collect)]-=self.no_chord_penalty
        result_logprob[:,bass_collect]+=np.log(prob_bass[:,result_array[bass_collect,1]])

        return result_array,result_logprob

    def decode(self,prob_list,beat_arr,triad_restriction=None):
        result_names,result_logprob=self.get_chord_tag_obs(prob_list,triad_restriction)
        n_frame=result_logprob.shape[0]
        n_chord=result_logprob.shape[1]
        dp=np.zeros_like(result_logprob)
        dp[0,1:]-=np.inf
        dp_max_at=np.zeros((n_frame),dtype=np.int)
        pre=np.zeros_like(result_logprob,dtype=np.int)
        dp[0,:]+=result_logprob[0,:]
        dp_max_at[0]=np.argmax(dp[0,:])
        pre[0,:]=-1
        for t in range(1,n_frame):
            same_trans=dp[t-1,:]
            if(beat_arr[t]):
                diff_trans=dp[t-1,dp_max_at[t-1]]-(self.diff_trans_penalty if beat_arr[t]==1 else self.beat_trans_penalty[beat_arr[t]-2])
                use_same_trans=same_trans>diff_trans
                # dp[t-1,use_same_trans]=same_trans[use_same_trans]
                dp[t,:]=np.maximum(diff_trans,same_trans)+result_logprob[t,:]
                pre[t,:]=dp_max_at[t-1]
                pre[t,use_same_trans]=np.arange(n_chord)[use_same_trans]
            else:
                dp[t,:]=same_trans+result_logprob[t,:]
                pre[t,:]=np.arange(n_chord)
            dp_max_at[t]=np.argmax(dp[t,:])
        decode_ids=[None]*n_frame
        decode_ids[-1]=dp_max_at[-1]
        for t in range(n_frame-2,-1,-1):
            decode_ids[t]=pre[t+1,decode_ids[t+1]]
        return [result_names[i] for i in decode_ids]

    def triad_decode(self,prob_list,beat_arr):
        result_array,triad_logprob=self.get_triad_bass_obs(prob_list)
        n_frame=triad_logprob.shape[0]
        n_chord=triad_logprob.shape[1]
        dp=np.zeros_like(triad_logprob)
        dp[0,1:]-=np.inf
        dp_max_at=np.zeros((n_frame),dtype=np.int)
        pre=np.zeros_like(triad_logprob,dtype=np.int)
        dp[0,:]+=triad_logprob[0,:]
        dp_max_at[0]=np.argmax(dp[0,:])
        pre[0,:]=-1
        for t in range(1,n_frame):
            same_trans=dp[t-1,:]
            if(beat_arr[t]>0):
                diff_trans=dp[t-1,dp_max_at[t-1]]-(self.diff_trans_penalty if beat_arr[t]==1 else self.beat_trans_penalty[beat_arr[t]-2])
                use_same_trans=same_trans>diff_trans
                # dp[t-1,use_same_trans]=same_trans[use_same_trans]
                dp[t,:]=np.maximum(diff_trans,same_trans)+triad_logprob[t,:]
                pre[t,:]=dp_max_at[t-1]
                pre[t,use_same_trans]=np.arange(n_chord)[use_same_trans]
            else:
                dp[t,:]=same_trans+triad_logprob[t,:]
                pre[t,:]=np.arange(n_chord)
            dp_max_at[t]=np.argmax(dp[t,:])
        decode_ids=[None]*n_frame
        decode_ids[-1]=dp_max_at[-1]
        for t in range(n_frame-2,-1,-1):
            decode_ids[t]=pre[t+1,decode_ids[t+1]]
        return [list(result_array[i]) for i in decode_ids]

    def layer_decode(self,prob_list,beat_arr):
        triad_restriction=self.triad_decode(prob_list,beat_arr)
        return self.decode(prob_list,beat_arr,triad_restriction)

    def __get_beat_arr(self,entry,length,use_beats,use_downbeats):
        delta_time=entry.prop.hop_length/entry.prop.sr
        beat_arr=np.ones((length,),dtype=np.int8)
        if(use_beats):
            valid_beats=[(int(np.round(token[0]/delta_time)),int(np.round(token[1]))) for token in entry.beat]
            valid_beats=[(token[0],token[1]) for token in valid_beats if token[0]>=0 and token[0]<beat_arr.shape[0]]
            for i in range(len(valid_beats)-1):
                beat_arr[valid_beats[i][0]+1:valid_beats[i+1][0]]=0
            if(use_downbeats and len(valid_beats)>0):
                num_beat_per_bar=np.max([token[1] for token in valid_beats])
                beat_arr[np.array([token[0] for token in valid_beats])]=4
                beat_arr[np.array([token[0] for token in valid_beats if token[1]==1])]=2
                if(num_beat_per_bar%2==0):
                    beat_arr[np.array([token[0] for token in valid_beats if token[1]==num_beat_per_bar//2+1])]=3
        return beat_arr

    def decode_to_chordlab(self,entry,prob_list,use_layer_decode,use_beats=False,use_downbeats=False):
        beat_arr=self.__get_beat_arr(entry,prob_list[0].shape[0],use_beats=use_beats,use_downbeats=use_downbeats)
        delta_time=entry.prop.hop_length/entry.prop.sr
        decode_tags=self.decode(prob_list,beat_arr) if not use_layer_decode else self.layer_decode(prob_list,beat_arr)
        result=[]
        last_frame=0
        n_frame=len(decode_tags)
        for i in range(n_frame):
            if(i+1==n_frame or decode_tags[i+1]!=decode_tags[i]):
                result.append([last_frame*delta_time,(i+1)*delta_time,decode_tags[i]])
                last_frame=i+1
        return result

    def decode_to_triad_chordlab(self,entry,prob_list,use_beats=False,use_downbeats=False):
        beat_arr=self.__get_beat_arr(entry,prob_list[0].shape[0],use_beats=use_beats,use_downbeats=use_downbeats)
        delta_time=entry.prop.hop_length/entry.prop.sr
        decode_tags=self.triad_decode(prob_list,beat_arr)
        result=[]
        last_frame=0
        n_frame=len(decode_tags)
        for i in range(n_frame):
            if(i+1==n_frame or decode_tags[i+1]!=decode_tags[i]):
                result.append([last_frame*delta_time,(i+1)*delta_time,decode_tags[i]])
                last_frame=i+1
        return decode_tags,result

    def decode_to_decoration_chordlab(self,entry,prob_list,triad_restriction,use_beats=False,use_downbeats=False):
        beat_arr=self.__get_beat_arr(entry,prob_list[0].shape[0],use_beats=use_beats,use_downbeats=use_downbeats)
        delta_time=entry.prop.hop_length/entry.prop.sr
        decode_tags=self.decode(prob_list,beat_arr,triad_restriction)
        result=[]
        last_frame=0
        n_frame=len(decode_tags)
        for i in range(n_frame):
            if(i+1==n_frame or decode_tags[i+1]!=decode_tags[i]):
                result.append([last_frame*delta_time,(i+1)*delta_time,decode_tags[i]])
                last_frame=i+1
        return decode_tags,result

def prob_to_spectrogram(prob_list,ref_chords):
    (result_triad,result_bass,result_7,result_9,result_11,result_13)=prob_list
    new_results=[]
    indices=ref_chords[:,0]>0
    for arr in [result_7,result_9,result_11,result_13]:
        new_result=np.zeros((arr.shape[0],arr.shape[2]))
        new_result[indices,:]=arr[np.arange(ref_chords.shape[0])[indices],(ref_chords[indices,0]-1).astype(np.int)%12,:]
        new_results.append(new_result)
    return np.concatenate((result_triad,result_bass,new_results[0],new_results[1],new_results[2],new_results[3]),
                          axis=1)

