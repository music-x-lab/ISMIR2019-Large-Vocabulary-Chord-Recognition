import datasets
from io_new.downbeat_io import DownbeatIO
import numpy as np
from mir.extractors import ExtractorBase
from mir import io
import librosa
EPS=1e-8

def create_common_intervals(entry):
    print('dealing with %s'%entry.name)
    salami=entry.salami
    chordlab=entry.chordlab
    assert(salami[-1][1]=='end')
    p=0
    total_chord=len(chordlab)

    beats=[]
    suffix_silence_list=[]
    # interval_beat_lengths=[None]*(len(salami)-1)
    bars_with_chords=[]
    for i in range(len(salami)-1):
        token=salami[i][1]
        if(token.split(',')[0].strip() not in ['Z','z','silence','Z\'','z\'']):
            bars_with_chords.append(i)
    assert(len(bars_with_chords)>0)

    last_beat_length=-1
    for i in range(len(salami)-1):
        token=salami[i][1]
        interval_start=salami[i][0]
        interval_end=salami[i+1][0]
        interval_chord=[]
        interval_length=interval_end-interval_start
        if(interval_length<1e-6):
            print('Warning: too short interval found @ %s, %d'%(entry.name,i))
        while(p<total_chord and chordlab[p][1]<=interval_end+EPS):
            interval_chord.append(chordlab[p])
            p+=1
        if(len(interval_chord)==0):
            print('Warning: empty interval found @ %s, %d'%(entry.name,i))
        if(token.split(',')[0].strip() in ['Z','z','silence','Z\'','z\'']):
            if(last_beat_length>0):
                cur_beat_pos=create_constant_speed_beats(last_beat_length,interval_start,interval_end,align_at_end=False)
                beats+=[[beat_pos,-2] for beat_pos in cur_beat_pos]
                test_mono(beats)
            else:
                pass # todo: first N
            continue
        bar_metres,bar_chords=split_salami_chords(token,salami[i][2],salami[i][3])
        valid,bar_beats,beat_length=create_beats_by_bar_info(bar_metres,bar_chords,interval_chord,'subsidiary',interval_start,interval_end)
        if(not valid):
            if(bars_with_chords[0]==i): # intro special deal
                while(len(interval_chord)>1 and interval_chord[0][2]=='N'):
                    interval_chord=interval_chord[1:]
                    interval_start=interval_chord[0][0]
                    valid,bar_beats,beat_length=create_beats_by_bar_info(bar_metres,bar_chords,interval_chord,'subsidiary',interval_start,interval_end)
                    if(valid):
                        break
                cur_beat_pos=create_constant_speed_beats(beat_length,0,interval_start,align_at_end=True)
                beats+=[[beat_pos,-2] for beat_pos in cur_beat_pos]
                beats+=bar_beats
                test_mono(beats)
            elif(bars_with_chords[-1]==i): # end special deal
                original_interval_end=interval_end
                while(len(interval_chord)>1 and interval_chord[-1][2]=='N'):
                    interval_chord=interval_chord[:-1]
                    interval_end=interval_chord[-1][1]
                    valid,bar_beats,beat_length=create_beats_by_bar_info(bar_metres,bar_chords,interval_chord,'subsidiary',interval_start,interval_end)
                    if(valid):
                        break
                beats+=bar_beats
                cur_beat_pos=create_constant_speed_beats(beat_length,interval_end,original_interval_end,align_at_end=False)
                beats+=[[beat_pos,-2] for beat_pos in cur_beat_pos]
                test_mono(beats)
            else:
                beats+=bar_beats
        else:
            beats+=bar_beats
        if(not valid):
            print('Warning: beat decision failed @ %s, %d'%(entry.name,i))
        test_mono(beats)
        # interval_beat_lengths[i]=beat_length
        last_beat_length=beat_length
    if(p!=total_chord):
        print('Some chords are not in any interval @ %s'%(entry.name))
    return beats
    # entry.append_data(beats,BeatLabIO,'beat')

def test_mono(beats):
    for i in range(len(beats)-1):
        assert(beats[i][0]+EPS/2<beats[i+1][0])

def split_salami_chords(raw_all_annotation,metre_up,metre_down):
    tokens=raw_all_annotation.split('|')
    if(len(tokens)==0):
        print('Warning: no chord segment found @ %s'%raw_all_annotation)
    suffix=tokens[-1].strip()
    chord_tokens=tokens[1:-1]
    repeat=1
    if(suffix!=''):
        suffix_tokens=suffix.split(',')
        if(suffix_tokens[0]!='' and suffix_tokens[0][0]=='x'):
            repeat=int(suffix_tokens[0][1:])
    chord_tokens*=repeat

    bar_metres=[]
    bar_chords=[]
    for chord_token in chord_tokens:
        chord_token=chord_token.strip()
        chord_label=chord_token
        if(chord_token.startswith('(')):
            left=chord_token.index('(')
            mid=chord_token.index('/')
            right=chord_token.index(')')
            override_up=int(chord_token[left+1:mid])
            override_down=int(chord_token[mid+1:right])
            bar_metres.append((override_up,override_down))
            chord_label=chord_token[right+1:].strip()
        else:
            bar_metres.append((metre_up,metre_down))
        bar_chords.append(chord_label)
    return bar_metres,bar_chords

def create_beats_by_bar_info(bar_metres,bar_chords,interval_chords,unit_type,interval_start,interval_end):
    # unit_size=8 # 8 means eighth note
    total_units=0
    bar_units=[]
    for i in range(len(bar_chords)):
        bar_metre=bar_metres[i]
        bar_chord=bar_chords[i]
        if(unit_type=='all'):
            bar_units.append(bar_metre[0])
        elif(unit_type=='all_4th'):
            if(4*bar_metre[0]%bar_metre[1]!=0):
                print('Warning: indivisible situation encountered')
            # assert(unit_size*bar_metre[0]%bar_metre[1]==0)
            bar_units.append(bar_metre[0]*4/bar_metre[1])
        elif(unit_type=='subsidiary'):
            assert(bar_metre[1] in [4,8])
            if(bar_metre[1]==4):
                bar_units.append(bar_metre[0]*4/bar_metre[1])
            elif(bar_metre[1]==8):
                assert(bar_metre[0] in [3,5,6,9,12])
                bar_units.append({3:1,5:3,6:2,9:3,12:4}[bar_metre[0]])
        else:
            raise Exception('Not recognized method %s'%unit_type)
        total_units+=bar_units[i]
    if(total_units==0):
        raise Exception('Empty bar found')
    beats=[]
    acc_units=0
    unit_duration=(interval_end-interval_start)/total_units
    for i in range(len(bar_chords)):
        bar_metre=bar_metres[i]
        beats_pos=list(np.arange(acc_units,acc_units+bar_units[i]-EPS)*unit_duration+interval_start)
        beats+=[[beats_pos[j],j%bar_metre[0]+1] for j in range(len(beats_pos))]
        acc_units+=bar_units[i]
    p=0
    def check_near_beat(time,p):
        while(p<len(beats) and beats[p][0]+EPS<time):
            p+=1
        if(p==len(beats)):
            return False,p
        return beats[p][0]-EPS<time,p
    if(len(interval_chords)==0):
        return False,beats,unit_duration
    check,p=check_near_beat(interval_chords[0][0],p)
    if(not check):
        return False,beats,unit_duration
    for i in range(len(interval_chords)-1):
        if(interval_chords[i][2]!=interval_chords[i+1][2]):
            check,p=check_near_beat(interval_chords[i][1],p)
            if(not check):
                return False,beats,unit_duration
    return True,beats,unit_duration

def create_constant_speed_beats(beat_length,interval_start,interval_end,align_at_end):
    beat_count=(interval_end-interval_start)/beat_length
    result=np.arange(beat_count-EPS)*beat_length+interval_start
    if(align_at_end and result.shape[0]>0):
        result+=interval_end-result[-1]-beat_length
    return list(result)

class BeatAnnotationFromBillboard(ExtractorBase):

    def get_feature_class(self):
        return DownbeatIO

    def extract(self,entry,**kwargs):
        return np.array(create_common_intervals(entry))

class TonicAnnotationFromBillboard(ExtractorBase):

    def get_feature_class(self):
        return io.SpectrogramIO

    def extract(self,entry,**kwargs):
        n_frame=entry.n_frame
        result=np.full((n_frame),-1,dtype=np.int16)
        hop_length=entry.prop.hop_length
        sr=entry.prop.sr
        salami=entry.salami
        for i in range(len(salami)-1):
            interval_start=max(0,int(np.round(salami[i][0]*entry.prop.sr/entry.prop.hop_length)))
            interval_end=min(n_frame,int(np.round(salami[i+1][0]*entry.prop.sr/entry.prop.hop_length)))
            interval_key=salami[i][4]
            result[interval_start:interval_end]=interval_key
        return result

class BasicStructureAnnotationFromBillboard(ExtractorBase):

    def get_feature_class(self):
        return io.SpectrogramIO

    def extract(self,entry,**kwargs):
        n_frame=entry.n_frame
        result=np.full((n_frame,3),-1,dtype=np.int16)
        hop_length=entry.prop.hop_length
        sr=entry.prop.sr
        salami=entry.salami
        current_segment=0
        current_repeat=0
        current_voicing=0
        segment_names=['silence','verse','chorus','bridge',True,'intro','outro',False]
        ignore_tag_list=['fade out','fadeout','fade in','instrumental','flute)']
        for i in range(len(salami)-1):
            interval_start=max(0,int(np.round(salami[i][0]*entry.prop.sr/entry.prop.hop_length)))
            interval_end=min(n_frame,int(np.round(salami[i+1][0]*entry.prop.sr/entry.prop.hop_length)))
            if('(voice' in salami[i][1]):
                current_voicing=True
            tokens=[s.strip() for s in salami[i][1].split('|')]
            if(tokens[0]!=''):
                comma_tokens=[s.strip() for s in tokens[0].split(',')]
                if(comma_tokens[0] in ['Z','z','silence','Z\'','z\'']):
                    current_segment=0
                    current_repeat=0
                    continue
                assert(comma_tokens[-1]=='')
                comma_tokens=comma_tokens[:-1]
                for tag in ignore_tag_list:
                    if(tag in comma_tokens):
                        comma_tokens.remove(tag)
                if(len(comma_tokens)>=2):
                    if('chorus' in comma_tokens and 'outro' in comma_tokens):
                        comma_tokens.remove('outro')
                    assert(len(comma_tokens)==2)
                    repeat_char=comma_tokens[0].strip()
                    assert(len(repeat_char.replace('\'',''))==1 and 'A'<=repeat_char[0]<='Y')
                    current_repeat=ord(repeat_char[0])-ord('A')+1
                    segment_str=comma_tokens[1].strip()
                    if(segment_str in segment_names):
                        current_segment=segment_names.index(segment_str)
                    else:
                        current_segment=segment_names.index(current_voicing)

            result[interval_start:interval_end,0]=current_segment
            result[interval_start:interval_end,1]=current_voicing
            result[interval_start:interval_end,2]=current_repeat
            if('voice)' in salami[i][1]):
                current_voicing=False
        return result




def get_simple_joint_framed_downbeat_tempo_annotation(entry,proxy_name):
    n_frame=entry.n_frame
    delta_time=entry.prop.hop_length/entry.prop.sr
    beat=np.array(entry.dict[proxy_name].get(entry))
    beat_time=beat[:,0]
    beat_pos=np.round(beat[:,1]).astype(np.int)
    beat_meters=np.zeros_like(beat_pos)
    beat_frames=np.round(beat_time/delta_time).astype(np.int)
    beat_length=np.zeros_like(beat_pos)
    result=np.zeros((n_frame,3),dtype=np.float32)
    p_back=0
    for i in range(len(beat)):
        if(i<len(beat)-1):
            beat_length[i]=np.round((beat_time[i+1]-beat_time[i])/delta_time)
            if(beat_pos[i+1]==1):
                while(p_back<=i):
                    beat_meters[p_back]=beat_pos[i]
                    p_back+=1
    if(len(beat)>1):
        beat_length[-1]=beat_length[-2]
    while(p_back>0 and p_back<len(beat)):
        beat_meters[p_back]=beat_meters[p_back-1]
        p_back+=1
    for i in range(len(beat)):
        fill_start=0 if i==0 else max(0,beat_frames[i])
        fill_end=n_frame if i==len(beat)-1 else min(n_frame,beat_frames[i+1])
        if(beat_frames[i]>0 and beat_frames[i]<n_frame):
            result[beat_frames[i],0]=beat_pos[i]
        result[fill_start:fill_end,1]=beat_meters[i]
        result[fill_start:fill_end,2]=beat_length[i]
    return result


class SimpleJointFramedDownbeatTempoAnnotation(ExtractorBase):

    def get_feature_class(self):
        return io.SpectrogramIO

    def extract(self,entry,**kwargs):
        proxy_name='beat' if 'source' not in kwargs else kwargs['source']
        return get_simple_joint_framed_downbeat_tempo_annotation(entry,proxy_name)

class SimpleFramedDownbeatAnnotation(ExtractorBase):

    def get_feature_class(self):
        return io.SpectrogramIO

    def extract(self,entry,**kwargs):
        proxy_name='beat' if 'source' not in kwargs else kwargs['source']
        result=get_simple_joint_framed_downbeat_tempo_annotation(entry,proxy_name)
        return result[:,0:2]


class SimpleFramedTempoAnnotation(ExtractorBase):

    def get_feature_class(self):
        return io.SpectrogramIO

    def extract(self,entry,**kwargs):
        proxy_name='beat' if 'source' not in kwargs else kwargs['source']
        result=get_simple_joint_framed_downbeat_tempo_annotation(entry,proxy_name)
        return result[:,2]

