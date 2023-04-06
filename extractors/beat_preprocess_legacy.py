import datasets
from io_new.beatlab_io import BeatLabIO
import numpy as np
from mir.extractors import ExtractorBase
from io_new.beat_align_io import BeatAlignCQTIO
from io_new.chordlab_io import ChordLabIO
from io_new.list_io import ListIO
from mir import io
import librosa
from mir.chord import Chord,ChordTypeComplexity
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
                beats+=create_constant_speed_beats(last_beat_length,interval_start,interval_end,align_at_end=False)
                test_mono(beats)
            else:
                pass # todo: first N
            continue
        bar_metres,bar_chords=split_salami_chords(token,salami[i][2],salami[i][3])
        valid,bar_beats,beat_length=create_beats_by_bar_info(bar_metres,bar_chords,interval_chord,4,interval_start,interval_end)
        if(not valid):
            if(bars_with_chords[0]==i): # intro special deal
                while(len(interval_chord)>1 and interval_chord[0][2]=='N'):
                    interval_chord=interval_chord[1:]
                    interval_start=interval_chord[0][0]
                    valid,bar_beats,beat_length=create_beats_by_bar_info(bar_metres,bar_chords,interval_chord,4,interval_start,interval_end)
                    if(valid):
                        break
                beats+=create_constant_speed_beats(beat_length,0,interval_start,align_at_end=True)
                beats+=bar_beats
                test_mono(beats)
            elif(bars_with_chords[-1]==i): # end special deal
                original_interval_end=interval_end
                while(len(interval_chord)>1 and interval_chord[-1][2]=='N'):
                    interval_chord=interval_chord[:-1]
                    interval_end=interval_chord[-1][1]
                    valid,bar_beats,beat_length=create_beats_by_bar_info(bar_metres,bar_chords,interval_chord,4,interval_start,interval_end)
                    if(valid):
                        break
                beats+=bar_beats
                beats+=create_constant_speed_beats(beat_length,interval_end,original_interval_end,align_at_end=False)
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
        assert(beats[i]+EPS/2<beats[i+1])

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

def create_beats_by_bar_info(bar_metres,bar_chords,interval_chords,unit_size,interval_start,interval_end):
    # unit_size=8 # 8 means eighth note
    total_units=0
    bar_units=[]
    for i in range(len(bar_chords)):
        bar_metre=bar_metres[i]
        bar_chord=bar_chords[i]
        if(unit_size*bar_metre[0]%bar_metre[1]!=0):
            print('Warning: indivisible situation encountered')
        # assert(unit_size*bar_metre[0]%bar_metre[1]==0)
        bar_units.append(bar_metre[0]*unit_size/bar_metre[1])
        total_units+=bar_units[i]
    if(total_units==0):
        raise Exception('Empty bar found')
    beats=[]
    acc_units=0
    unit_duration=(interval_end-interval_start)/total_units
    for i in range(len(bar_chords)):
        beats+=list(np.arange(acc_units,acc_units+bar_units[i]-EPS)*unit_duration+interval_start)
        acc_units+=bar_units[i]
    p=0
    def check_near_beat(time,p):
        while(p<len(beats) and beats[p]+EPS<time):
            p+=1
        if(p==len(beats)):
            return False,p
        return beats[p]-EPS<time,p
    check,p=check_near_beat(interval_chords[0][0],p)
    if(not check):
        return None
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
        return BeatLabIO

    def extract(self,entry,**kwargs):
        return create_common_intervals(entry)

class BeatAlignCQT(ExtractorBase):

    def get_feature_class(self):
        return BeatAlignCQTIO

    def extract(self,entry,**kwargs):
        assert(entry.prop.hop_length==256)
        div_count=kwargs['div']
        beat=entry.beat
        cqt=entry.cqt
        frame_size=entry.prop.hop_length/entry.prop.sr
        def get_range_cqt(cqt,a,b):
            left=max(0,int(np.ceil(a)))
            right=min(cqt.shape[0],1+int(np.floor(b)))
            if(right>left):
                result=cqt[left:right,:].sum(axis=0)
            else:
                result=np.zeros(cqt.shape[1],dtype=np.float32)
            if(right>=left):
                if(left>0):
                    result+=cqt[left-1,:]*(left-a)
                if(right<cqt.shape[0]):
                    result+=cqt[right,:]*(b-right)
            elif(right+1==left and right>=0 and right<cqt.shape[0]):
                return cqt[right,:]

            return result/(b-a)
        result=np.zeros((len(beat)-1,div_count,cqt.shape[1]),dtype=np.float32)
        for i in range(len(beat)-1):
            start_frame=beat[i]/frame_size
            end_frame=beat[i+1]/frame_size
            interval=(end_frame-start_frame)/div_count
            # print(i,start_frame,end_frame)
            for d in range(div_count):
                result[i,d,:]=get_range_cqt(cqt,start_frame+interval*d,start_frame+interval*(d+1))
        return result

class BeatAlignCQTV2(ExtractorBase):

    def get_feature_class(self):
        return BeatAlignCQTIO

    def extract(self,entry,**kwargs):
        div_count=kwargs['div']
        beat=entry.beat
        cqt=librosa.core.hybrid_cqt(entry.music,
                                bins_per_octave=36,
                                fmin=librosa.note_to_hz('F#0'),
                                n_bins=288,
                                tuning=None,
                                hop_length=128).T
        frame_size=128/entry.prop.sr
        def get_range_cqt(cqt,a,b):
            left=max(0,int(np.ceil(a)))
            right=min(cqt.shape[0],1+int(np.floor(b)))
            if(right>left):
                result=cqt[left:right,:].sum(axis=0)
            else:
                result=np.zeros(cqt.shape[1],dtype=np.float32)
            if(right>=left):
                if(left>0):
                    result+=cqt[left-1,:]*(left-a)
                if(right<cqt.shape[0]):
                    result+=cqt[right,:]*(b-right)
            elif(right+1==left and right>=0 and right<cqt.shape[0]):
                return cqt[right,:]

            return result/(b-a)
        result=np.zeros((len(beat)-1,div_count,cqt.shape[1]),dtype=np.float32)
        for i in range(len(beat)-1):
            start_frame=beat[i]/frame_size
            end_frame=beat[i+1]/frame_size
            interval=(end_frame-start_frame)/div_count
            # print(i,start_frame,end_frame)
            for d in range(div_count):
                result[i,d,:]=get_range_cqt(cqt,start_frame+interval*d,start_frame+interval*(d+1))
        return result

class BeatAlignChord(ExtractorBase):

    def get_feature_class(self):
        return ChordLabIO

    def extract(self,entry,**kwargs):
        chordlab=entry.dict[kwargs['chordlab']].get(entry)
        beat=entry.beat
        p=0
        result=[]
        for i in range(len(beat)-1):
            start_time=beat[i]
            end_time=beat[i+1]
            current_chord=''
            while(p<len(chordlab) and chordlab[p][1]<start_time+1e-6):
                p+=1
            while(p<len(chordlab) and chordlab[p][0]<end_time-1e-6):
                if(current_chord==''):
                    current_chord=chordlab[p][2]
                elif(current_chord!=chordlab[p][2]):
                    current_chord='X'
                p+=1
            if(current_chord==''):
                current_chord='?'
            if(p>0):
                p-=1
            result.append([start_time,end_time,current_chord])
        return result

class SimpleBeatAlignChordID(ExtractorBase):
    def get_feature_class(self):
        return ListIO

    def extract(self,entry,**kwargs):
        return [Chord.from_string(c[2],ChordTypeComplexity.full).to_id() for c in entry.chordlab_majmin_beat]

class SimpleChordIDToBeatAlign(ExtractorBase):
    def get_feature_class(self):
        return ChordLabIO

    def extract(self,entry,**kwargs):
        source=entry.dict[kwargs['list']].get(entry)
        beat=entry.beat
        result=[]
        for i in range(len(beat)-1):
            start_time=beat[i]
            end_time=beat[i+1]
            result.append([start_time,end_time,Chord.from_id(source[i]).to_string(ChordTypeComplexity.full)])
        return result
