import os
from mir import DataPool,DataEntry,io
from io_new.jams_io import JamsIO
from io_new.chordlab_io import ChordLabIO
from io_new.beatlab_io import BeatLabIO
from io_new.salami_io import SalamiIO
from io_new.complex_chord_io import ComplexChordIO
from io_new.midilab_io import MidiLabIO
from io_new.lyric_io import LyricIO
from io_new.downbeat_io import DownbeatIO

from settings import *
from collections import OrderedDict
from extractors.jam_converter import JamsToChordLabs
from mir.extractors.librosa_extractor import HPSS

def set_default_dataset_properties(dataset):
    dataset.set_property('sr', DEFAULT_SR)
    dataset.set_property('hop_length', DEFAULT_HOP_LENGTH)
    dataset.set_property('beat_hop_length', DEFAULT_BEAT_HOP_LENGTH)
    dataset.set_property('win_size', DEFAULT_WIN_SIZE)
    dataset.set_property('chroma_tuple_size', DEFAULT_CHROMA_TUPLE_SIZE)
    dataset.set_property('chord_dict', DEFAULT_CHORD_DICT)

def create_rwc_dataset(chord_dict_name=DEFAULT_CHORD_DICT):
    rwc=DataPool('rwc')
    set_default_dataset_properties(rwc)
    rwc.set_property('chord_dict',chord_dict_name)
    rwc.append_folder(os.path.join(RWC_DATASET_PATH,'AUDIO'),'.wav',io.MusicIO,'music')
    file_list=os.listdir(os.path.join(RWC_DATASET_PATH,'LAB'))
    file_list.sort()
    assert(len(file_list)==len(rwc.entries))
    for i in range(len(rwc.entries)):
        rwc.entries[i].append_file(os.path.join(RWC_DATASET_PATH,'LAB',file_list[i]),ChordLabIO,'chordlab')
        rwc.entries[i].append_file(os.path.join(RWC_DATASET_PATH,'LAB',file_list[i]),ComplexChordIO,'xchord')
    file_list=os.listdir(os.path.join(RWC_DATASET_PATH,'BEATS'))
    file_list.sort()
    for i in range(len(rwc.entries)):
        rwc.entries[i].append_file(os.path.join(RWC_DATASET_PATH,'BEATS',file_list[i]),DownbeatIO,'beat')
    return rwc

def create_jam_dataset():
    pool=DataPool('jam')
    set_default_dataset_properties(pool)
    f=open(os.path.join(JAM_DATASET_PATH,'audio/filelist.txt'),'r')
    lines=f.readlines()
    f.close()
    for line in lines:
        filename=os.path.basename(line.strip())
        entry=pool.new_entry(filename[:-4])
        music_file=os.path.join(JAM_DATASET_PATH,'audio',filename)
        jam_file=os.path.join(JAM_DATASET_PATH,'references_v2',filename[:-4]+'.jams')
        chordlab_file=os.path.join(JAM_DATASET_PATH,'chordlab',filename[:-4]+'.lab')
        entry.append_file(music_file,io.MusicIO,'music')
        entry.append_file(jam_file,JamsIO,'jam')
        entry.append_file(chordlab_file,ChordLabIO,'chordlab')
        entry.append_file(chordlab_file,ComplexChordIO,'xchord')
    return pool

def create_uspop_dataset():
    uspop=DataPool('uspop')
    set_default_dataset_properties(uspop)
    uspop.append_folder(os.path.join(USPOP_DATASET_PATH,'label'),'.lab',io.ChordIO,'chord')
    uspop.append_folder(os.path.join(USPOP_DATASET_PATH,'label'),'.lab',ChordLabIO,'chordlab')
    uspop.append_folder(os.path.join(USPOP_DATASET_PATH,'label'),'.lab',ComplexChordIO,'xchord')
    uspop.append_folder(os.path.join(USPOP_DATASET_PATH,'audio'),'.mp3',io.MusicIO,'music')
    return uspop


def create_full_dataset():
    pool=create_jam_dataset().join(create_rwc_dataset()).join(create_uspop_dataset())
    return pool

def preprocess_lookup_table():
    lookup_table=OrderedDict()
    jam=create_jam_dataset()
    for entry in jam:
        if("id" in entry.jam.sandbox):
            id=int(entry.jam.sandbox["id"])
            if(id in lookup_table):
                raise Exception("Duplicated ID %d"%id)
            lookup_table[id]=os.path.basename(entry.dict['music'].filepath)
    f=open(os.path.join(BILLBOARD_DATASET_PATH,"lookup.txt"),'w')
    for (k,v) in sorted(lookup_table.items()):
        if(k!=974): # buggy entry
            f.write('%04d\t%s\n'%(k,v))
    f.close()
    print('done')

def preprocess_jam_chordlab():
    jam=create_jam_dataset()
    jam.append_extractor(CQTV2,'cqt')
    # jam.activate_proxy('cqt',thread_number=-1)
    jam.append_extractor(JamsToChordLabs,'chordlab',cache_enabled=False)
    for entry in jam.entries:
        entry.save('chordlab',R'D:\workplace\dataset\chord_data_1217\chordlab\%s.lab'%entry.name[4:])

def create_billboard_dataset(chord_dict_name=DEFAULT_CHORD_DICT,raw=False):
    billboard=DataPool('billboard')
    set_default_dataset_properties(billboard)
    billboard.set_property('chord_dict',chord_dict_name)

    f=open(os.path.join(BILLBOARD_DATASET_PATH,"jams_link.txt"),'r')
    for line in f.readlines():
        line=line.strip()
        if(line==''):
            continue
        tokens=line.split('\t')
        entry=billboard.new_entry(tokens[0])
        entry.append_file(os.path.join(JAM_DATASET_PATH,'audio',tokens[1]),io.MusicIO,'music')
        entry.append_file(os.path.join(BILLBOARD_DATASET_PATH,'LAB',tokens[0],'full.lab'),ChordLabIO,'chordlab')
        entry.append_file(os.path.join(BILLBOARD_DATASET_PATH,'LAB',tokens[0],'full.lab'),ComplexChordIO,'xchord')
        #entry.append_file(os.path.join(BILLBOARD_DATASET_PATH,'MIREX',tokens[0],'majmin.lab'),ChordLabIO,'chordlab_majmin')
        #entry.append_file(os.path.join(BILLBOARD_DATASET_PATH,'MIREX',tokens[0],'majmin.lab'),io.ChordIO,'chord_majmin')
        #entry.append_file(os.path.join(BILLBOARD_DATASET_PATH,'MIREX',tokens[0],'majmin7.lab'),ChordLabIO,'chordlab_majmin7')
        #entry.append_file(os.path.join(BILLBOARD_DATASET_PATH,'MIREX',tokens[0],'majmin7inv.lab'),ChordLabIO,'chordlab_majmin7inv')
        #entry.append_file(os.path.join(BILLBOARD_DATASET_PATH,'MIREX',tokens[0],'majmininv.lab'),ChordLabIO,'chordlab_majmininv')
        entry.append_file(os.path.join(BILLBOARD_DATASET_PATH,'SALAMI',tokens[0],'salami_chords.txt'),SalamiIO,'salami')
    if(not raw):
        billboard.append_folder(os.path.join(BILLBOARD_DATASET_PATH,'tonic'),'.txt',io.SpectrogramIO,'tonic')
        billboard.append_folder(os.path.join(BILLBOARD_DATASET_PATH,'sub_beat'),'.txt',DownbeatIO,'beat')
    return billboard

def create_osu_dataset(chord_dict_name=DEFAULT_CHORD_DICT):
    osu=DataPool('osu')
    set_default_dataset_properties(osu)
    osu.set_property('chord_dict',chord_dict_name)
    # osu.append_folder(os.path.join(MY_DATASET_PATH,'chordlab'),'.lab',io.ChordIO,'chord')
    osu.append_folder(os.path.join(MY_DATASET_PATH,'chordlab'),'.lab',ChordLabIO,'chordlab')
    osu.append_folder(os.path.join(MY_DATASET_PATH,'chordlab'),'.lab',ComplexChordIO,'xchord')
    osu.append_folder(os.path.join(MY_DATASET_PATH,'music'),'.mp3',io.MusicIO,'music')
    osu.append_folder(os.path.join(MY_DATASET_PATH,'keylab'),'.lab',ChordLabIO,'key')
    osu.append_folder(os.path.join(MY_DATASET_PATH,'beatlab'),'.lab',BeatLabIO,'beat')
    return osu


def create_osu_key_dataset(chord_dict_name=DEFAULT_CHORD_DICT):
    osu=DataPool('osu_key')
    set_default_dataset_properties(osu)
    osu.set_property('chord_dict',chord_dict_name)
    osu.append_folder(os.path.join(OSU_KEY_DATASET_PATH,'music'),'.mp3',io.MusicIO,'music')
    osu.append_folder(os.path.join(OSU_KEY_DATASET_PATH,'keylab'),'.lab',ChordLabIO,'key')
    osu.append_folder(os.path.join(OSU_KEY_DATASET_PATH,'beatlab'),'.lab',BeatLabIO,'beat')
    for entry in osu.entries:
        entry.append_file('data/chord_placeholder.lab',ComplexChordIO,'xchord')
    return osu


def create_cb_dataset(limit=-1,debug=False,bootstrap_round=2):
    cb=DataPool('cb2')
    set_default_dataset_properties(cb)
    f=open(os.path.join(CB_DATASET_PATH,'dataset.txt'),'r')
    name_list=f.readlines()
    f.close()
    id=0
    for item in name_list[1:]:
        token=item.strip().split('\t')
        folder_name=token[1]
        folder_path=os.path.join(CB_DATASET_PATH,folder_name)
        entry=cb.new_entry(folder_name)
        entry.append_file(os.path.join(folder_path,'music.mp3'),io.MusicIO,'music')
        entry.append_file(os.path.join(folder_path,'original_mp3.mp3'),io.MusicIO,'music_av')
        entry.append_file(os.path.join(folder_path,'vocal.wav'),io.MusicIO,'music_v')
        # entry.append_file(os.path.join(folder_path,'music.mp3.chroma'),io.ChromaIO,'chroma_aligned')
        if(os.path.exists(os.path.join(folder_path,'midi.lab.corrected.lab'))):
            entry.append_file(os.path.join(folder_path,'midi.lab.corrected.lab'),MidiLabIO,'midilab_old')
            entry.append_file(os.path.join(CB_BOOTSTRAP_DATASET_PATH,folder_name,'midilab_%d.lab'%bootstrap_round),MidiLabIO,'midilab')
            entry.append_file(os.path.join(CB_BOOTSTRAP_DATASET_PATH,folder_name,'lyric_%d.lab'%bootstrap_round),LyricIO,'lyric')
        if(os.path.exists(os.path.join(folder_path,'original_mp3.mp3.det.0.40.wav'))):
            entry.append_file(os.path.join(folder_path,'original_mp3.mp3.det.0.40.wav'),io.MusicIO,'music_m40')
            if(debug):
                print(entry.name)
                assert(os.path.getsize(os.path.join(folder_path,'original_mp3.mp3.det.0.40.wav'))
                       ==os.path.getsize(os.path.join(folder_path,'vocal.wav')))
        entry.append_file(os.path.join(folder_path,'lyric.lab'),LyricIO,'lyric_old')
        entry.append_file(os.path.join(folder_path,'midi.lab'),MidiLabIO,'midilab_raw')
        id+=1
        if(id==limit):
            break
    return cb

def create_cb_1000_dataset(limit=-1,debug=False,bootstrap_round=2):
    cb=DataPool('cb2')
    set_default_dataset_properties(cb)
    f=open(os.path.join(CB_DATASET_PATH,'dataset1000.txt'),'r')
    name_list=f.readlines()
    f.close()
    id=0
    for item in name_list[1:]:
        token=item.strip().split(' - ')
        folder_name=token[0]+' - '+token[1]
        folder_path=os.path.join(CB_DATASET_PATH,folder_name)
        entry=cb.new_entry(folder_name)
        entry.append_file(os.path.join(folder_path,'music.mp3'),io.MusicIO,'music')
        entry.append_file(os.path.join(folder_path,'original_mp3.mp3'),io.MusicIO,'music_av')
        entry.append_file(os.path.join(folder_path,'vocal.wav'),io.MusicIO,'music_v')
        # entry.append_file(os.path.join(folder_path,'music.mp3.chroma'),io.ChromaIO,'chroma_aligned')
        if(os.path.exists(os.path.join(folder_path,'midi.lab.corrected.lab'))):
            entry.append_file(os.path.join(folder_path,'midi.lab.corrected.lab'),MidiLabIO,'midilab_old')
            #entry.append_file(os.path.join(CB_BOOTSTRAP_DATASET_PATH,folder_name,'midilab_%d.lab'%bootstrap_round),MidiLabIO,'midilab')
            #entry.append_file(os.path.join(CB_BOOTSTRAP_DATASET_PATH,folder_name,'lyric_%d.lab'%bootstrap_round),LyricIO,'lyric')
        if(os.path.exists(os.path.join(folder_path,'original_mp3.mp3.det.0.40.wav'))):
            entry.append_file(os.path.join(folder_path,'original_mp3.mp3.det.0.40.wav'),io.MusicIO,'music_m40')
            if(debug):
                print(entry.name)
                assert(os.path.getsize(os.path.join(folder_path,'original_mp3.mp3.det.0.40.wav'))
                       ==os.path.getsize(os.path.join(folder_path,'vocal.wav')))
        entry.append_file(os.path.join(folder_path,'lyric.lab'),LyricIO,'lyric_old')
        entry.append_file(os.path.join(folder_path,'midi.lab'),MidiLabIO,'midilab_raw')
        id+=1
        if(id==limit):
            break
    return cb

def create_valid_cb_dataset():
    cb=create_cb_dataset()
    cb_valid=DataPool('cb2')
    for entry in cb.entries:
        if(entry.has('midilab')):
            cb_valid.add_entry(entry)
    return cb_valid

def create_beatles_dataset(raw=False):
    beatles=DataPool('beatles')
    set_default_dataset_properties(beatles)
    if(raw==False):
        beatles.append_folder(os.path.join(BEATLES_DATASET_PATH,'beat'),'.txt.fix',DownbeatIO,'beat')
        beatles.append_folder(os.path.join(BEATLES_DATASET_PATH,'chordlab'),'.lab.fix',ChordLabIO,'chordlab')
        beatles.append_folder(os.path.join(BEATLES_DATASET_PATH,'chordlab'),'.lab.fix',ComplexChordIO,'xchord')
    else:
        beatles.append_folder(os.path.join(BEATLES_DATASET_PATH,'beat'),'.txt',DownbeatIO,'beat')
        beatles.append_folder(os.path.join(BEATLES_DATASET_PATH,'chordlab'),'.lab',ChordLabIO,'chordlab')
    beatles.append_folder(os.path.join(BEATLES_DATASET_PATH,'audio'),'.wav',io.MusicIO,'music')
    return beatles

def create_joint_beat_chord_dataset():
    billboard=create_billboard_dataset()
    beatles=create_beatles_dataset()
    rwc=create_rwc_dataset()
    return billboard.join(rwc).join(beatles)

if __name__ == '__main__':
    from extractors.beat_analysis import analyze_double_speed_error
    cb=create_valid_cb_dataset()
    from extractors.cqt import CQTV2
    from extractors.key_preprocess import FramedKey
    from mir.extractors.misc import FrameCount
    from extractors.madmom_extractor import DBNDownBeatExtractor,DBNDownBeatProbability
    from extractors.beat_preprocess import BeatAnnotationFromBillboard,TonicAnnotationFromBillboard,BasicStructureAnnotationFromBillboard
    from extractors.complex_chord_preprocess import chordlab_to_flat_vocab,get_flat_chord_vocab
    from mir.extractors.librosa_extractor import HPSS
    chord_dict=get_flat_chord_vocab('data/submission_chord_list.txt')

    cb.append_extractor(DBNDownBeatExtractor,'beat',source='music')
    cb.append_extractor(DBNDownBeatProbability,'prob',source='music')
    for entry in cb.entries:
        print(entry.name)

        #entry.append_extractor(FrameCount,'n_frame',source='cqt')
        #tags=chordlab_to_flat_vocab(entry,entry.chordlab,chord_dict)
        #entry.append_data(tags,io.SpectrogramIO,'tags')
        #entry.visualize(['chordlab','tags'])
        if(analyze_double_speed_error(entry.beat)):
            entry.append_extractor(HPSS,'music_h',margin=1.0)
            entry.append_extractor(DBNDownBeatProbability,'prob_hpss',source='music_h')
            entry.visualize(['beat','prob','midilab','prob_hpss'])
