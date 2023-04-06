import numpy as np
import os
from complex_chord import ChordTypeLimit,Chord,NUM_TO_ABS_SCALE
import mir_eval
from settings import JAM_DATASET_PATH
import matplotlib.pyplot as plt

MAX_CLASS_SIZE=13
chord_limit=ChordTypeLimit(
    triad_limit=6,
    seventh_limit=3,
    ninth_limit=3,
    eleventh_limit=2,
    thirteenth_limit=2
)

def get_names_values_to_plot(total,correct,l0,categories):
    values=[]
    sample_counts=[]
    l0_counts=[]
    names=[]
    for k in categories:
        if(k==0):
            range_obj=range(chord_limit.triad_limit+1)
            name_list=['N','maj','min','sus4','sus2','dim','aug']
        elif(k==1):
            range_obj=[x+1 for x in [2,3,4,5,7,9,10,11]]
            name_list=['/2','/b3','/3','/4','/5','/6','/b7','/7']#['N']+NUM_TO_ABS_SCALE
        elif(k==2):
            range_obj=range(1,chord_limit.seventh_limit+1)
            name_list=['+7','+b7','+bb7']
        elif(k==3):
            range_obj=range(1,chord_limit.ninth_limit+1)
            name_list=['+9','+#9','+b9']
        elif(k==4):
            range_obj=range(1,chord_limit.eleventh_limit+1)
            name_list=['+11','+#11']
        elif(k==5):
            range_obj=range(1,chord_limit.thirteenth_limit+1)
            name_list=['+13','+b13']
        else:
            raise NotImplementedError()
        for i,j in enumerate(range_obj):
            names.append(name_list[i])
            values.append(correct[k,j]/total[k,j])
            sample_counts.append(total[k,j]*12)
            l0_counts.append(l0[k,j])
    return names,values,sample_counts,l0_counts

def read_chordlab_from_file(file_name):
    f = open(file_name, 'r')
    content = f.read()
    lines=content.split('\n')
    f.close()
    result=[]
    for i in range(len(lines)):
        line=lines[i].strip()
        if(line==''):
            continue
        tokens=line.split('\t')
        assert(len(tokens)==3)
        result.append([float(tokens[0]),float(tokens[1]),tokens[2]])
    return result

def process_folder(folder_est,folder_ref):
    ref_files=os.listdir(folder_ref)
    result=[]
    for file in ref_files:
        try:
            ref=read_chordlab_from_file(os.path.join(folder_ref,file))
            est=read_chordlab_from_file(os.path.join(folder_est,file))
            result.append((est,ref))
        except:
            print('Warning: comparison failure: %s'%file)
    return result

def split_chordlab(chordlab):
    return (np.array([[data[0],data[1]] for data in chordlab],dtype=np.float64),[data[2] for data in chordlab])

def compute_part_recall_single(chordlab_est,chordlab_ref):
    total=np.zeros((6,MAX_CLASS_SIZE))
    correct=np.zeros((6,MAX_CLASS_SIZE))
    (gd_intervals,gd_labels) = split_chordlab(chordlab_ref)
    (est_intervals,est_labels) = split_chordlab(chordlab_est)
    est_intervals,est_labels = mir_eval.util.adjust_intervals(est_intervals,est_labels,gd_intervals.min(),gd_intervals.max(),start_label='X',end_label='X')
    (intervals,gd_labels,est_labels)=mir_eval.util.merge_labeled_intervals(gd_intervals,gd_labels,est_intervals,est_labels)
    durations = mir_eval.util.intervals_to_durations(intervals)
    for (duration,gd_label,est_label) in zip(durations,gd_labels,est_labels):
        ref_xchord=Chord(gd_label).to_numpy()
        est_xchord=Chord(est_label).to_numpy()
        for k in range(6):
            ref_id=ref_xchord[k]
            est_id=est_xchord[k]
            if(k==0):
                ref_id=(ref_id+11)//12
                est_id=(est_id+11)//12
            #else:
            #    if((ref_xchord[0]+11)//12!=(est_xchord[0]+11)//12):
            #        continue #todo: counting error
            if(k==1):
                ref_id+=1
                est_id+=1
                if(ref_id>0):
                    ref_id=((ref_id-ref_xchord[0])%12+12)%12+1
                if(est_id>0):
                    est_id=((est_id-est_xchord[0])%12+12)%12+1
            if(ref_id>=0):
                total[k,ref_id]+=duration
            if(ref_id==est_id):
                correct[k,ref_id]+=duration
    return total,correct

def compute_part_recall(pool):
    total=np.zeros((6,MAX_CLASS_SIZE))
    correct=np.zeros((6,MAX_CLASS_SIZE))
    l0=np.zeros((6,MAX_CLASS_SIZE))
    for (chordlab_est,chordlab_ref) in pool:
        cur_total,cur_correct=compute_part_recall_single(chordlab_est,chordlab_ref)
        total+=cur_total
        correct+=cur_correct
        l0+=(cur_total>0)
    return total,correct,l0

def plot_result(names,values,sample_counts,l0_counts):
    x=np.arange(len(names))
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax2.bar(x,l0_counts,color='b', zorder=1)
    ax2.set_ylabel(r"Number of Appearance in Distinct Songs", labelpad=10)
    #ax2.set_yscale('log')
    ax.set_zorder(ax2.get_zorder()+1)
    ax.patch.set_visible(False)
    #bar = ax.bar(x,values,0.8,align="center")
    plot = ax.plot(x,values,color='r',marker='o', zorder=100)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title(r"Evaluation on Chord Components")
    ax.set_ylabel(r"Chord Component Recall", labelpad=10)
    ax.set_xlabel("Chord Component Label",labelpad=10)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_ylim([0.0,1.0])
    plt.show()


def plot_multiple_results(model_template,legend_list,name_list,plot_id):
    from figures import FIG_OUTPUT_PATH
    from mir import cache
    try:
        values_list,names,sample_counts,l0_counts=cache.load('figure_data_upd2')
    except:
        values_list=[]
        for filename in name_list:
            pool=process_folder((model_template%filename).replace('[d]','%d'),
                                      os.path.join(JAM_DATASET_PATH,'chordlab')+'/')
            total,correct,l0=compute_part_recall(pool)
            names,values,sample_counts,l0_counts=get_names_values_to_plot(total,correct,l0,[0,1,2,3,4,5])
            values_list.append(values)
        cache.save((values_list,names,sample_counts,l0_counts),'figure_data_upd2')
    x=np.arange(len(names))
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(figsize=(16,4))
    if(plot_id==1):
        ax2 = ax.twinx()
        ax2.bar(x,l0_counts,color='#cccccc', zorder=1)
        ax2.set_ylabel(r"Number of Appearances in Distinct Songs", labelpad=10)
        #ax2.set_yscale('log')
        ax.set_zorder(ax2.get_zorder()+1)
        ax.patch.set_visible(False)
        #bar = ax.bar(x,values,0.8,align="center")
        for i,filename in enumerate(name_list):
            plot = ax.plot(x,values_list[i],marker='ov^s*<>'[i],markersize=8, zorder=100,label=legend_list[i])
        ax.set_ylabel(r"Chord Component Recall", labelpad=10)
        ax.set_ylim([0.0,1.0])
        ax.legend()
    else:
        ax.bar(x,l0_counts,color='#9c9c9c', zorder=1)
        ax.set_ylabel(r"Number of Appearances in Distinct Songs", labelpad=10)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    #ax.set_title(r"Evaluation on Chord Components")
    ax.set_xlabel("Chord Component Label",labelpad=10)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    if(plot_id==1):
        fig.savefig(os.path.join(FIG_OUTPUT_PATH,'component_recall.pdf'), transparent=True, pad_inches=0,bbox_inches='tight')
    else:
        fig.savefig(os.path.join(FIG_OUTPUT_PATH,'sample_count_song_level.pdf'), transparent=True, pad_inches=0,bbox_inches='tight')

    plt.show()



if __name__ == '__main__':
    plot_multiple_results("output/output_joint_chord_net_ismir_naive_v1.0_reweight(%.1f,%.1f)_s[d].best_hmm_full/jam/",
                          ['no_reweight','(0.3,10.0)','(0.5,10.0)','(0.7,20.0)','(1.0,20.0)'],[(0.0,10.0),(0.3,10.0),(0.5,10.0),(0.7,20.0),(1.0,20.0)],
                          plot_id=1)
    #pool=process_folder("output/output_joint_chord_net_ismir_naive_v1.0_reweight(1.0,20.0)_s%d.best_hmm_full/jam/",
    #                          os.path.join(JAM_DATASET_PATH,'chordlab')+'/')
    #total,correct,l0=compute_part_recall(pool)
    #names,values,sample_counts,l0_counts=get_names_values_to_plot(total,correct,l0,[0,1,2,3,4,5])
    #print(names,values,sample_counts,l0_counts)
    #plot_result(names,values,sample_counts,l0_counts)
