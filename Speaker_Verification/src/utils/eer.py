import numpy as np
import glob
import re
import os
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from tqdm import tqdm
import time
from multiprocessing import Pool

import csv



def calculate_eer(inferred_emb_list, all_labels):
    print("started calculate_eer")
    start_dot = time.time()

    # score matrix between embeddings
    score = np.dot(inferred_emb_list, inferred_emb_list.T) # (1920, 1920)
    #print(f"finished np.dot {time.time()-start_dot} secs")

    # speaker matching matrix
    label = np.zeros(score.shape)
    # speaker labels
    spkr_label = ["_".join(i.split('_')[1:-1]) for i in all_labels]

    
    for i in tqdm(range(label.shape[0])):
	    # row == column, all the columns matched with each row speaker plus one
        idx = tuple([n for n, x in enumerate(spkr_label) if x == spkr_label[i]])
        label[i, idx] += 1
    #print(f"done with forloop {time.time()}")

    # except for self label & score
    label = label[~np.eye(label.shape[0],dtype=bool)]
    #print(f"done with label {time.time()}")
    score = score[~np.eye(score.shape[0],dtype=bool)]
    #print(f"done with score {time.time()}")

    # get ROC Curve
    fpr, tpr, _ = roc_curve(label, score)
    #print(f"done with roc {time.time()}")

    # Find a root of a function in given interval, EER
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    #print(f"done with brentq {time.time()}")

    return eer

def read_npy(path):
   emb = np.load(path)
   lab = os.path.basename(path).split('.npy')[0]
   return emb, lab

def get_eer_from_checkpoint(emb_indir,num_pool=5):
    # list of all the embedding arrays
    # utt 개수가 최소인 화자에 맞춰서 화자별 utt 개수 선정 (공정한 eer계산위함)
    min_utt_ct,data_dict = get_min_utt_ct(emb_indir)
    print(f"spkr {len(data_dict.keys())} min {min_utt_ct}")
    emb_npy_list=[]
    for spkr in data_dict.keys():
        emb_npy_list.extend(data_dict[spkr][:min_utt_ct])
    print(len(emb_npy_list))
    emb_list=[]
    utt_label_list=[]

    # load the embedding arrays
    pool=Pool(num_pool)
    result=pool.map(read_npy,emb_npy_list)

    # map pooled result : (N,1,256) => (N,256)
    emb_array = np.squeeze([i[0] for i in result])
    lab_array =[i[1] for i in result]

    eer = calculate_eer(emb_array,lab_array)
    return eer

def get_eer_over_range(embedding_path, end, start=1, step=1):
    with open("eer_from_{}_to_{}.csv".format(start, end), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["epoch", ""])
        for i in range(start, end+1, step):
            eer = get_eer_from_checkpoint(embedding_path, i)
            writer.writerow([i, eer])


def get_min_utt_ct(emb_root):
    test_dict=dict()
    for i in glob.glob(emb_root+'/*npy'):
        # if os.path.basename(i)[0]=='i':
        #     spkr=os.path.basename(i).split('_')[0]
        # else:
        #     spkr=os.path.basename(i).split('-')[0]    
        spkr="_".join(os.path.basename(i).split('_')[:-1])
        #print(spkr)
        if spkr not in list(test_dict.keys()):
            test_dict[spkr]=[i]
        else:
            test_dict[spkr].append(i)
    
    utt_ct_by_spkr=list()
    for i in test_dict.keys():
        utt_ct_by_spkr.append(len(test_dict[i]))
    return min(utt_ct_by_spkr),test_dict

            
if __name__ == "__main__":
    
    embedding_path = '/mnt/data1/youngsunhere/inferred_embs/kor_test/ckpt_001716'
    eer = get_eer_from_checkpoint(embedding_path)
    print(eer)
