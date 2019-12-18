import os
import re
import glob
from tqdm import tqdm
import numpy as np
import pickle
from multiprocessing import Pool

emb_dir='/mnt/data1/youngsunhere/inferred_embs/kor_dev/ckpt_000510'
emb_list=glob.glob(emb_dir+'/*npy')
print(f"emb_list: {len(emb_list)} embeddings")
wav_src='/mnt/nas/01_ASR/01_korean/01_speech_text/KOR-CLN-V1'

def read_emb(npy_path):
    emb=np.load(npy_path)
    uttfields=re.search(r'(?<=emb_).*(?=.npy)',os.path.basename(npy_path)).group().split('_')
    dbname="_".join(uttfields[:2])
    spkr="_".join(uttfields[2:-1])
    uttname="_".join(uttfields[2:])

    wavname=f"{wav_src}/{dbname}/1/{spkr}/{uttname}.wav"
    return emb,wavname

pool=Pool(5)
result=pool.map(read_emb,emb_list)

print(f"done reading {len(emb_list)} embeddings")

dvector_list=np.array([],dtype=float).reshape(0,256)
label_list=[]

for i in tqdm(result):
    dvector_list=np.vstack([dvector_list,i[0]])
    label_list.append(i[1])

emb_dict={'dvectors':dvector_list,'labels':label_list}

ckpt_epoch=emb_dir.split('/')[-1]
with open(f'kor_dev_{ckpt_epoch}.pickle','wb') as f:
    pickle.dump(emb_dict,f)
