import os
import json
import librosa
import requests
import numpy as np


db_root = "/mnt/data1/sh77/speaker_id/dev/namz_ko_speaker/origin/"
save_root = "/mnt/data1/sunkist/projects/mz_kor_embeddings/"

def process_wav(path):
    try:
    	sig, sr = librosa.load(path, sr=None)
    	sig = np.expand_dims(sig,0)
    	r = requests.post('http://192.168.124.28:50599/v1/models/ge2e:predict',
    	                          json={"inputs" :  sig.tolist() })
    	embedding = r.json()["outputs"]
    except KeyError as e:
    	print("There is no key, ",e)
    	return None
    except json.decoder.JSONDecodeError as e:
    	print(e)
    	print(r.content)
    	return None
    return embedding


for each_db in os.listdir(db_root):
    print("DB_NAME : ", each_db)
    for (path, dir, files) in os.walk(os.path.join(db_root, each_db)):
        speaker = path.split("/")[-1]
        for filename in files:
            base, ext = os.path.splitext(filename)
            if ext == '.wav':
                source = os.path.join(db_root, each_db, speaker, filename)
                to_be_saved = os.path.join(save_root, each_db, speaker)
                target = os.path.join(save_root, each_db, speaker, base+".npy")
                # 경로 없을 시 생성
                if not os.path.exists(to_be_saved):
                    os.makedirs(to_be_saved, exist_ok=True)
                # 아직 처리 안했으면
                if not os.path.exists(target):
                    embedding = process_wav(source)
                    if embedding is not None:
                    	np.save(target, embedding)