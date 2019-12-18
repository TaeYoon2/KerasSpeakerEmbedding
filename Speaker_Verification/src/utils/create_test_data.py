import glob
import numpy as np
import json
import time
import datetime
import os

#create test batches used during training

def create_test_data(test_json_path,test_data_path,batch_creator,num_spkrs_in_batch,num_utts_per_spkr,num_batches):
    # if batches npy already exist, simply load them
    test_data_list = glob.glob(test_data_path+"/batch*.npy")
    if len(test_data_list) == num_batches: # load pre-saved test batches
        print(f"[skip create_test_data] {num_batches} test batches already exist ")
        batch_list = []
        batch_spkr_utt_lab_list=[]

        for npyfile in test_data_list:
            batch_and_labs = np.load(npyfile,allow_pickle=True)
            batch_list.append(batch_and_labs[0])
            batch_spkr_utt_lab_list.append(batch_and_labs[1])

    else: # create new batches for test
        #delete whatever is already in the directory
        [ os.remove(npyfile) for npyfile in test_data_list ]
        print(f"[create_test_data] creating test batches... ")

        # create  batch_mel_paths_list = # select [[ 640 * 10 ], [ 640 * 10 ] ,[ 640 * 10 ]]

        with open(test_json_path) as f:
            test_json = json.load(f)
        print(f"{test_json_path} is LOADED")

        spkr_list=test_json["spkrs"]
        utt_list = test_json["utts"]
        selected_spkr_list = np.random.choice(spkr_list, size=num_spkrs_in_batch*num_batches, replace=False)
        print(f"selected_spkr_list: {len(selected_spkr_list)} {len(list(set(selected_spkr_list)))}")

        selected_utt_list = []
        for spkr in selected_spkr_list:
            spkr_utt_list = utt_list[spkr]
            random_chosen=np.random.choice(spkr_utt_list, size=num_utts_per_spkr, replace=False)
            selected_utt_list.append(random_chosen)
        print(f"selected_utt_list: {len(selected_utt_list)}")

        batch_mel_paths_list = []
        for i in range(num_batches):
            i+=1
            batch_mel_path = selected_utt_list[num_spkrs_in_batch*(i-1):num_spkrs_in_batch*(i)]
            batch_mel_paths_list.append(batch_mel_path)

        batch_list=[]
        batch_spkr_utt_lab_list=[]
        for idx,mlist_by_spkr in enumerate(batch_mel_paths_list):
            # start2=time.time()
            start1= time.time()
            results=[]
            for mlist in mlist_by_spkr:
                for mel_path in mlist:
                    mel, len_mel, filename = batch_creator.load_single_mel_npy(mel_path)
                    results.append([mel, len_mel, filename])

            timenow= datetime.datetime.now().strftime("%H:%M:%S")
            print(f"[ creating test batches / current time:{timenow} ] reading 640 mel npy files took: {time.time()-start1:0.5f} secs")

            datadict=dict()
            datadict['mel'] = [i[0] for i in results]
            datadict['mel_len'] = [i[1] for i in results]
            datadict['file_name'] = [i[2] for i in results]

            one_batch = batch_creator.create_batch(datadict)
            one_batch = one_batch[np.newaxis,:]
            batch_list.append(one_batch)
            one_batch_spkr_utt_lab = datadict['file_name']
            batch_spkr_utt_lab_list.append(one_batch_spkr_utt_lab)
            np.save(os.path.join(test_data_path,f"batch_{idx:02d}.npy"),[one_batch,one_batch_spkr_utt_lab])

    return batch_list,batch_spkr_utt_lab_list
