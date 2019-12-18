# random_selector

# preprocessor : GE2E
import os
import json
import random
import re
import glob

class RandomSelector(object):
    def __init__(self, config):

        #self.TOTAL_NUM_SPK = config["total_num_spk"]
        #self.TOTAL_NUM_DATASET = config["total_num_dataset"]
        #self.PART_TOTAL_NUM_SPK = config["part_total_num_spk"]
        #self.TOTAL_NUM_UTT = config["total_num_utt"]

        self.PATH_INDEX_JSON_DEV = config["path_index_json"]+"_dev.json"
        self.PATH_INDEX_JSON_TEST = config["path_index_json"]+"_test.json"
        self.DATA_DIR_PATH = config["data_dir_path"]
        self.BATCH_NUM_SPK = config["num_speakers"]
        self.BATCH_NUM_UTT = config["num_utterance"]
        self.PATH_INDEX_DEV = None
        self.PATH_INDEX_TEST = None

        if not os.path.exists(self.PATH_INDEX_JSON_DEV):
            print("path_index.json DOES NOT EXIST. Creating new ...")
            dblist = glob.glob(self.DATA_DIR_PATH+'/*')
            
            dict_list=self.create_db_dict_list(dblist)
            dict_list_test_dev=self.split_dev_test(dict_list,0.03)

            _ = self.merge_index_dicts_to_jsonfile(dict_list_test_dev,"test")
            _ = self.merge_index_dicts_to_jsonfile(dict_list_test_dev,"dev")

        with open(self.PATH_INDEX_JSON_DEV) as f:
            # load dev json
            self.PATH_INDEX_DEV = json.load(f)
            print(f"{self.PATH_INDEX_JSON_DEV} JSON is LOADED")

    def create_db_dict_list(self,dblist):

        dict_list=[]
        for db in dblist:
            print(f"create_db_dict_list {db}")
            tmp_dict = self.save_path_index_3_json_spkr_full_path(db)
            spkrs_list=tmp_dict["spkrs"]
            utts_list=tmp_dict["utts"]
            print(f"db {db} size {len(spkrs_list)} {len(utts_list)}")
            dict_list.append(tmp_dict)

        return dict_list

    def split_dev_test(self,dict_list,test_ratio):

        dict_list_test_dev=list()

        for d in dict_list:
            all_spkrs=d["spkrs"]
            all_utts=d["utts"]
            num_all_spkrs=len(all_spkrs)
            num_test = int(num_all_spkrs*test_ratio)

            d_test_spkrs=all_spkrs[:num_test]
            d_test_utts=dict()
            for spkr in d_test_spkrs:
                print(all_utts[spkr])
                d_test_utts.update({spkr:all_utts[spkr]})
            d_test={"spkrs":d_test_spkrs,"utts":d_test_utts}


            d_dev_spkrs=all_spkrs[num_test:]
            d_dev_utts=dict()
            for spkr in d_dev_spkrs:
                d_dev_utts.update({spkr:all_utts[spkr]})
            d_dev={"spkrs":d_dev_spkrs,"utts":d_dev_utts}

            d_all = {"test_json":d_test,"dev_json":d_dev}

            dict_list_test_dev.append(d_all)

        return dict_list_test_dev

    def merge_index_dicts_to_jsonfile(self,list_of_dicts,test_or_dev):

        multi_db_index_paths_json = dict()
        all_spkrs=list()
        all_utts=dict()

        for d in list_of_dicts:
            if test_or_dev == "test":
                all_spkrs.extend(d["test_json"]["spkrs"])
                all_utts.update(d["test_json"]["utts"])
                json_name=self.PATH_INDEX_JSON_TEST
            elif test_or_dev == "dev":
                all_spkrs.extend(d["dev_json"]["spkrs"])
                all_utts.update(d["dev_json"]["utts"])
                json_name=self.PATH_INDEX_JSON_DEV


        ct_all_utts= sum([len(all_utts[spkr]) for spkr in all_spkrs])
        print(f"from {len(list_of_dicts)} dbs {test_or_dev} spkrs {len(all_spkrs)} utts {ct_all_utts}")
        multi_db_json = {"spkrs":all_spkrs,"utts":all_utts}

        with open(json_name, 'w') as f:
            json.dump(multi_db_json, f)

        return multi_db_json



    def random_batch_select(self):
        batch = []
        spk_dir_paths = random.sample(self.PATH_INDEX_DEV["spkrs"], k=self.BATCH_NUM_SPK)
        for spk_dir_path in spk_dir_paths:
            batch.extend(random.sample(self.PATH_INDEX_DEV["utts"][spk_dir_path], k=self.BATCH_NUM_UTT))
        return batch



    def save_path_index_3_json_spkr_full_path(self, data_dir):
        index_dict = dict()
        index_dict["spkrs"] = []
        spkr_folder_list = self.search_sub_dir_return_full_path(data_dir, fmt="docs", sign=False)
        index_dict["utts"] = dict()

        p = re.compile(r"(?<=(" + self.DATA_DIR_PATH + r"/)).*")
        each_spk_dirs= [p.search(spkr_folder).group() for spkr_folder in spkr_folder_list]
        index_dict["spkrs"].extend(each_spk_dirs)

        for spkr in index_dict["spkrs"]:
            spkr_utts=self.search_sub_dir_return_full_path(os.path.join(self.DATA_DIR_PATH,spkr), isdir=False, fmt=".npy")
            
            if len(spkr_utts) >= 10:
                index_dict["utts"][spkr] = spkr_utts
            else: #if spkr contains less than 10 utts, delete from dict
                print(spkr)
                index_dict["spkrs"].remove(spkr)
                
        return index_dict

    ##############################################################################################################################

    # 2-depth directory
    ### save path index to json
    def save_path_index_2_json(self, data_dir):
        index_dict = dict()
        index_dict["spkrs"] = self.search_sub_dir(data_dir)
        index_dict["utts"] = dict()
        for spk in index_dict["spkrs"]:
            index_dict["utts"][spk] = self.search_sub_dir(os.path.join(data_dir, spk))
        with open('path_index.json', 'w') as f:
            json.dump(index_dict, f)
        return


    # 3-depth directory
    ### save path index to json
    def save_path_index_3_json(self, data_dir):
        index_dict = dict()
        index_dict["spkrs"] = []
        dataset_dict = self.search_sub_dir(data_dir, fmt="docs", sign=False)
        index_dict["utts"] = dict()
        # iterate over spk_dirs of each dataset
        for dataset_dir in dataset_dict:
            each_spk_dirs = self.search_sub_dir(os.path.join(data_dir, dataset_dir))
            index_dict["spkrs"].extend(["{}/{}".format(dataset_dir, each_spk_dir) for each_spk_dir in each_spk_dirs])
        for spk in index_dict["spkrs"]:
            index_dict["utts"][spk] = self.search_sub_dir(os.path.join(data_dir, spk), isdir=False, fmt=".npy")
        with open('path_index.json', 'w') as f:
            json.dump(index_dict, f)
        return


    ##############################################################################################################################

    ### random sequence generate by dir path
    ### directory base path generate
    def search_sub_dir(self, dirname, isdir=True, fmt=None, sign=True):
        list_sub = []
        try:
            filenames = os.listdir(dirname)
            for filename in filenames:
                full_filename = os.path.join(dirname, filename)
                if isdir:
                    if os.path.isdir(full_filename):
                        if fmt == None or ((filename == fmt) if sign else (filename != fmt)):
                            list_sub.append(filename)
                else:
                    if os.path.isfile(full_filename):
                        fname, ext = os.path.splitext(full_filename)
                        if (ext == fmt) if sign else (ext != fmt):
                            list_sub.append(filename)

            return list_sub
        except PermissionError:
            print("PermissionError")
        return None



    def search_sub_dir_return_full_path(self, dirname, isdir=True, fmt=None, sign=True):
        list_sub = []
        try:
            filenames = os.listdir(dirname)

            for filename in filenames:
                full_filename = os.path.join(dirname, filename)
                if isdir:
                    if os.path.isdir(full_filename):
                        if fmt == None or ((filename == fmt) if sign else (filename != fmt)):
                            list_sub.append(os.path.join(dirname,filename))
                else:
                    if os.path.isfile(full_filename):
                        fname, ext = os.path.splitext(full_filename)
                        if (ext == fmt) if sign else (ext != fmt):
                            list_sub.append(os.path.join(dirname,filename))
            #print(f"list_sub {list_sub}")
            return list_sub
        except PermissionError:
            print("PermissionError")
        return None


# if __name__ == "__main__":
#     tail = {  "PATH_INDEX_JSON" : "./path_index.json",
#               "TOTAL_NUM_SPK" : 4500,
#               "TOTAL_NUM_DATASET" : 7,
#               "PART_TOTAL_NUM_SPK" : 1800,
#               "TOTAL_NUM_UTT" : 400,
#               "DATA_DIR_PATH" : "/mnt/data1/youngsunhere/data/namz_kor_spkrid",
#               "BATCH_NUM_SPK" : 64,
#               "BATCH_NUM_UTT" : 10,
#               "PATH_INDEX" : None }
#     prprocess = RandomSelector(tail)
#     prprocess.random_batch_select()
