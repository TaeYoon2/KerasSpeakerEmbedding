def create_serving_pickle(config, wav_path, emb_save_path):
    # give args.wavdir so that [ args.wavdir+'/*wav' ] can capture all the wavfiles you intend to infer
    mel_save_path = emb_save_path + '_preprocessed'
            if not os.path.exists(mel_save_path):
                        os.makedirs(mel_save_path)

                            wavlist = glob.glob(wav_path+'/*/*/*wav')
                                emb_done_list = [os.path.basename(i) for i in glob.glob(emb_save_path+'/*npy')]
                                    # filter out wavfiles that are already preprocessed
                                        wavlist_to_preprocess = [ i for i in wavlist if re.sub('wav', 'npy', os.path.basename(i)) not in emb_done_list]
                                            print(f"todo wavlist: {len(wavlist_to_preprocess)} of {len(wavlist)}")

                                                # config update for Inference preprocessing
                                                    config_update = {
