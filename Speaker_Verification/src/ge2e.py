#-*- coding: utf-8 -*-
import tensorflow as tf
import os
import json
import argparse
import numpy as np
import librosa
import time
from multiprocessing import Pool
import glob
import re
import datetime
from tqdm import tqdm

# 레이어, 데이터셋, 콜백 관련 모듈
from layers.centroid_matrix import Centroid_matrix
from layers.centroid_matrix_basic import Centroid_matrix_basic
from layers.similarity import Similarity_matrix
from layers.ge2e_loss import Ge2e_loss
from layers.callbacks import InferenceCallback
from layers.callback_test import TESTCallback
from layers.preprocess import Preprocess
from layers.batch_preprocess import Batch_preprocess
from utils.batch_creator import BatchCreator
from utils.create_test_data import create_test_data
from utils.optimizer_ge2e import Ge2eOptimizer
from utils.wav_to_mel import Wav2Mel
from utils.infer_process import InferPrep
from utils.utils import lr_scheduler
from utils.eer import *
#######
# Model
#######

# Define custom loss
def custom_loss(y_true, y_pred):
    return y_pred

def dummy_loss(y_true, y_pred):
    return 0 * y_pred


class Speaker_verification(object):
    '''
    Note:
        create, train, inference & export the model from config
    attributes:
            __init__: make layers of model from config
            inputs: return model inputs
            outputs: return model outputs
            _run: train the model
            _infer: inference from the model
            _export: export the model
            _build: build the model from the layers
            _infer_build: build the inference model from the layers
            _callbacks: make callbacks
            _train_generator: train generator func
            _prepare_batch: create batch from mels
    '''

    def __init__(self, config,mode):
        # common parameters
        self.mode = mode
        self._num_mel = config["num_mel"]
        self._dense_unit = config["feature_dimension"]
        self._num_lstm = config["num_lstm"]
        self._lstm_unit = config["num_lstm_unit"]

        # multi-lstm layers
        self._lstm_layers = []
        self._dense_layers = []
        for lstm_idx in range(self._num_lstm):
            return_sequences = True
            # 마지막이면
            if lstm_idx == self._num_lstm - 1:
                return_sequences = False
            lstm_layer = tf.keras.layers.CuDNNLSTM(units=(self._lstm_unit), return_sequences=return_sequences, name="lstm_{}".format(lstm_idx))
            dense_layer = tf.keras.layers.Dense(units=(self._dense_unit), name="dense_{}".format(lstm_idx))

            self._lstm_layers.append(lstm_layer)
            self._dense_layers.append(dense_layer)

        self._embedding_norm_layer = tf.keras.layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=-1), name='embeding_output')

        if self.mode == "train":

            self._batch_creator = BatchCreator(config)

            # prepare eer / tsne data to be used during training (3 batches = 2190 utts)
            self._test_data_path = config["test_data_path"]
            self._num_spkr_per_test_batch = config["num_spkr_per_test_batch"]
            self._num_utt_per_test_spkr = config["num_utt_per_test_spkr"]
            self._num_batches_test = config["num_batches_test"]

            # check if test data dir exists
            if not os.path.isdir(self._test_data_path):
                os.makedirs(self._test_data_path)

            # get test batches ready
            self.PATH_INDEX_JSON_TEST = config["path_index_json"]+"_test.json"
            batch_list, batch_spkr_utt_lab_list = create_test_data(self.PATH_INDEX_JSON_TEST,self._test_data_path,self._batch_creator,
                                                                self._num_spkr_per_test_batch,
                                                                self._num_utt_per_test_spkr,self._num_batches_test)
            self._test_data = [batch_list, batch_spkr_utt_lab_list]

            # PATH
            self._data_dir = config["data_dir_path"]
            self._tensorboard_log_dir = config["tensorboard_log"]
            self._checkpoint_path = config["checkpoint_path"]
            self._checkpoint_dir = os.path.dirname(self._checkpoint_path)
            self._checkpoint_latest = tf.train.latest_checkpoint(self._checkpoint_dir)

            # EMB PLOTTING 2d PARAMS
            self._test_tsne_eer_period=config["test_every_n_epoch"]
            self._plot_spkr_num=config["plot_spkr_num"]

            # parameters
            self._num_speakers = config["num_speakers"]
            self._num_utterance = config["num_utterance"]
            self._batch_size = self._num_speakers * self._num_utterance
            self._seq_length = None # config["sequence_length"]
            self._shuffle_buffer = config["shuffle_buffer"]
            self._exclusive_centroid = bool(config["exclusive_centroid"])  # 0 or 1  ==> False or True
            self._loss_type = config["loss_type"]
            #self._num_inference = config["num_inference"]
            ## fit_generator params
            self._epochs = config["num_epochs"],
            self._steps_per_epoch = config["steps_per_epoch"],
            self._validation_steps = config["validation_step"],#,
            self._max_queue_size = config["max_queue_size"],
            self._use_multiprocessing = config["multiprocessing"],
            self._workers = config["workers"]

            # in-out
            self._inputs = []
            self._outputs = []

            # input layers
            self._input_layer = tf.keras.layers.Input(shape=(self._batch_size, self._seq_length, self._num_mel), name="mel_input")
            self._squeeze_layer = tf.keras.layers.Lambda(lambda x: tf.keras.backend.squeeze(x, axis=0), name='64_x_10_batch_input')

            # loss layers
            if self._exclusive_centroid:
                self._centroid_layer = Centroid_matrix(self._num_speakers, self._num_utterance, name='centroids')
            else:
                self._centroid_layer = Centroid_matrix_basic(self._num_speakers, self._num_utterance, name='centroids')
            self._similarity_layer = Similarity_matrix(self._num_speakers, self._num_utterance, is_exclusive=self._exclusive_centroid, name='similarity')
            self._ge2e_loss_layer = Ge2e_loss(self._num_speakers, self._num_utterance, self._loss_type, name='ge2e_loss')

            # nodes
            self._X = None
            self._centroid = None


        else: # INFER & EXPORT
            # preprocess parameters
            self._fft_window = config["fft_window_size"]
            self._fft_hop = config["fft_hop_size"]
            self._num_fft = config["num_fft"]
            self._sr = config["sample_rate"]
            self._window = config["ge2e_window_size"]
            self._overlap = config["ge2e_hop_size"]
            self._win_unit = config["ge2e_window_unit"]

            # model parameters
            self._checkpoint_path = config["checkpoint_path"]
            self._checkpoint_dir = os.path.dirname(self._checkpoint_path)
            self._checkpoint_latest = tf.train.latest_checkpoint(self._checkpoint_dir)
            self._export_path = config["export_path"]
            self._seq_length = None

            # in-out
            self._inputs = []
            self._outputs = []

            # input layers
            if self.mode == "infer":
                #reshape the flat input list to ndarray and re-export the model!!!
                self._input_layer = tf.keras.layers.Input(shape=(None,self._window,self._num_mel), name="mel_input")
                self._reshape_layer = tf.keras.layers.Lambda(lambda x: tf.keras.backend.squeeze(x,axis=0), name='N_x_10_batch_input')
            elif self.mode == "export":
                self._input_layer = tf.keras.layers.Input(shape=(1,), name="flattend_mel_input")
                self._reshape_layer = tf.keras.layers.Reshape((self._window,self._num_mel), name="ndarray_mel_input")

            # "The final utterance-wise d-vector is generated by L2 normalizing the window-wise d-vectors,
            # then taking the element-wise averge (section 3.2)"
            self._mean_layer = tf.keras.layers.Lambda(lambda x: tf.keras.backend.l2_normalize(tf.keras.backend.mean(x, axis=0, keepdims=True), axis=-1), name='mean_embedding')

            # nodes
            self._X = None
            self._exclusive_centroid = bool(config["exclusive_centroid"])  # 0 or 1  ==> False or True

            self._num_speakers = config["num_speakers"]
            self._num_utterance = config["num_utterance"]
            self._loss_type = config["loss_type"]

            # loss layers
            if self._exclusive_centroid:
                self._centroid_layer = Centroid_matrix(self._num_speakers, self._num_utterance, name='centroids')
            else:
                self._centroid_layer = Centroid_matrix_basic(self._num_speakers, self._num_utterance, name='centroids')
            self._similarity_layer = Similarity_matrix(self._num_speakers, self._num_utterance, is_exclusive=self._exclusive_centroid, name='similarity')
            self._ge2e_loss_layer = Ge2e_loss(self._num_speakers, self._num_utterance, self._loss_type, name='ge2e_loss')

            # nodes
            self._centroid = None

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    def _run(self):

        # TF GRAPH 빌드
        model = self._build()
        # TF 콜백 만들기
        callbacks = self._callbacks() # [multiple batches in list, labels ]
        # TF 모델 컴파일
        model.compile(optimizer=Ge2eOptimizer(), loss=[custom_loss, dummy_loss])
        # TF KERAS 모델 요약 출력
        model.summary()
        # TF 모델 최신 체크포인트 확인
        last_epoch = 0
        if self._checkpoint_latest is None:
            print('There is no checkpoint.')
        else:
            model.load_weights(self._checkpoint_latest)
            print('Checkpoint \'{}\' is loaded.'.format(self._checkpoint_latest))

        if self._checkpoint_latest:
            num_epoch = re.findall(r'cp-(\d+)\.ckpt$', self._checkpoint_latest)[0]
            last_epoch = int(num_epoch.lstrip('0'))
        else: # if not latest check point is found
            last_epoch = 0


        model.fit_generator(generator=self._train_generator(),
                            epochs=self._epochs[0],
                            steps_per_epoch=self._steps_per_epoch[0],
                            callbacks=callbacks,
                            validation_data=self._train_generator(),
                            validation_steps=self._validation_steps[0],#,
                            initial_epoch=last_epoch,
                            max_queue_size=self._max_queue_size[0],
                            use_multiprocessing=self._use_multiprocessing[0],
                            workers=self._workers
                        )
        return

    def _infer_mels(self,config,emb_savedir,ckpt_to_load,mellist):
        model = self._infer_build()

        if ckpt_to_load == 'latest':
            ckpt_to_load = self._checkpoint_latest

        num_epoch = re.findall(r'cp-(\d+)\.ckpt$',ckpt_to_load)[0]
        emb_savedir_byckpt = os.path.join(emb_savedir,f'ckpt_{num_epoch}')
        if not os.path.exists(emb_savedir_byckpt):
            os.makedirs(emb_savedir_byckpt)

        model.load_weights(ckpt_to_load)
        new_model = tf.keras.Model(model.inputs, model.layers[-5].output)
        new_model.summary()

        print('Checkpoint \'{}\' is loaded.'.format(ckpt_to_load))

        #mel_save_path=emb_savedir+"_preprocessed"
        inferprep=InferPrep(config)

        for melpath in tqdm(mellist):
            mel = np.load(melpath)
            # mel -> mel batch
            mel_batch=inferprep.window_and_stack_batch(mel,inferprep.window,inferprep.fft_window,inferprep.fft_hop,inferprep.overlap)
            
            # mel batch -> emb 
            if mel_batch != []:
                mel_batch=np.expand_dims(mel_batch,0)
                result = new_model.predict(mel_batch, batch_size=1)
                filename=re.sub('.wav','',os.path.basename(melpath))
                emb_savename=f"emb_{filename}.npy"
                np.save(os.path.join(emb_savedir_byckpt,emb_savename),result)
            else:
                print("{} : couldn't get valid mel batch".format(melpath))

        timenow = datetime.datetime.now().strftime('%H:%M:%S')
        print(f"[{timenow}] finished inference! check: {emb_savedir_byckpt}")

        eer = get_eer_from_checkpoint(emb_savedir_byckpt)
        timenow = datetime.datetime.now().strftime('%H:%M:%S')
        print(f"[{timenow}] EER at {num_epoch} {eer} ({emb_savedir_byckpt})")
                
        return [timenow, num_epoch, eer]

    def _process_and_infer(self,config,emb_savedir,ckpt_to_load,wavlist):
        model = self._infer_build()

        if ckpt_to_load == 'latest':
            ckpt_to_load = self._checkpoint_latest

        num_epoch = re.findall(r'cp-(\d+)\.ckpt$',ckpt_to_load)[0]
        emb_savedir_byckpt = os.path.join(emb_savedir,f'ckpt_{num_epoch}')
        if not os.path.exists(emb_savedir_byckpt):
            os.makedirs(emb_savedir_byckpt)

        model.load_weights(ckpt_to_load)
        new_model = tf.keras.Model(model.inputs, model.layers[-5].output)
        new_model.summary()

        print('Checkpoint \'{}\' is loaded.'.format(ckpt_to_load))

        #mel_save_path=emb_savedir+"_preprocessed"
        inferprep=InferPrep(config)

        for wavpath in tqdm(wavlist):
            # wav -> mel batch
            mel_batch=inferprep.run_preprocess_infer(wavpath)
            
            # mel batch -> emb 
            if mel_batch != []:
                mel_batch=np.expand_dims(mel_batch,0)
                result = new_model.predict(mel_batch, batch_size=1)
                filename=re.sub('.wav','',os.path.basename(wavpath))
                emb_savename=f"emb_{filename}.npy"
                np.save(os.path.join(emb_savedir_byckpt,emb_savename),result)
            else:
                print("{} : couldn't get valid mel batch".format(wavpath))

        timenow = datetime.datetime.now().strftime('%H:%M:%S')
        print(f"[{timenow}] finished inference! check: {emb_savedir_byckpt}")

        eer = get_eer_from_checkpoint(emb_savedir_byckpt)
        timenow = datetime.datetime.now().strftime('%H:%M:%S')
        print(f"[{timenow}] EER at {num_epoch} {eer}% ({emb_savedir_byckpt})")
                
        return

    def _export(self,ckpt_to_load):
        model = self._infer_build()

        if ckpt_to_load == 'latest':
            ckpt_to_load = self._checkpoint_latest
        print(ckpt_to_load)
        model.load_weights(ckpt_to_load)
        print('Checkpoint \'{}\' is loaded.'.format(ckpt_to_load))
        new_model = tf.keras.Model(model.inputs, model.layers[-5].output)
        new_model.summary()
        #set up export path
        try:
            model_path = '/'.join(self._export_path.split('/')[:-1])
            model_version = max([int(o) for o in os.listdir(model_path.format(model="ge2e"))]) + 1
        except:
            model_version = 1

        export_path = self._export_path.format(model="ge2e", version=model_version)

        # Fetch the Keras session and save the model
        # The signature definition is defined by the input and output tensors
        # And stored with the default serving key
        with tf.keras.backend.get_session() as sess:
            tf.compat.v1.saved_model.simple_save(
                sess,
                export_path,
                inputs={'flattend_mel_input': new_model.input},
                outputs={t.name:t for t in new_model.outputs})

    def _build(self, is_trainig=False):
        # inputs
        self._X = self._input_layer
        self._inputs.append(self._X)
        self._X = self._squeeze_layer(self._X)

        # multi lstm + dense
        for lstm_idx in range(self._num_lstm):
            self._X = self._lstm_layers[lstm_idx](self._X)
            self._X = self._dense_layers[lstm_idx](self._X)
        self._X = self._embedding_norm_layer(self._X)       # [tot_utt, embed_dim]

        # loss layers
        self._centroid = self._centroid_layer(self._X)       # [num_spkr, embed_dim]
        self._similarity_matrix = self._similarity_layer((self._X, self._centroid)) # [tot_utt, num_spkr]
        self._loss = self._ge2e_loss_layer(self._similarity_matrix)

        # outputs
        self._outputs.append(self._loss)
        self._outputs.append(self._X)
        model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs)
        return model

    def _infer_build(self):
        # inputs
        self._X = self._input_layer
        print(f"before reshape layer {self._X}")
        self._X = self._reshape_layer(self._X)
        print(f"after reshape layer {self._X}")

        # multi lstm + dense
        for lstm_idx in range(self._num_lstm):
            self._X = self._lstm_layers[lstm_idx](self._X)
            self._X = self._dense_layers[lstm_idx](self._X)
        self._X = self._embedding_norm_layer(self._X)       # [tot_utt, embed_dim]
        print(f"before mean layer {self._X}")
        self._X = self._mean_layer(self._X)

        # loss layers
        print(f"after mean layer {self._X}")
        tile_layer = tf.keras.layers.Lambda(lambda x : tf.tile(x, [640,1]), name='mean_emb_tile')
        self._X =  tile_layer(self._X)
        self._centroid = self._centroid_layer(self._X)       # [num_spkr, embed_dim]

        self._similarity_matrix = self._similarity_layer((self._X, self._centroid)) # [tot_utt, num_spkr]
        self._loss = self._ge2e_loss_layer(self._similarity_matrix)

        model = tf.keras.Model(inputs=[self._input_layer], outputs=[self._X,self._loss])
        return model

    def _callbacks(self):
        # learning rate scheduler
        lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

        # ckpt save callback
        cp_save_callback = tf.keras.callbacks.ModelCheckpoint(self._checkpoint_path, save_weights_only=True, verbose=1, period=1)

        # test (eer + cluster) callback
        test_callback = TESTCallback(self._test_data,self._test_data_path,self._plot_spkr_num,self._test_tsne_eer_period)

        # # 텐서보드 출력 콜백 만들기
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(
        #                             log_dir="logs/{}".format(self._tensorboard_log_dir),
        #                   r          histogram_freq=1,
        #                             write_images=True)

        callbacks = [
            #lr_callback,
            cp_save_callback,
            test_callback
            # write utterance embedding in ckpt & Tb events.
            #inference_callback,
            # Write TensorBoard logs to `./logs/ge2e` directory
            #tensorboard_callback
           ]
        return callbacks

    ### train_generator 추가
    def _train_generator(self):
        while 1:

            mel_one_batch = self._prepare_batch()
            mel_one_batch = np.expand_dims(mel_one_batch, axis=0)

            dummy_loss = np.zeros([1, 640, 1], dtype="f")
            dummy_output = np.zeros([1, 640, 256], dtype="f")

            yield ({'mel_input': mel_one_batch}, {'ge2e_loss': dummy_loss, 'embeding_output': dummy_output})

    def _prepare_batch(self):

        start1 = time.time()
        batch_mel_paths_list = self._batch_creator.get_batch_mel_path()

        timerandom_create = datetime.datetime.now().strftime("%H:%M:%S")
        filelist=sorted([os.path.basename(i) for i in batch_mel_paths_list])
        spkr_list=set(["_".join(i.split('_')[:-1]) for i in filelist])
       
        start2 = time.time()
        results = []
        for mel_path in batch_mel_paths_list:
            mel, len_mel, spk_name = self._batch_creator.load_single_mel_npy(mel_path)
            results.append([mel, len_mel, spk_name])

        timenow = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[ current time:{timenow} ] reading 640 mel npy files took: {time.time()-start2:0.5f} secs")


        datadict = dict()
        datadict['mel'] = [i[0] for i in results]
        datadict['mel_len'] = [i[1] for i in results]
        datadict['spk_list'] = [i[2] for i in results]


        start3 = time.time()
        one_batch = self._batch_creator.create_batch(datadict)
        print(one_batch.shape)
        return one_batch
