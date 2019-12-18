import os
import tensorflow as tf
import pandas as pd
from tensorflow.python.keras import backend as K
from tensorflow.python.eager import context
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary as tf_summary
from tensorflow.python.ops import summary_ops_v2
from sklearn.preprocessing import StandardScaler
from tensorflow.contrib.tensorboard.plugins import projector
from sklearn.decomposition import PCA
import numpy as np


################################################################
def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    from PIL import Image
    height, width, channel = tensor.shape
    image = Image.fromarray(tensor.reshape((16,16)))
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                         width=width,
                         colorspace=channel,
                         encoded_image_string=image_string)


class InferenceCallback(tf.keras.callbacks.Callback):
    """
    callback to observe the output of the network
    """

    def __init__(self, xy, num_batch):
        super(InferenceCallback, self).__init__()
        self.out_log = []
        self.xy = xy
        self._num_batch = num_batch

    def on_epoch_end(self, epoch, logs={}):
        ## Inference by dataset(self.xy or x_train)
        self.out_log.append(self.model.predict(self.xy, steps=self._num_batch))

        ## tensorboard
        embedding_writer = tf.summary.FileWriter('./logs/ge2e')
        df = pd.DataFrame(data=self.out_log[-1][-1])

        ## Load the metadata file.
        ## Metadata consists your labels. This is optional.
        ## Metadata helps us visualize(color) different clusters that form t-SNE
        metadata = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "config", 'metadata_{}.tsv'.format(self._num_batch))

        ## Generating PCA and
        pca = PCA(n_components=50, random_state=123, svd_solver='auto')
        df_pca = pd.DataFrame(pca.fit_transform(df))
        df_pca = df_pca.values

        ## TF Variable from data
        tf_data = tf.Variable(df_pca)
        config = projector.ProjectorConfig()
        saver = tf.train.Saver([tf_data])
        sess = K.get_session()
        sess.run(tf_data.initializer)

        ## 이전 임베딩 체크포인트 로딩 tf_data_{}.cpkt
        latest_epoch = epoch
        checkpoint_latest = tf.train.latest_checkpoint('./logs/ge2e')
        if checkpoint_latest is not None:
        	# 가장 최신의 에포크
            latest_epoch = int(os.path.splitext(os.path.basename(checkpoint_latest))[0].split('_')[-1])

        ## TB PROJECTOR 용 임베딩 저장
        saver.save(sess, os.path.join('./logs/ge2e', 'tf_data_{}.ckpt').format(latest_epoch + 1))

        ## One can add multiple embeddings.
        embedding = config.embeddings.add()
        embedding.tensor_name = tf_data.name

        ## 임베딩 텐서와 메타파일(레이블) 링크
        embedding.metadata_path = metadata

        # TB PROJECTOOR 용 config 저장. 텐서보드 시작 시 읽어들임.
        projector.visualize_embeddings(embedding_writer, config)


        ## INFERENCE 가장 마지막 임베딩 벡터를 16 X 16 grayscale 이미지로 변환 저장
        image_t = np.reshape(self.out_log[-1][-1], [-1, 16, 16, 1])
        image = 255 * image_t[-1, :, :, :]
        image = make_image(image.astype(np.uint8))
        summary = tf.Summary(value=[tf.Summary.Value(tag="embedding", image=image)])
        embedding_writer.add_summary(summary, epoch)
        print("EMBEDDING 16 X 16 IMAGE will be saved.")

        # WRITER 닫기
        embedding_writer.close()

        # CSV 저장
        # np.savetxt("{}.csv".format(epoch), self.out_log[-1][-1], delimiter=",", fmt='%s')