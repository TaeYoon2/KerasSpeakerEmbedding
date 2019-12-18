import tensorflow as tf

def _parse_function(proto):
    """
    Note:
        Parse the tf.dataset
    Args:
        proto: dataset to be parsed

    Returns:

    """

    # 파싱할 TF RECORD 데이터의 키이름과 형식
    keys_to_features = {
              "mel": tf.io.FixedLenFeature([], tf.string),
              "shape": tf.io.FixedLenFeature([], tf.string)}

    # Load one example
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)

    # 읽어드린 TF RECORD를 원래의 데이터 타입으로 디코딩
    parsed_features['mel'] = tf.decode_raw(
        parsed_features['mel'], tf.float32)
    parsed_features['shape'] = tf.decode_raw(
        parsed_features['shape'], tf.uint8)

    # 모델에 들어가기 알맞게 reshape
    parsed_features['mel'] = tf.reshape(parsed_features['mel'], [64, 10, -1, 40])
    parsed_features['mel'] = tf.reshape(parsed_features['mel'], [640, -1, 40])


    return parsed_features['mel']


def create_dataset(filepath, dataset_config):
    """
    Note:
        Create dataset of ge2e model from TFRecord
    Args:
        filepath: the path of TFRecord file
        dataset_config: the configurations of batching

    Returns:
        dataset
    """

    SHUFFLE_BUFFER, BATCH_SIZE, NUM_MEL_BINS = dataset_config
    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filepath)

    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(_parse_function, num_parallel_calls=8)

    # This dataset will go on forever
    dataset = dataset.repeat()
    # This is for tf.keras.Model.fit(); Because it needs dummy target to be compared with model's outputs.
    output_dataset = tf.data.Dataset.from_tensor_slices((tf.zeros([640, 1], dtype=tf.float32), tf.zeros([640, 256], dtype=tf.float32)))

    # Set the number of datapoints you want to load and shuffle
    dataset = dataset.shuffle(SHUFFLE_BUFFER)

    # Set the batchsize
    dataset = dataset.batch(1)
    output_dataset = output_dataset.repeat().batch(1)
    dataset = tf.data.Dataset.zip((dataset, output_dataset))

    return dataset