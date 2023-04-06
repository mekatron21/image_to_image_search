 # This function generate embedings of examples, then calculate similarities, finds neighbours then create datasets of examples with neighbors. 
import tensorflow as tf
import numpy 

IMG_SIZE = 224
#PROJECTED_DIM = 128
#original_dim=1024

class HParams2(object):
  """Hyperparameters used for training."""
  def __init__(self):
    self.num_neighbors = 2 #4 #2
    #self.batch_size = 48
    
    ### eval parameters
    self.eval_steps = None  # All instances in the test set are evaluated.

HPARAMS2 = HParams2()

### Feature extractor ====> Densenet121

def create_feature_extractor_model(IMG_SIZE):
  """Creates a feature extractor model with DenseNet121."""
  inputs = tf.keras.layers.Input((IMG_SIZE, IMG_SIZE, 3))
  
  densenet_model = tf.keras.applications.DenseNet121(weights="imagenet", 
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling="avg", include_top=False
  )
  densenet_model.trainable = False
  x = tf.keras.applications.densenet.preprocess_input(inputs)
  outputs = densenet_model(x, training=False)

  return tf.keras.Model(inputs, outputs, name="densenet_feature_extractor")

feature_extractor = create_feature_extractor_model(IMG_SIZE)
#feature_extractor.summary()


def _int64_feature(value):
  """Returns int64 tf.train.Feature."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value.tolist()))


def _bytes_feature(value):
  """Returns bytes tf.train.Feature."""
  return tf.train.Feature(
    bytes_list=tf.train.BytesList(value=[value.encode('utf-8')]))


def _float_feature(value):
  """Returns float tf.train.Feature."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value.tolist()))


def create_embedding_example(feature_extractor, image,
                             projection_matrix, record_id):
  """Create tf.Example containing the sample's embedding and its ID."""

  image_features = feature_extractor(image[None, ...])
  image_features_numpy = image_features.numpy().squeeze()
  compressed_image_features = image_features_numpy.dot(projection_matrix)

  features = {
    "id": _bytes_feature(str(record_id)),
    "embedding": _float_feature(compressed_image_features)
  }
  return tf.train.Example(features=tf.train.Features(feature=features))


#def generate_random_projection_weights(original_dim=1024,
 #                                      projected_dim=PROJECTED_DIM):
 # """Generates a random projection matrix."""
 # random_projection_matrix = numpy.random.randn(projected_dim, original_dim).T
  #return random_projection_matrix


def create_embeddings(dataset, output_path, original_dim, PROJECTED_DIM, starting_record_id):
  """Creates TFRecords with embeddings of the images."""
  random_projection_matrix = numpy.random.randn(PROJECTED_DIM, original_dim).T
  projection_matrix = random_projection_matrix #generate_random_projection_weights()
  record_id = int(starting_record_id)
  with tf.io.TFRecordWriter(output_path) as writer:
    for image, _ in dataset:
      example = create_embedding_example(feature_extractor,
                                         image,                              # in default no 255
                                         projection_matrix,
                                         record_id)
      record_id = record_id + 1
      writer.write(example.SerializeToString())
  return record_id


# Persist TF.Example features containing embeddings for training data in
# TFRecord format.


def _bytes_feature_image(value):
    """Returns bytes tf.train.Feature."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[value]))
    

def create_example(image, label, record_id):
    """Create tf.Example containing the image, label, and ID."""
    features = {
        "id": _bytes_feature(str(record_id)),
        "image": _bytes_feature_image(image.numpy()),
        "label": _int64_feature(numpy.asarray([label])),
    }
    return tf.train.Example(features=tf.train.Features(feature=features))


def create_records(dataset, record_path, starting_record_id):
  """Generates TFRecords from a tf.data.Dataset object."""
  record_id = int(starting_record_id)
  with tf.io.TFRecordWriter(record_path) as writer:
    for image, label in dataset:
      image = tf.cast(image, tf.uint8)       #########################################image/255 normalizing  # default was tf.cast(image, tf.uint8)
      image = tf.image.encode_jpeg(image, optimize_size=True,
                                    chroma_downsampling=False)
      
      example = create_example(image, label, record_id)
      record_id = record_id + 1
      writer.write(example.SerializeToString())
  return record_id
# Persist TF.Example features (images and labels) for training and validation
# data in TFRecord format.




############## MAKE DATASET WITH GLOBAL NEIGHBORS  #######

NBR_FEATURE_PREFIX = "NL_nbr_"
NBR_WEIGHT_SUFFIX = "_weight"

default_jpeg_value = tf.ones((IMG_SIZE, IMG_SIZE, 3), dtype=tf.uint8)    #dtype=tf.uint8
default_jpeg_value *= 255 #255
default_jpeg_value = tf.image.encode_jpeg(default_jpeg_value, optimize_size=True,
                                         chroma_downsampling=False)

def make_dataset(file_path,BATCH_SIZE, num_train, training=False):
  """Creates a `tf.data.TFRecordDataset`.

  Args:
    file_path: Name of the file in the `.tfrecord` format containing
      `tf.train.Example` objects.
    training: Boolean indicating if we are in training mode.

  Returns:
    An instance of `tf.data.TFRecordDataset` containing the `tf.train.Example`
    objects.
  """

  def parse_example(example_proto):
    """Extracts relevant fields from the `example_proto`.

    Args:
      example_proto: An instance of `tf.train.Example`.

    Returns:
      A pair whose first value is a dictionary containing relevant features
      and whose second value contains the ground truth labels.
    """
    feature_spec = {
      'image': tf.io.FixedLenFeature([], tf.string, 
                                      default_value=default_jpeg_value),
      'label': tf.io.FixedLenFeature((), tf.int64, default_value=-1),
    }
    # We also extract corresponding neighbor features in a similar manner to
    # the features above during training.
    if training:
      for i in range(HPARAMS2.num_neighbors):
        nbr_feature_key = '{}{}_{}'.format(NBR_FEATURE_PREFIX, i, 'image')
        nbr_weight_key = '{}{}{}'.format(NBR_FEATURE_PREFIX, i,
                                         NBR_WEIGHT_SUFFIX)
        feature_spec[nbr_feature_key] = tf.io.FixedLenFeature([], tf.string,
                                            default_value=default_jpeg_value)

        # We assign a default value of 0.0 for the neighbor weight so that
        # graph regularization is done on samples based on their exact number
        # of neighbors. In other words, non-existent neighbors are discounted.
        feature_spec[nbr_weight_key] = tf.io.FixedLenFeature(
            [1], tf.float32, default_value=tf.constant([0.0]))
    
    features = tf.io.parse_single_example(example_proto, feature_spec)
    labels = features.pop('label')
    
    # We need to convert the byte-strings back to images.
    features['image'] = tf.image.decode_jpeg(features['image'], channels=3)    /255     #normalizing images again [0-1]
    if training:
      for i in range(HPARAMS2.num_neighbors):
        nbr_feature_key = '{}{}_{}'.format(NBR_FEATURE_PREFIX, i, 'image')
        features[nbr_feature_key] = tf.image.decode_jpeg(features[nbr_feature_key],
                                                         channels=3)/255

    return features, labels

  dataset = tf.data.TFRecordDataset([file_path])
  if training:
    dataset = dataset.shuffle(num_train * 2)
  dataset = dataset.map(parse_example)
  dataset = dataset.batch(BATCH_SIZE)

  return dataset



