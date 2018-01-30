# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from trainer_mat import model
from trainer_mat.model import DataSequence, create_data, rank_decode
import keras
import logging
from logging import StreamHandler
from sys import stdout
from keras.models import load_model
import multiprocessing
import subprocess
"""This code implements a Feed forward neural network using Keras API."""

import argparse
import glob
import os


from tensorflow.python.lib.io import file_io
import numpy as np

# CHUNK_SIZE specifies the number of lines
# to read in case the file is very large
FILE_PATH = 'checkpoint.{epoch:02d}.hdf5'
CENSUS_MODEL = 'ordinal_face.hdf5'


class CustomTensorBoard(keras.callbacks.TensorBoard):
    def __init__(self, data_sequence, *args, **kwargs):
        # One extra argument to indicate whether or not to use the PR curve
        # summary.
        self.pr_curve = kwargs.pop('pr_curve', True)
        super(CustomTensorBoard, self).__init__(*args, **kwargs)
        self.data_sequence = data_sequence

        import tensorflow as tf
        global tf

    def set_model(self, model):
        super(CustomTensorBoard, self).set_model(model)

        # if self.pr_curve:
        # Get the prediction and label tensor placeholders.
#             predictions = self.model.output
#             labels = tf.cast(self.model.targets[0], tf.bool)
#             # Create the PR summary OP.
#             self.pr_summary = pr_summary.op(tag='pr_curve',
#                                             predictions=predictions,
#                                             labels=labels,
# display_name='Precision-Recall Curve')

    def on_epoch_end(self, epoch, logs=None):
        super(CustomTensorBoard, self).on_epoch_end(epoch, logs)
        #print('len=%s' % len(self.validation_data[0]))
        #x_val = self.validation_data[0]
        #y_val = self.validation_data[1]
        # print(self.validation_data)
#         if self.pr_curve and self.validation_data:
#             # Get the tensors again.
#             tensors = self.model._feed_targets + self.model._feed_outputs
#             # Predict the output.
        import numpy as np
        mae_arr = []
        predictions = np.array([])
        y_tot = np.array([])
        for i in range(self.data_sequence.length):
            x_val, y_val = self.data_sequence.__getitem__(i)
            predictions_small = self.model.predict(x_val)
            predictions_small = rank_decode(predictions_small)
            y_val = rank_decode(y_val)
            print('pred=%s, truth=%s' % (predictions_small,y_val))
            y_tot = np.append(y_tot, y_val)
            predictions = np.append(predictions, predictions_small)
        mae = np.mean(np.abs(predictions - y_tot))

#             # Build the dictionary mapping the tensor to the data.
        #val_data = [self.validation_data[-2], predictions]
        #print('preds=%s' % len(predictions))
#             feed_dict = dict(zip(tensors, val_data))
#             # Run and add summary.
#             result = self.sess.run([self.pr_summary], feed_dict=feed_dict)
        # print(result[0])
        #mae = np.mean(np.abs(predictions - y_val)) / len(predictions)
        #print('mae=%s' % mae)
        #mae = np.array()
        print('mae=%s' % mae)
        summary = tf.Summary(value=[
            tf.Summary.Value(tag="mae", simple_value=mae),
        ])
        self.writer.add_summary(summary, epoch)
        self.writer.flush()

class ContinuousEval(keras.callbacks.Callback):
    """Continuous eval callback to evaluate the checkpoint once
       every so many epochs.
    """

    def __init__(self,
                 eval_frequency,
                 x, y,
                 learning_rate,
                 job_dir):
        self.eval_frequency = eval_frequency
        self.learning_rate = learning_rate
        self.job_dir = job_dir
        self.x = x
        self.y = y

    def on_epoch_begin(self, epoch, logs={}):
        if epoch > 0 and epoch % self.eval_frequency == 0:

            # Unhappy hack to work around h5py not being able to write to GCS.
            # Force snapshots and saves to local filesystem, then copy them
            # over to GCS.
            model_path_glob = 'checkpoint.*'
            if not self.job_dir.startswith("gs://"):
                model_path_glob = os.path.join(self.job_dir, model_path_glob)
            checkpoints = glob.glob(model_path_glob)
            if len(checkpoints) > 0:
                checkpoints.sort()
                census_model = load_model(checkpoints[-1])
                census_model = model.compile_model(
                    census_model, self.learning_rate)
                data_sequence = DataSequence(
                    self.x, self.y, batch_size=32)
                loss_acc = census_model.evaluate_generator(
                    data_sequence,
                    steps=data_sequence.length)
                loss = loss_acc[0:80]
                acc = loss_acc[80:]
                for ac, lo in zip(acc,loss):
                    print('\nEvaluation epoch[{}] metrics[{:.2f}, {:.2f}] {}'.format(
                        epoch, lo, ac, census_model.metrics_names))
                if self.job_dir.startswith("gs://"):
                    copy_file_to_gcs(self.job_dir, checkpoints[-1])
            else:
                print(
                    '\nEvaluation epoch[{}] (no checkpoints found)'.format(epoch))


def dispatch(train_files,
             job_dir,
             train_batch_size,
             learning_rate,
             model_file,
             num_epochs,
             checkpoint_epochs,
             lam,
             dropout
             ):

    x_train, y_train, x_test, y_test, input_shape = create_data(train_files)
    logger = logging.getLogger()
    sh = StreamHandler(stdout)
    logger.addHandler(sh)
    logger.setLevel(logging.INFO)
    logger.info('learning_rate=%s' % learning_rate)
    if model_file is not None:
        cmd = 'gsutil cp %s /tmp' % model_file
        subprocess.check_call(cmd.split())
        census_model = load_model('/tmp/%s' % model_file.split('/')[-1])
    else:
        census_model = model.model_fn(learning_rate, lam, dropout)

    try:
        os.makedirs(job_dir)
    except:
        pass

    # Unhappy hack to work around h5py not being able to write to GCS.
    # Force snapshots and saves to local filesystem, then copy them over to
    # GCS.
    checkpoint_path = FILE_PATH
    if not job_dir.startswith("gs://"):
        checkpoint_path = os.path.join(job_dir, checkpoint_path)
        verbose=1
    else:
        verbose=2
#
#     meta_data = get_meta(train_files)
#     indexes = [i for i in range(len(meta_data))]
#     random.shuffle(indexes)
#     meta_data = meta_data.loc[indexes].reset_index(drop=True)

    # Model checkpoint callback
    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        verbose=1,
        period=checkpoint_epochs,
        mode='max')

    # Continuous eval callback
#     evaluation = ContinuousEval(eval_frequency,
#                                 x_test, y_test,
#                                 learning_rate,
#                                 job_dir,
#                                 )

    val_data_sequence = DataSequence(
         x_test, y_test,
         batch_size=train_batch_size
    )
    # Tensorboard logs callback
    tblog = CustomTensorBoard(val_data_sequence,
        log_dir=os.path.join(job_dir, 'logs'),
        histogram_freq=0,
        write_graph=True,
        embeddings_freq=0)

    callbacks = [checkpoint, tblog]

    train_data_sequence = DataSequence(
         x_train, y_train,
         batch_size=train_batch_size
    )
#     test_data_sequence = DataSequence(
#         x_test, y_test,
#         batch_size=train_batch_size)
    
    census_model.fit_generator(#x_train, y_train,
        #model.generator_input(train_files, chunk_size=CHUNK_SIZE),
        train_data_sequence,
        validation_data=val_data_sequence,
        validation_steps=val_data_sequence.length,
        verbose=verbose,
        steps_per_epoch=train_data_sequence.length,
        epochs=num_epochs, workers=multiprocessing.cpu_count(),
        use_multiprocessing=True,
        callbacks=callbacks)

    #plot_history(history)
    # Unhappy hack to work around h5py not being able to write to GCS.
    # Force snapshots and saves to local filesystem, then copy them over to
    # GCS.
    if job_dir.startswith("gs://"):
        census_model.save(CENSUS_MODEL)
        copy_file_to_gcs(job_dir, CENSUS_MODEL)
    else:
        census_model.save(os.path.join(job_dir, CENSUS_MODEL))

    # Convert the Keras model to TensorFlow SavedModel
    model.to_savedmodel(census_model, os.path.join(job_dir, 'export'))

# h5py workaround: copy local models over to GCS if the job_dir is GCS.

# def plot_history(history):
#     # print(history.history.keys())
# 
#     # 精度の履歴をプロット
#     plt.plot(history.history['acc'])
#     plt.plot(history.history['val_acc'])
#     plt.title('model accuracy')
#     plt.xlabel('epoch')
#     plt.ylabel('accuracy')
#     plt.legend(['acc', 'val_acc'], loc='lower right')
#     plt.show()
# 
#     # 損失の履歴をプロット
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('model loss')
#     plt.xlabel('epoch')
#     plt.ylabel('loss')
#     plt.legend(['loss', 'val_loss'], loc='lower right')
#     plt.show()
# 
# # 学習履歴をプロット
#     plot_history(history)

def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='r') as input_f:
        with file_io.FileIO(os.path.join(job_dir, file_path), mode='w+') as output_f:
            output_f.write(input_f.read())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-files',
                        required=True,
                        type=str,
                        help='Training files local or GCS', nargs='+')
    parser.add_argument('--job-dir',
                        required=True,
                        type=str,
                        help='GCS or local dir to write checkpoints and export model')
    parser.add_argument('--train-batch-size',
                        type=int,
                        default=32,
                        help='Batch size for training steps')
    parser.add_argument('--learning-rate',
                        type=float,
                        default=0.003,
                        help='Learning rate for Adam')
    parser.add_argument('--model-file',
                        default=None,
                        help='model file')
    parser.add_argument('--lam',
                        type=float,
                        default=0.0,
                        help='l2 regularizaion lambda')
    parser.add_argument('--dropout',
                        type=float,
                        default=1.0,
                        help='dropout')
    parser.add_argument('--num-epochs',
                        type=int,
                        default=20,
                        help='Maximum number of epochs on which to train')
    parser.add_argument('--checkpoint-epochs',
                        type=int,
                        default=1,
                        help='Checkpoint per n training epochs')
    parse_args, unknown = parser.parse_known_args()

    dispatch(**parse_args.__dict__)
