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
from trainer_mat.model import DataSequence, get_meta, create_data
import random
import keras
import logging
from logging import StreamHandler
from sys import stdout
"""This code implements a Feed forward neural network using Keras API."""

import argparse
import glob
import os


from tensorflow.python.lib.io import file_io


# CHUNK_SIZE specifies the number of lines
# to read in case the file is very large
FILE_PATH = 'checkpoint.{epoch:02d}.hdf5'
CENSUS_MODEL = 'ordinal_face.hdf5'


class ContinuousEval(keras.callbacks.Callback):
    """Continuous eval callback to evaluate the checkpoint once
       every so many epochs.
    """

    def __init__(self,
                 eval_frequency,
                 x, y,
                 eval_prefix,
                 eval_files,
                 learning_rate,
                 job_dir,
                 debug_mode=False):
        self.eval_files = eval_files
        self.eval_prefix = eval_prefix
        self.eval_frequency = eval_frequency
        self.learning_rate = learning_rate
        self.job_dir = job_dir
        self.debug_mode = debug_mode
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
                loss, acc = census_model.evaluate_generator(
                    data_sequence,
                    steps=data_sequence.length)
                print('\nEvaluation epoch[{}] metrics[{:.2f}, {:.2f}] {}'.format(
                    epoch, loss, acc, census_model.metrics_names))
                if self.job_dir.startswith("gs://"):
                    copy_file_to_gcs(self.job_dir, checkpoints[-1])
            else:
                print(
                    '\nEvaluation epoch[{}] (no checkpoints found)'.format(epoch))


def dispatch(train_files,
             eval_files,
             job_dir,
             train_steps,
             eval_steps,
             train_batch_size,
             eval_batch_size,
             learning_rate,
             eval_frequency,
             eval_num_epochs,
             num_epochs,
             checkpoint_epochs,
             image_input_prefix,
             debug_mode):

    x_train, y_train, x_test, y_test, input_shape = create_data()
    logger = logging.getLogger()
    sh = StreamHandler(stdout)
    logger.addHandler(sh)
    logger.setLevel(logging.INFO)
    logger.info('learning_rate=%s' % learning_rate)
    census_model = model.model_fn(learning_rate)

#     try:
#         os.makedirs(job_dir)
#     except:
#         pass

    # Unhappy hack to work around h5py not being able to write to GCS.
    # Force snapshots and saves to local filesystem, then copy them over to
    # GCS.
#     checkpoint_path = FILE_PATH
#     if not job_dir.startswith("gs://"):
#         checkpoint_path = os.path.join(job_dir, checkpoint_path)
#
#     meta_data = get_meta(train_files)
#     indexes = [i for i in range(len(meta_data))]
#     random.shuffle(indexes)
#     meta_data = meta_data.loc[indexes].reset_index(drop=True)

    # Model checkpoint callback
#     checkpoint = keras.callbacks.ModelCheckpoint(
#         checkpoint_path,
#         monitor='val_loss',
#         verbose=1,
#         period=checkpoint_epochs,
#         mode='max')

    # Continuous eval callback
#     evaluation = ContinuousEval(eval_frequency,
#                                 meta_data,
#                                 image_input_prefix,
#                                 eval_files,
#                                 learning_rate,
#                                 job_dir,
#                                 debug_mode)

    # Tensorboard logs callback
#     tblog = keras.callbacks.TensorBoard(
#         log_dir=os.path.join(job_dir, 'logs'),
#         histogram_freq=0,
#         write_graph=True,
#         embeddings_freq=0)
#
#     callbacks = [checkpoint, evaluation, tblog]

    train_data_sequence = DataSequence(
        x_train, y_train,
        batch_size=train_batch_size
    )
    census_model.fit_generator(  # x_train, y_train,
        #model.generator_input(train_files, chunk_size=CHUNK_SIZE),
        train_data_sequence,
        # batch_size=32,
        steps_per_epoch=train_data_sequence.length,
        epochs=num_epochs)

    # Unhappy hack to work around h5py not being able to write to GCS.
    # Force snapshots and saves to local filesystem, then copy them over to
    # GCS.
#     if job_dir.startswith("gs://"):
#         census_model.save(CENSUS_MODEL)
#         copy_file_to_gcs(job_dir, CENSUS_MODEL)
#     else:
#         census_model.save(os.path.join(job_dir, CENSUS_MODEL))

    # Convert the Keras model to TensorFlow SavedModel
#    model.to_savedmodel(census_model, os.path.join(job_dir, 'export'))

# h5py workaround: copy local models over to GCS if the job_dir is GCS.


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
    parser.add_argument('--eval-files',
                        required=True,
                        type=str,
                        help='Evaluation files local or GCS', nargs='+')
    parser.add_argument('--job-dir',
                        required=True,
                        type=str,
                        help='GCS or local dir to write checkpoints and export model')
    parser.add_argument('--train-steps',
                        type=int,
                        default=100,
                        help="""\
                       Maximum number of training steps to perform
                       Training steps are in the units of training-batch-size.
                       So if train-steps is 500 and train-batch-size if 100 then
                       at most 500 * 100 training instances will be used to train.
                      """)
    parser.add_argument('--eval-steps',
                        help='Number of steps to run evalution for at each checkpoint',
                        default=100,
                        type=int)
    parser.add_argument('--train-batch-size',
                        type=int,
                        default=32,
                        help='Batch size for training steps')
    parser.add_argument('--eval-batch-size',
                        type=int,
                        default=32,
                        help='Batch size for evaluation steps')
    parser.add_argument('--learning-rate',
                        type=float,
                        default=0.003,
                        help='Learning rate for Adam')
    parser.add_argument('--eval-frequency',
                        default=1,
                        help='Perform one evaluation per n epochs')
    parser.add_argument('--eval-num-epochs',
                        type=int,
                        default=1,
                        help='Number of epochs during evaluation')
    parser.add_argument('--num-epochs',
                        type=int,
                        default=2,
                        help='Maximum number of epochs on which to train')
    parser.add_argument('--checkpoint-epochs',
                        type=int,
                        default=1,
                        help='Checkpoint per n training epochs')
    parser.add_argument('--image-input-prefix',
                        type=str,
                        default='/home/jiman/facedata/imdb/intermediate',
                        help='Image Input Prefix')
    parser.add_argument('--debug-mode',
                        type=str,
                        default=False,
                        help='debug mode for small data')
    parse_args, unknown = parser.parse_known_args()

    dispatch(**parse_args.__dict__)
