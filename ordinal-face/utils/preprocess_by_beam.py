# -*- coding: utf-8 -*-
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.io.filebasedsource import FileBasedSource
from scipy.io.matlab.mio import loadmat
from apache_beam.io import iobase
import os
import subprocess
from datetime import datetime
from apache_beam.metrics.metric import Metrics
from apache_beam.io.range_trackers import OffsetRangeTracker
import cv2
import argparse
import logging
from apache_beam.options.pipeline_options import SetupOptions
from logging import StreamHandler
from apache_beam.runners.dataflow.dataflow_runner import DataflowRunner
from apache_beam.runners.direct.direct_runner import DirectRunner

KCEGS_ML_ROOT='gs://kceproject-1113-ml'
GS_PATH=KCEGS_ML_ROOT+'/imdb'
GS_UPPATH=KCEGS_ML_ROOT+'/intermediate'
CROP_DIM=60

def del_destination(gs_path):
    cmd="gsutil ls %s " % gs_path
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    print(stdout)
    if stdout:
        cmd=["gsutil", "-m", "rm", "-r", gs_path]
        subprocess.check_call(cmd)
    

def download_gs(gs_path, filename):
    cmd=["gsutil","cp", gs_path+"/"+filename,"/tmp/"]
    retcode = subprocess.check_call(cmd)
    filename = filename.split('/')[-1]
    return "/tmp/"+filename


def upload_gs(gs_path,gs_filename, local_full_filename):
    cmd=["gsutil","cp",local_full_filename, gs_path+'/'+gs_filename]
    retcode = subprocess.check_call(cmd)


def get_meta(meta, db='imdb'):
    full_path = meta[db][0, 0]["full_path"][0]
    dob = meta[db][0, 0]["dob"][0]  # Matlab serial date number
    gender = meta[db][0, 0]["gender"][0]
    photo_taken = meta[db][0, 0]["photo_taken"][0]  # year
    face_score = meta[db][0, 0]["face_score"][0]
    second_face_score = meta[db][0, 0]["second_face_score"][0]
    age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]

    return full_path, dob, gender, photo_taken, face_score, second_face_score, age

def calc_age(taken, dob):
    birth = datetime.fromordinal(max(int(dob) - 366, 1))

    # assume the photo was taken in the middle of the year
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1

def check_face(score, age):
    if score > 1.0 and 0 <= age <=80:
        return True
    return False

class MatSource(iobase.BoundedSource):

  def __init__(self, gs_path, filename, runner):
    self._filename = download_gs(gs_path, filename)
    _meta = loadmat(self._filename)
    #del_destination(GS_UPPATH)
    db = 'imdb'
    logger.info('loadmat end')
    full_path = _meta[db][0, 0]["full_path"][0]
    dob = _meta[db][0, 0]["dob"][0]  # Matlab serial date number
    gender = _meta[db][0, 0]["gender"][0]
    photo_taken = _meta[db][0, 0]["photo_taken"][0]  # year
    face_score = _meta[db][0, 0]["face_score"][0]
    #self.second_face_score = _meta[db][0, 0]["second_face_score"][0]
    length = len(dob)
    logger.info('data length=%s' % length)
    age = [calc_age(photo_taken[j], dob[j]) for j in range(length)]

    logger.info('self etc end')
    self.value = [(full_path[i][0].strip(), gender[i], age[i]) for i in range(length) if check_face(face_score[i], age[i])]
    logger.info('runner=%s' % type(runner))
    if runner == 'DirectRunner':
        self.value = self.value[:10]
    self.records_read = Metrics.counter(self.__class__, 'recordsRead')
    self._count = len(self.value)
    logger.info('final length=%s' % self._count)
    logger.info('init end')

  def __call__(self):
      return self.value

  def estimate_size(self):
    return self._count

  def get_range_tracker(self, start_position, stop_position):
    logger.info('get tracker start')
    if start_position is None:
      start_position = 0
    if stop_position is None:
      stop_position = self._count
    logger.info('get tracker end')
    return OffsetRangeTracker(start_position, stop_position)

  def read(self, range_tracker):
    logger.info('read start')

    for i in range(self._count):
      if not range_tracker.try_claim(i):
        logger.info('read not try claim')
        return
      self.records_read.inc()
      logger.info('read inner')
    
      yield self.full_path[i], self.age[i]

  def split(self, desired_bundle_size, start_position=None,
            stop_position=None):
    if start_position is None:
      start_position = 0
    if stop_position is None:
      stop_position = self._count

    bundle_start = start_position
    while bundle_start < self._count:
      bundle_stop = max(self._count, bundle_start + desired_bundle_size)
      print('bundle split')
      yield iobase.SourceBundle(weight=(bundle_stop - bundle_start),
                                source=(self.full_path,self.age),
                                start_position=bundle_start,
                                stop_position=bundle_stop)
      bundle_start = bundle_stop


# 3.イメージのリサイズ
class ProcessImgFn(beam.DoFn):
  def process(self, element):
    path, _, _ = element
    logger.info(path)
    filepath = download_gs(GS_PATH, path)
    logger.info(filepath)
    img = cv2.imread(filepath)
    img = cv2.resize(img, (CROP_DIM, CROP_DIM))
    cv2.imwrite(filepath,img)
    upload_gs(GS_UPPATH, path, filepath)
    os.remove(filepath)

# csv形式に変換
class ConvertToStr(beam.DoFn):
  def process(self, element):
    path, gender, age = element
    return ['%s, %s, %s' % (path, gender, age)]    

def run(argv=None):
  """Main entry point; defines and runs the wordcount pipeline."""
  parser = argparse.ArgumentParser()
  known_args, pipeline_args = parser.parse_known_args(argv)

  # We use the save_main_session option because one or more DoFn's in this
  # workflow rely on global context (e.g., a module imported at module level).
  pipeline_options = PipelineOptions(pipeline_args)
  pipeline_options.view_as(SetupOptions).save_main_session = True
  p = beam.Pipeline(options=pipeline_options)

  # 1..matから配列を生成し、パイプラインの入力に設定
  rows = (p
           | 'new rows' >> beam.Create(MatSource(GS_PATH, 'imdb.mat', pipeline_options.get_all_options()['runner'])()))
    
        
  p_img = rows | 'process image file path' >> beam.ParDo(ProcessImgFn())
  p_csv = rows | 'produce csv' >> beam.ParDo(ConvertToStr()) | 'write to csv' >> beam.io.WriteToText(GS_UPPATH+'/csv/path_age.csv')
  result = p.run()
  result.wait_until_finish()

if __name__ == '__main__':
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  run()