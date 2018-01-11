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

def download_mat(gs_path, filename):
    if not os.path.exists("/tmp/"+filename):
      cmd=["gsutil","cp",gs_path+"/"+filename,"/tmp/"]
      #cmd='gsutil ls gs://kceproject-1113-ml'
      retcode = subprocess.check_call(cmd)
      print(retcode)
    return "/tmp/"+filename

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

class CountingSource(iobase.BoundedSource):

  def __init__(self, gs_path, filename):
    self._filename = download_mat(gs_path, filename)
    _meta = loadmat(self._filename)
    db = 'imdb'
    print('loadmat end')
    full_path = _meta[db][0, 0]["full_path"][0]
    dob = _meta[db][0, 0]["dob"][0]  # Matlab serial date number
    #self.gender = _meta[db][0, 0]["gender"][0]
    photo_taken = _meta[db][0, 0]["photo_taken"][0]  # year
    #self.face_score = _meta[db][0, 0]["face_score"][0]
    #self.second_face_score = _meta[db][0, 0]["second_face_score"][0]
    age = [calc_age(photo_taken[j], dob[j]) for j in range(len(dob))]

    print('self etc end')
    self.value = [(full_path[i], age[i]) for i in range(10)]
    self.records_read = Metrics.counter(self.__class__, 'recordsRead')
    self._count = 10 #len(_mat[db][0, 0])
    print('init end')

  def __call__(self):
      return self.value

#   def next(self):
#       #self.records_read.inc()
#       return self.full_path[0], self.age[0]

  def estimate_size(self):
    return self._count

  def get_range_tracker(self, start_position, stop_position):
    print('get tracker start')
    if start_position is None:
      start_position = 0
    if stop_position is None:
      stop_position = self._count
    print('get tracker end')
    return OffsetRangeTracker(start_position, stop_position)

  def read(self, range_tracker):
    print('read start')

    for i in range(self._count):
      if not range_tracker.try_claim(i):
        print('read not try claim')
        return
      self.records_read.inc()
      print('read inner')
    
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

# まずパイプラインを作る
p = beam.Pipeline(options=PipelineOptions())

# 1.配列をパイプラインの入力に設定（４行を入力とする）
rows = (p
       | 'new rows' >> beam.Create(CountingSource('gs://kceproject-1113-ml/imdb', 'imdb.mat')()))

# 2.変換処理として各行の文字列カウントを設定
#word_lengths = rows | beam.Map(len)

# 3.最後に標準出力にカウント数を出力して終わる
class ExtractWordsFn(beam.DoFn):
  def process(self, element):
    path, age = element
    print(path, age)
p_end = rows | beam.ParDo(ExtractWordsFn())
result = p.run()
result.wait_until_finish()