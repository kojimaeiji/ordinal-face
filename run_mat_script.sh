DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=ordinal_face_$DATE
export GCS_JOB_DIR=gs://kceproject-1113-ml/ml-job/$JOB_NAME
echo $GCS_JOB_DIR

gcloud ml-engine jobs submit training $JOB_NAME \
  --stream-logs \
  --runtime-version 1.4 \
  --job-dir $GCS_JOB_DIR \
  --module-name trainer_mat.task \
  --package-path trainer_mat/ \
  --region us-central1 \
  --scale-tier basic-gpu \
  -- \
  --train-files "gs://kceproject-1113-ml/ordinal-face/wiki_processed_all.mat" \
  --train-batch-size 64 \
  --model-file "gs://kceproject-1113-ml/ml-job/ordinal_face_20180129_235715/ordinal_face.hdf5" \ 
  --lam 0.0 \
  --dropout 0.0 \
  --num-epochs 50 \
  --learning-rate 0.001
