DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=ordinal_face_$DATE
export GCS_JOB_DIR=/home/jiman/mljob
echo $GCS_JOB_DIR
export TRAIN_STEPS=100
rm -rf /home/jiman/mljob

gcloud ml-engine local train \
  --job-dir $GCS_JOB_DIR \
  --module-name trainer_mat.task \
  --package-path trainer_mat/ \
  -- \
  --train-files "gs://kceproject-1113-ml/ordinal-face/wiki_process_10000.mat" \
  --train-batch-size 64 \
  --num-epochs 10 \
  --lam 0.0 \
  --dropout 0.5 \
  --learning-rate 0.001
