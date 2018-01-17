DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=simple_linear_$DATE
export GCS_JOB_DIR=/home/jiman/mljob
echo $GCS_JOB_DIR
export TRAIN_STEPS=100

gcloud ml-engine local train \
  --job-dir $GCS_JOB_DIR \
  --module-name trainer.task \
  --package-path trainer/ \
  -- \
  --train-files "gs://kceproject-1113-ml/intermediate/csv/path_age.csv-00000-of-00221" \
  --eval-files "gs://kceproject-1113-ml/intermediate/csv/path_age.csv-00000-of-00221" \
  --debug-mode True \
  --train-steps $TRAIN_STEPS \
  --eval-steps 100 \
  --image-input-prefix gs://kceproject-1113-ml/intermediate