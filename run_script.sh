DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=simple_linear_$DATE
export GCS_JOB_DIR=gs://kceproject-1113-ml/ml-job/$JOB_NAME
echo $GCS_JOB_DIR
export TRAIN_STEPS=5000

gcloud ml-engine jobs submit training $JOB_NAME \
                                    --stream-logs \
                                    --runtime-version 1.4 \
                                    --job-dir $GCS_JOB_DIR \
                                    --module-name trainer.task \
                                    --package-path trainer/ \
                                    --region us-central1 \
                                    -- \
                                    --train-files gs://kceproject-1113-ml/train/train.csv \
                                    --eval-files gs://kceproject-1113-ml/test/test.csv \
                                    --train-steps $TRAIN_STEPS \
                                    --eval-steps 100
