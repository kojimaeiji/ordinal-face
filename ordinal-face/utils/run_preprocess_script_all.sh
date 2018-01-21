export BUCKET="gs://kceproject-1113-ml"
export PROJECT="kceproject-1113"
python preprocess_by_beam.py \
  --project $PROJECT \
  --runner DirectRunner \
  --staging_location $BUCKET/staging \
  --temp_location $BUCKET/temp \
  --requirements_file requirements.txt \
  --disk_size_gb 30 \
  --num_workers 6 \
  --setup_file /home/jiman/workspace/ordinal-face/ordinal-face/utils/setup.py \
  --worker_machine_type n1-standard-4
  