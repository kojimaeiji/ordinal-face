export PYTHONPATH="~/trainer_mat:$PATHONPATH"

python trainer_mat/predict.py \
  --model-file "gs://kceproject-1113-ml/ml-job/ordinal_face_20180121_184032/ordinal_face.hdf5" \
  --img-file "/home/jiman/deep_learning/Fundations-of-Convolutional-Neural-Networks/week4/exercise_week4/Face Recognition/intermediate/saori/IMG_0092.jpg"
#  --img-file "/home/jiman/deep_learning/Fundations-of-Convolutional-Neural-Networks/week4/exercise_week4/Face Recognition/intermediate/eiji/IMG_20151122_104802.jpg"