import numpy as np
import cv2
import scipy.io
import argparse
from tqdm import tqdm
from utils import get_meta
import random

def get_args():
    parser = argparse.ArgumentParser(description="This script cleans-up noisy labels "
                                                 "and creates database for training.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="path to output database mat file")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="path to input database mat file")
    parser.add_argument("--db", type=str, default="wiki",
                        help="dataset; wiki or imdb")
    parser.add_argument("--img_size", type=int, default=32,
                        help="output image size")
    parser.add_argument("--min_score", type=float, default=1.0,
                        help="minimum face_score")
    parser.add_argument("--nums", "-n", type=int, default=100, help="number of samples")
    args = parser.parse_args()
    return args

def convert_to_ordinal(age):
    age_vec = np.zeros(shape=(100,), dtype=np.float32)
    for i in range(0, age_vec.shape[0]):
        if age > i:
            age_vec[i] = 1.0
    #print('age=%s, age_vec = %s' % (age, age_vec))
    return age_vec


def main():
    args = get_args()
    output_path = args.output
    input_path = args.input
    db = args.db
    img_size = args.img_size
    min_score = args.min_score
    nums = args.nums

    root_path = "data/{}_crop/".format(db)
    full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta(input_path, db)

    out_genders = []
    out_ages = []
    out_imgs = []
    out_full_paths = []

    for i in tqdm(range(len(face_score))):
        if face_score[i] < min_score:
            continue

        if (~np.isnan(second_face_score[i])) and second_face_score[i] > 0.0:
            continue

        if ~(0 <= age[i] <= 100):
            continue

        if np.isnan(gender[i]):
            continue

#         out_genders.append(int(gender[i]))
        out_full_paths.append(full_path[i])
        age[i] = convert_to_ordinal(age[i])
        out_ages.append(age[i])
#         img = cv2.imread(root_path + str(full_path[i][0]))
#         out_imgs.append(cv2.resize(img, (img_size, img_size)))
    indexes = [i for i in range(len(out_ages))]
    indexes = random.sample(indexes, nums)
    out_full_paths = [out_full_paths[index] for index in indexes]
    out_ages = [out_ages[index] for index in indexes]
    output = {"full_path": np.array(out_full_paths),
              "age": np.array(out_ages)}
    scipy.io.savemat(output_path, output)


if __name__ == '__main__':
    main()
