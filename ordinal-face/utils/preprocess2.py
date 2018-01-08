import argparse
import glob
import logging
import multiprocessing as mp
import os
import time

import cv2

from align_dlib import AlignDlib

logger = logging.getLogger(__name__)

align_dlib = AlignDlib(os.path.join(os.path.dirname(__file__), 'shape_predictor_68_face_landmarks.dat'))


def main(input_dir, output_dir, crop_dim, skip, resize_only):
    start_time = time.time()
    pool = mp.Pool(processes=mp.cpu_count())

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_dir in os.listdir(input_dir):
        image_output_dir = os.path.join(output_dir, os.path.basename(os.path.basename(image_dir)))
        if not os.path.exists(image_output_dir):
            os.makedirs(image_output_dir)

    image_paths = glob.glob(os.path.join(input_dir, '**/*.jpg'))
    for index, image_path in enumerate(image_paths):
        image_output_dir = os.path.join(output_dir, os.path.basename(os.path.dirname(image_path)))
        output_path = os.path.join(image_output_dir, os.path.basename(image_path))
        if not os.path.exists(output_path) or skip == 0:
            pool.apply_async(preprocess_image, (image_path, output_path, crop_dim, resize_only))

    pool.close()
    pool.join()
    logger.info('Completed in {} seconds'.format(time.time() - start_time))


def preprocess_image(input_path, output_path, crop_dim, resize_only):
    """
    Detect face, align and crop :param input_path. Write output to :param output_path
    :param input_path: Path to input image
    :param output_path: Path to write processed image
    :param crop_dim: dimensions to crop image to
    """
    image = _process_image(input_path, crop_dim, resize_only)
    if image is not None:
        logger.debug('Writing processed file: {}'.format(output_path))
        cv2.imwrite(output_path, image)
    else:
        logger.warning("Skipping filename: {}".format(input_path))


def _process_image(filename, crop_dim, resize_only=False):
    image = None
    aligned_image = None

    image = _buffer_image(filename)

    if image is not None and not resize_only:
        aligned_image = _align_image(image, crop_dim)
    if image is not None and resize_only:
        aligned_image = cv2.resize(image, (crop_dim, crop_dim), interpolation = cv2.INTER_CUBIC)
    else:
        raise IOError('Error buffering image: {}'.format(filename))

    return aligned_image


def _buffer_image(filename):
    logger.debug('Reading image: {}'.format(filename))
    image = cv2.imread(filename, )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _align_image(image, crop_dim):
    bb = align_dlib.getLargestFaceBoundingBox(image)
    aligned = align_dlib.align(crop_dim, image, bb, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
    if aligned is not None:
        aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    return aligned


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--input-dir', type=str, action='store', default='data', dest='input_dir')
    parser.add_argument('--output-dir', type=str, action='store', default='output', dest='output_dir')
    parser.add_argument('--crop-dim', type=int, action='store', default=180, dest='crop_dim',
                        help='Size to crop images to')

    parser.add_argument('--skip', type=int, action='store', default=0, dest='skip', help='skip exists')
    parser.add_argument('--resize-only', type=int, action='store', default=0, dest='resize_only', help='resize only')
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.crop_dim, args.skip, args.resize_only)
