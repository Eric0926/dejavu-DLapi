import utils
import utils_vgg16
import utils_evaluation
from keras.applications.vgg16 import VGG16
import faiss
import sys


def vgg_search(img_url, nb_results):

    # load the index file
    index = faiss.read_index("mini_index.index")

    # A VGG16 pre-trained model
    vgg16_model = VGG16(weights='imagenet', include_top=False)
    # vgg16_model.summary()

    # number of nearest neighbors for each descriptor
    k = 100

    # compute query vector of descriptors
    xq = utils_vgg16.eval_vgg_with_l2_norm_compute_xq_cloud(
        vgg16_model, img_url
    )

    if xq.shape == (0,):
        return None, None, None, None, None

    # actual search
    # xq - the descriptors of the query image
    # index - the whole index of the images in the storage
    # k - number of nearest neighbors for each descriptor
    # I - the indexes of the relevant similar images
    D, I = utils.search(xq, index, k)

    # Return top N file_ids (images' names) according to IC
    top_n, scores, sources = utils_evaluation.eval_retrieve_top_n(
        I, nb_results)

    numbers = list(range(1, 11))

    images = []
    for num in numbers:
        idx = str(top_n[num - 1])
        idx1 = idx[:3]
        idx2 = idx[3:]
        # https://storage.cloud.google.com/evaluation_images/111_14.jpg?authuser=2
        images.append(
            "https://storage.cloud.google.com/evaluation_images/{}_{}.jpg\n".format(idx1, idx2))

    return images


if __name__ == '__main__':
    # image_path = sys.argv[1]
    # top_n, scores = vgg_search(image_path, 10)
    images = vgg_search(
        'https://storage.googleapis.com/flagged_evaluation_images/444_10_r2.png', 10)

    for image in images:
        print(image)

#top_n, scores = vgg_search('https://static01.nyt.com/images/2020/03/19/world/19virus-briefing-notravel/merlin_170711415_4eef2c15-1dac-47f6-b305-531f35736ed7-articleLarge.jpg?quality=75&auto=webp&disable=upscale', 10)
