from google.cloud import datastore, storage
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras import backend as K
from urllib.request import urlopen
import io
import json
import utils
import utils_evaluation
# import faiss
from heapq import nlargest
from sklearn.preprocessing import normalize


# a function for preprocess the input image
def vgg16_preprocess(img_path, siamese=False):
    if siamese == False:

        fd = urlopen(img_path)
        image_file = io.BytesIO(fd.read())

        try:
            # Converting them to size of (224,224,3)
            img = image.load_img(image_file, target_size=(224, 224))
        except:
            return None

    else:
        # Converting them to size of (224,224,3)
        img = image.load_img(img_path, target_size=(224, 224))


    # convert the images to be numpy.ndarray type with shape of (224, 224, 3)
    img_data = image.img_to_array(img)

    # convert to 4-diminsional array with an additional dimension: (number of images, 224, 224, 3)
    img_data = np.expand_dims(img_data, axis=0)

    # The preprocess_input “mode == ‘caffe’” - converts the images from RGB to BGR, then zero-center each color channel
    img_data = preprocess_input(img_data)

    return img_data


def cumpute_vgg_desc(vgg16_model, img_preprocessed):
    # extract the feature vectors of last max pooling layer
    input_m = vgg16_model.input
    output_m = vgg16_model.layers[18].output
    new_m_fc = K.function([input_m], [output_m])

    features = new_m_fc([img_preprocessed])[0]
    features = features[0]
    features = features.reshape(49, 512)

    return features

def convert_nparray_to_List(nparray):
    result = []
    for a in nparray:
        result.append(a.item())
    return result


def create_vgg_desc_datastore(vgg16_model):


    storage_client = storage.Client('adina-image-analysis')
    bucket = storage_client.get_bucket('adina-images')

    datastore_client = datastore.Client('adina-image-analysis')
    kind = "Evaluation_Reddit_Index_VGG"
    n = 0
    for blob in bucket.list_blobs():

        if n % 500 == 0:
            print(n)

        if n <= 1767000:
            n += 1
            continue
        #
        # if n >= 40000:
        #     return

        n+=1

        X = []

        if not (utils.ext(blob.public_url) == '.jpeg' or utils.ext(blob.public_url) == '.png' or utils.ext(
                blob.public_url) == '.jpg'):
            continue

        name = blob.name

        if not (name.startswith('RE') or name.startswith('R')):
            continue

        k = datastore_client.key(kind, utils.del_ext(blob.name))

        entity = datastore_client.get(k)


        if not entity:

            entity = datastore.Entity(key=k, exclude_from_indexes=['VGG16 Descriptors'])
            # entity = datastore.Entity(key=k, exclude_from_indexes=['ORB Descriptors', 'VGG16 Descriptors'])

            try:
                img_preprocessed = vgg16_preprocess(blob.public_url)

            except AssertionError as e0:
                print("e0", e0)
                print(blob.public_url)
                continue

            if img_preprocessed is None:
                print("None")
                print(blob.public_url)
                continue

            try:
                features = cumpute_vgg_desc(vgg16_model, img_preprocessed)

            except AssertionError as e1:
                print("e1", e1)
                print(blob.public_url)
                continue

            try:
                for i in range(features.shape[0]):
                    X.append(convert_nparray_to_List(features[i]))
                    # X.append(utils.convert_intList_to_bit(list(features[i])))

                des_json = json.dumps(X)

                entity.update(
                    {'VGG16 Descriptors': des_json, 'Indexed(VGG16)': "No"})

                datastore_client.put(entity)

            except AssertionError as e2:
                print("e2", e2)
                print(blob.public_url)
                continue
    return


def eval_vgg_index(index2, kind):

    datastore_client = datastore.Client('adina-image-analysis')

    q = datastore_client.query(kind=kind)
    q.add_filter('Indexed(VGG16)', '=', "No")
    q.keys_only()
    q_results = list(q.fetch())
    print("len_q_results", len(q_results))

    h= 0
    desc_all = []
    filenames = []
    looped_entities = []

    for key_entity in q_results:

        key = datastore_client.key(kind, key_entity.key.id_or_name)
        image_entity = datastore_client.get(key)

        # the_key = key_entity.key
        # q2 = datastore_client.query(kind=kind)
        # q2.key_filter(the_key)
        # image_entity = (list(q2.fetch()))[0]

        if utils.check_if_indexed(image_entity,'Indexed(VGG16)') == True:
            continue

        desc = json.loads(image_entity['VGG16 Descriptors'])

        image_entity_name = image_entity.key.id_or_name

        try:
            image_entity_name = int(image_entity_name)
        except:
            print(image_entity_name)
            continue

        for i in range(len(desc)):
            desc_all.append(desc[i])
            filenames.append(image_entity_name)

        looped_entities.append(image_entity)
        h += 1

    np_desc_all = np.array(desc_all).astype('float32')
    filenames = np.array(filenames)
    filenames = filenames.astype(int)

    try:
        index2.add_with_ids(np_desc_all, filenames)
    except ValueError as e:
        print(e)
        print("filenames", filenames.shape, filenames)

    # write the index to a file
    faiss.write_index(index2, "eval_flagged_originals_vgg16_index.index")

    # write the files to the bucket
    utils.add_file_to_bucket("eval_flagged_originals_vgg16_index.index")

    utils.set_indexed_Yes(looped_entities, 'Indexed(VGG16)')
    print("set_indexed_Yes", h)

    return index2

def eval_vgg_with_l2_norm_index(index2, kind, batch_size):

    datastore_client = datastore.Client('adina-image-analysis')

    q = datastore_client.query(kind=kind)
    q.add_filter('Indexed(VGG16)', '=', "No")
    q.keys_only()
    q_results = list(q.fetch())
    print("len_q_results", len(q_results))

    h = 0
    c = 0
    desc_all = []
    filenames = []
    looped_entities = []
    indexed_entities = []

    for key_entity in q_results:

        key = datastore_client.key(kind, key_entity.key.id_or_name)
        image_entity = datastore_client.get(key)

        if utils.check_if_indexed(image_entity,'Indexed(VGG16)') == True:
            continue

        if c < batch_size:
            desc = json.loads(image_entity['VGG16 Descriptors'])
            desc = np.array(desc)
            desc = normalize(desc, norm='l2')

            image_entity_name = image_entity.key.id_or_name

            try:
                image_entity_name = int(image_entity_name)
            except:
                print(image_entity_name)
                continue

            looped_entities.append(image_entity)

            for i in range(desc.shape[0]):
                desc_all.append(convert_nparray_to_List(desc[i]))
                filenames.append(image_entity_name)
            c += 1
            h += 1

        else:
            np_desc_all = np.array(desc_all).astype('float32')
            filenames = np.array(filenames)
            filenames = filenames.astype(int)

            try:
                index2.add_with_ids(np_desc_all, filenames)
            except ValueError as e:
                print(e)
                print("filenames", filenames.shape, filenames)

            indexed_entities.extend(looped_entities)

            c = 0
            h += 1
            desc_all = []
            filenames = []

        if h % 1000 == 0:
            print("batches_index", h, "!!!")

        if h % 10000 == 0:

            # write the index to a file
            faiss.write_index(index2, "eval_full_vgg16_l2_index.index")

            # write the files to the bucket
            utils.add_file_to_bucket("eval_full_vgg16_l2_index.index")

            utils.set_indexed_Yes(looped_entities, 'Indexed(VGG16)')
            print("set_indexed_Yes", h)

            indexed_entities = []

    return index2, indexed_entities


def eval_vgg_compute_xq_cloud(vgg16_model, img_url):
    """compute the query vector of descriptors from the image url"""

    xq = []

    try:

        img_preprocessed = vgg16_preprocess(img_url)
        features = cumpute_vgg_desc(vgg16_model, img_preprocessed)

    except AssertionError as e1:
        print("e1", e1)

    try:
        for i in range(features.shape[0]):
            xq.append(convert_nparray_to_List(features[i]))

    except AssertionError as e2:
        print("e2", e2)

    return np.array(xq).astype('float32')

def eval_vgg_with_l2_norm_compute_xq_cloud(vgg16_model, img_url):
    """compute the query vector of descriptors from the image url"""

    xq = []

    try:

        img_preprocessed = vgg16_preprocess(img_url)
        features = cumpute_vgg_desc(vgg16_model, img_preprocessed)
        features = normalize(features, norm='l2')

    except AssertionError as e1:
        print("e1", e1)

    try:
        for i in range(features.shape[0]):
            xq.append(convert_nparray_to_List(features[i]))

    except AssertionError as e2:
        print("e2", e2)

    return np.array(xq).astype('float32')


