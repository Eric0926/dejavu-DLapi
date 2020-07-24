import numpy as np
import cv2
import os
from sklearn.decomposition import PCA
import time
import pickle
import pandas as pd
from google.cloud import datastore, storage
import faiss
from urllib.request import FancyURLopener
import json
from heapq import nlargest
from keras.preprocessing import image
from urllib.request import urlopen
import random
import imageio


class MyOpener(FancyURLopener):
    version = 'My new User-Agent'  # Set this to a string you want for your user agent


def convert_intList_to_bit(l):
    """converts a 32-bytes vector in a 256 bits list"""
    result = []
    for a in l:
        result += [int(n) for n in bin(a)[2:].zfill(8)]
    return result


def find_kind(filename):
    if filename.startswith('4C'):
        return 'chan'
    elif filename.startswith('RE'):
        return 'redd'
    elif filename.startswith('R'):
        return 'redd'


# Returns image extension (type)
def ext(url):
    '''
    Takes in an image url and returns the extension
    '''

    s = url.split('.')
    ext = '.' + s[-1]

    return ext


def del_ext(filename):
    """delete the extension of the filename"""
    filename = filename.replace('.jpeg', '')
    filename = filename.replace('.png', '')
    filename = filename.replace('.gif', '')
    filename = filename.replace('.jpg', '')
    filename = filename.replace('.webm', '')
    return filename


def del_breakingNews_pre_name(filename):
    """delete the "222_ of the filename"""
    filename = filename.replace('222_', '')

    return filename

# Returns image filename


def get_filename(url):
    '''
    Takes in an image url and returns the filename
    '''

    full_name = url.split('/')

    filename = del_ext(full_name[-1])

    name = del_breakingNews_pre_name(filename)

    return int(name)


def url_to_img(url):
    """download the image, convert it to a NumPy array, and then read
    it into OpenCV format"""
    myopener = MyOpener()
    resp = myopener.open(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


def compute_ORB(img, orb, show_image=False):
    """compute ORB descriptors"""

    key_points, des = orb.detectAndCompute(img, None)

    if show_image:
        try:
            img2 = cv2.drawKeypoints(img, key_points, outImage=np.array(
                []), color=(0, 255, 0), flags=0)
            #plt.imshow(img2), plt.show()
        except:
            None

    return des


def load_or_add_orb(img_name, img, orb):

    datastore_client = datastore.Client('adina-image-analysis')
    kind = "Images Descriptors And Indexing"

    k = datastore_client.key(kind, del_ext(img_name))

    entity = datastore_client.get(k)

    if not entity:

        entity = datastore.Entity(
            key=k, exclude_from_indexes=['The Descriptors'])

        try:
            des = compute_ORB(img, orb, show_image=False)

            if des.any() != None:
                Number_Of_Dscriptors = des.shape[0]
                Descriptors_Length = des.shape[1]

                des_list = des.tolist()
                des_json = json.dumps(des_list)

                entity.update(
                    {'Number Of Descriptors': Number_Of_Dscriptors, 'Descriptors Length (byts)': Descriptors_Length,
                     'The Descriptors': des_json})

                try:
                    datastore_client.put(entity)
                    return entity['The Descriptors']

                except:
                    return None

            else:
                return None

        except:

            return None

    else:
        return entity['The Descriptors']


def create_desc_datastore(orb):

    storage_client = storage.Client('adina-image-analysis')
    bucket = storage_client.get_bucket('adina-images')

    datastore_client = datastore.Client('adina-image-analysis')
    kind = "Images Descriptors And Indexing"

    url_err_count = 0
    url_err_list = []

    orb_err_count = 0
    orb_err_list = []

    p = 0

    for blob in bucket.list_blobs():

        X = []

        if not (ext(blob.public_url) == '.jpeg' or ext(blob.public_url) == '.png' or ext(
                blob.public_url) == '.jpg'):
            continue

        try:
            img = url_to_img(blob.public_url)

        except:
            url_err_count += 1
            url_err_list.append(blob.public_url)
            continue

        k = datastore_client.key(kind, del_ext(blob.name))

        entity = datastore_client.get(k)

        if not entity:

            entity = datastore.Entity(
                key=k, exclude_from_indexes=['The Descriptors'])

            try:

                des = compute_ORB(img, orb, show_image=False)
                if des.any() != None:

                    Number_Of_Dscriptors = des.shape[0]
                    Descriptors_Length = des.shape[1]

                    for i in range(des.shape[0]):
                        X.append(convert_intList_to_bit(list(des[i])))

                    des_json = json.dumps(X)

                    entity.update(
                        {'Number Of Descriptors': Number_Of_Dscriptors,
                         'Descriptors Length (byts)': Descriptors_Length,
                         'The Descriptors': des_json, 'Is Indexed': "No"})

                    datastore_client.put(entity)

                    p += 1
                    if p % 1000 == 0:
                        print(p, 'images processed')

                else:
                    orb_err_count += 1
                    orb_err_list.append(str(blob.name))
                    continue

            except:
                orb_err_count += 1
                orb_err_list.append(str(blob.name))
                continue

    print(url_err_count, "errors while trying to convert URL to images. The images are:", url_err_list)
    print(orb_err_count, "errors while trying to compute_ORB descriptors. The images are:", orb_err_list)

    return True


# def training_create_desc_datastore(orb):
#
#     storage_client = storage.Client('adina-image-analysis')
#     bucket = storage_client.get_bucket('adina-images')
#
#     datastore_client = datastore.Client('adina-image-analysis')
#     kind = "Images Descriptors And Indexing"
#
#     url_err_count = 0
#     url_err_list = []
#
#     orb_err_count = 0
#     orb_err_list = []
#
#     p = 0
#
#     for blob in bucket.list_blobs():
#
#         X = []
#
#         if not ( ext(blob.public_url) == '.jpeg' or ext(blob.public_url) == '.png' or ext(blob.public_url) == '.jpg' ):
#             continue
#
#         try:
#             img = url_to_img(blob.public_url)
#
#         except:
#             url_err_count += 1
#             url_err_list.append(blob.public_url)
#             continue
#
#         k = datastore_client.key(kind, del_ext(blob.name))
#
#         entity = datastore.Entity(key=k, exclude_from_indexes=['The Descriptors'])
#
#         try:
#             des = compute_ORB(img, orb, show_image=False)
#
#             if des.any() != None:
#
#                 Number_Of_Dscriptors = des.shape[0]
#                 Descriptors_Length = des.shape[1]
#
#                 for i in range(des.shape[0]):
#                     X.append(convert_intList_to_bit(list(des[i])))
#
#                 # des_list = X.tolist()
#                 des_json = json.dumps(X)
#
#                 entity.update(
#                     {'Number Of Descriptors': Number_Of_Dscriptors, 'Descriptors Length (byts)': Descriptors_Length,
#                      'The Descriptors': des_json})
#
#                 datastore_client.put(entity)
#
#                 p += 1
#                 if p % 1000 == 0:
#                     print(p, 'images processed')
#
#             else:
#                 orb_err_count += 1
#                 orb_err_list.append(str(blob.name))
#                 continue
#
#         except:
#             orb_err_count += 1
#             orb_err_list.append(str(blob.name))
#             continue
#
#     print(url_err_count, "errors while trying to convert URL to images. The images are:", url_err_list)
#     print(orb_err_count, "errors while trying to compute_ORB descriptors. The images are:", orb_err_list)
#
#     return True


def bootstrap_images(n, kind):

    datastore_client = datastore.Client('adina-image-analysis')

    q = datastore_client.query(kind=kind)
    q.keys_only()
    q_results = list(q.fetch())

    n_keys_for_pca = np.random.choice(q_results, n, replace=False)

    return n_keys_for_pca


def create_pca(n_keys_for_pca, kind, label):

    print("n_keys_for_pca", type(n_keys_for_pca),
          len(n_keys_for_pca), n_keys_for_pca[1:10])

    datastore_client = datastore.Client('adina-image-analysis')

    desc_all = []

    for k in n_keys_for_pca:
        the_key = k.key
        q = datastore_client.query(kind=kind)
        q.key_filter(the_key)
        image_entity = (list(q.fetch()))[0]
        desc = json.loads(image_entity[label])

        for i in range(len(desc)):
            desc_all.append(desc[i])

    print("in create_pca the desc_all is", type(desc_all), len(desc_all))

    # j = 0
    # for j in range(len(desc_all)-101):
    #     try:
    #         np_desc_all = np.array(desc_all[j:j+100]).astype('float32')
    #         j += 100
    #     except:
    #         print(desc_all[j:j+100])

    np_desc_all = np.array(desc_all).astype('float32')

    # np_desc_all = np.asarray(desc_all, dtype=np.float32)

    print("in create_pca the np_desc_all is",
          type(np_desc_all), np_desc_all.shape)

    desc_matrix_post_pca, pca = reduce_DB_dim(
        np_desc_all, d=128, return_pca=True)

    return pca


def reduce_DB_dim(xb, d, return_pca=False):
    """applies PCA to matrix xb to reduce dimension to d"""
    start_time = time.time()
    pca = PCA(n_components=d)
    print("in reduce_DB_dim the xb is", type(xb), xb.shape)
    pca.fit(xb)
    xb = pca.transform(xb)

    print('time to perform PCA :', time.time() - start_time)
    print('shape of database after PCA:', xb.shape)

    if return_pca:
        return xb, pca

    return xb


def ids_to_numbers(image_entity_name):
    if image_entity_name.startswith('4C'):
        image_entity_name = image_entity_name.replace('4C', '444')
    elif image_entity_name.startswith('RE'):
        image_entity_name = image_entity_name.replace('RE', '777')
    elif image_entity_name.startswith('R'):
        image_entity_name = image_entity_name.replace('R', '77')
    elif image_entity_name.startswith('JO'):
        image_entity_name = image_entity_name.replace('JO', '999')
    return image_entity_name


def numbers_to_ids(image_entity_name):
    if image_entity_name.startswith('444'):
        image_entity_name = image_entity_name.replace('444', '4C')
    elif image_entity_name.startswith('777'):
        image_entity_name = image_entity_name.replace('777', 'RE')
    elif image_entity_name.startswith('77'):
        image_entity_name = image_entity_name.replace('77', 'R')
    elif image_entity_name.startswith('999'):
        image_entity_name = image_entity_name.replace('999', 'JO')
    return image_entity_name


def check_if_indexed(image_entity, label):
    try:
        is_indexed = image_entity[label]

        if is_indexed == "Yes":
            return True

        else:
            return False

    except AttributeError:
        return False


def set_indexed_No(kind, label):
    datastore_client = datastore.Client('adina-image-analysis')

    q = datastore_client.query(kind=kind)
    q.keys_only()
    q_results = list(q.fetch())

    i = 0

    for key_entity in q_results:

        the_key = key_entity.key
        q2 = datastore_client.query(kind=kind)
        q2.key_filter(the_key)
        image_entity = (list(q2.fetch()))[0]
        image_entity[label] = 'No'
        datastore_client.put(image_entity)

        # i += 1
        # if i%1000 == 0:
        #     print(i, "set to No")

    return


def set_indexed_Yes(indexed_entities, label):
    datastore_client = datastore.Client('adina-image-analysis')
    for i_e in indexed_entities:
        i_e[label] = 'Yes'
        datastore_client.put(i_e)
    return


def batches_index(index2, pca, batch_size):

    datastore_client = datastore.Client('adina-image-analysis')

    q = datastore_client.query(kind='Images Descriptors And Indexing')
    q.add_filter('Is Indexed', '=', "No")
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

        the_key = key_entity.key
        q2 = datastore_client.query(kind='Images Descriptors And Indexing')
        q2.key_filter(the_key)
        image_entity = (list(q2.fetch()))[0]

        if check_if_indexed(image_entity, 'Is Indexed') == True:
            continue

        if c < batch_size:
            desc = json.loads(image_entity['The Descriptors'])
            image_entity_name = image_entity.key.id_or_name
            image_entity_name = ids_to_numbers(image_entity_name)
            try:
                image_entity_name = int(image_entity_name)
            except:
                print(image_entity_name)
                continue
            looped_entities.append(image_entity)
            for i in range(len(desc)):
                desc_all.append(desc[i])
                filenames.append(image_entity_name)
            c += 1
            h += 1

        else:
            np_desc_all = np.array(desc_all).astype('float32')
            des_orb_post_pca = pca.transform(np_desc_all)
            filenames = np.array(filenames)
            filenames = filenames.astype(int)
            try:
                index2.add_with_ids(des_orb_post_pca, filenames)
            except ValueError as e:
                print(e)
                print("filenames", filenames.shape, filenames)
                print("c", c)
                print("h", h)
                print("des_orb_post_pca", type(
                    des_orb_post_pca), des_orb_post_pca)
                continue
            indexed_entities.extend(looped_entities)
            c = 0
            h += 1
            desc_all = []
            filenames = []

        if h % 10000 == 0:
            print("batches_index", h, "!!!")

        if h % 10000 == 0:

            # write the index to a file
            faiss.write_index(index2, "index_file.index")
            print("write_index", h)

            # write the files to the bucket
            add_file_to_bucket("index_file.index")
            print("add_file_to_bucket", h)

            set_indexed_Yes(indexed_entities, 'Is Indexed')
            print("set_indexed_Yes", h)

            indexed_entities = []

    return index2, indexed_entities


def compute_xq_cloud(orb, img_url, pca):
    """compute the query vector of descriptors from the image url"""

    img = url_to_img(img_url)
    xq = []

    try:
        des = compute_ORB(img, orb, show_image=True)
        if des.any() != None:

            # update database
            for i in range(des.shape[0]):
                xq.append(convert_intList_to_bit(list(des[i])))

    except AttributeError:
        None

    if len(xq) == 0:
        return xq
    else:
        return np.array(pca.transform(xq)).astype('float32')


def search(xq, index, k):
    """compute the search in the database"""
    start_time = time.time()
    D, I = index.search(xq, k)  # actual search
    # print('time to perform actual search :', time.time() - start_time)
    return D, I


# Calculates how many times each des_file_ids appears in I, and according to the desired N, it returns the mapping to file_ids (images' names) of the highest N in the list.
def retrieve_top_n(I, n=5):

    indexes_list = list(I.flatten())
    distinct_indexes = list(set(indexes_list))
    filename_count = dict.fromkeys(distinct_indexes, 0)

    for index in indexes_list:
        filename_count[index] += 1

    top_n = nlargest(n, filename_count, key=filename_count.get)

    num = 1
    top_n_ids = []

    for num_i in top_n:

        str_i = numbers_to_ids(str(num_i))

        top_n_ids.append(str_i)

        print("Top N_", num, ":")
        print(str_i, "with score:", filename_count[num_i])

        num += 1

    return top_n_ids


def add_file_to_bucket(filename):
    storage_client = storage.Client(project='adina-image-analysis')
    bucket = storage_client.get_bucket('index-files')
    blob = storage.Blob(filename, bucket)
    blob.upload_from_filename(filename)


def get_index_from_bucket(index_filename):
    storage_client = storage.Client(project='adina-image-analysis')
    bucket = storage_client.get_bucket('index-files')
    blob = storage.Blob(index_filename, bucket)
    blob.download_to_filename("file_download.index")
    return faiss.read_index("file_download.index")


def load_file_from_bucket(filename):
    storage_client = storage.Client(project='adina-image-analysis')
    bucket = storage_client.get_bucket('index-files')
    blob = storage.Blob(filename, bucket)
    blob.download_to_filename(filename)
    with open(filename, 'rb') as f:
        des_imgID = pickle.load(f)
    return des_imgID


# def get_filename(index_id):
#     """retrieve filename from index_id"""
#
#     entity_list = []
#     kinds = ['redd', 'chan']
#     datastore_client = datastore.Client('adina-image-analysis')
#
#     for kind in kinds:
#         query = datastore_client.query(kind=kind)
#         query.add_filter('index_id', '=', index_id)
#         print(query)
#         entity_list += list(query.fetch())
#         print(entity_list)
#
#     if len(entity_list) > 1:
#         raise Exception("Multiple images for this index_id")
#     elif not entity_list:
#         raise Exception("index_id not found")
#     else:
#         entity = entity_list[0]
#         ext = entity['ext']
#         ID = entity.key.id_or_name
#         return ID + ext


def count_storage(bucket_name):

    storage_client = storage.Client('adina-image-analysis')
    bucket = storage_client.get_bucket(bucket_name)

    count = 0

    for blob in bucket.list_blobs():
        count += 1

    print("count storage of", bucket_name, count)

    return count


def count_datastore(kind):

    datastore_client = datastore.Client('adina-image-analysis')
    q = datastore_client.query(kind=kind)
    q.keys_only()
    q_results = list(q.fetch())

    count = 0

    for key_entity in q_results:
        count += 1

    print("count datastore of", kind, count)

    return count
