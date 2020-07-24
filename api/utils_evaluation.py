import utils
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
from sklearn.decomposition import PCA
import io
import requests


def create_evaluation_desc_datastore(orb, kind):

    storage_client = storage.Client('adina-image-analysis')
    bucket = storage_client.get_bucket('sampled_social_media_images')

    datastore_client = datastore.Client('adina-image-analysis')

    for blob in bucket.list_blobs():

        X = []

        if not (utils.ext(blob.public_url) == '.jpeg' or utils.ext(blob.public_url) == '.png' or utils.ext(blob.public_url) == '.jpg'):
            continue

        try:
            img = utils.url_to_img(blob.public_url)

        except:
            continue

        k = datastore_client.key(kind, utils.del_ext(blob.name))
        entity = datastore_client.get(k)

        if not entity:

            entity = datastore.Entity(
                key=k, exclude_from_indexes=['ORB Descriptors'])
            try:

                des = utils.compute_ORB(img, orb, show_image=False)
                if des.any() != None:

                    for i in range(des.shape[0]):
                        X.append(utils.convert_intList_to_bit(list(des[i])))

                    des_json = json.dumps(X)

                    entity.update(
                        {'ORB Descriptors': des_json, 'Indexed(ORB)': "No"})

                    datastore_client.put(entity)

                else:
                    continue

            except:
                continue
    return


def sample_social_media_datastore(orb, quantity, kind):
    storage_client = storage.Client('adina-image-analysis')
    bucket = storage_client.get_bucket('adina-images')
    datastore_client = datastore.Client('adina-image-analysis')

    p = utils.count_storage('adina-images')

    l = list(range(p))

    choseen = random.sample(l, quantity)

    i = 0
    s = 31

    for blob in bucket.list_blobs():

        i += 1

        X = []

        if i in choseen:

            if not (utils.ext(blob.public_url) == '.jpeg' or utils.ext(blob.public_url) == '.png' or utils.ext(
                    blob.public_url) == '.jpg'):
                continue

            if not (blob.name.startswith('4C') or blob.name.startswith('RE') or blob.name.startswith('R')):
                continue

            try:
                img = utils.url_to_img(blob.public_url)

            except:
                continue

            try:

                des = utils.compute_ORB(img, orb, show_image=False)
                if des.any() != None:

                    for i in range(des.shape[0]):
                        X.append(utils.convert_intList_to_bit(list(des[i])))

                    des_json = json.dumps(X)

                    name = '333_' + str(s)
                    s += 1

                    k = datastore_client.key(kind, name)

                    entity = datastore.Entity(key=k, exclude_from_indexes=[
                                              'ORB Descriptors', 'VGG16 Descriptors'])

                    # print("1")
                    # print(entity)

                    entity.update(
                        {'ORB Descriptors': des_json, 'Indexed(ORB)': "No"})

                    # print("2")
                    # print(entity)

                    datastore_client.put(entity)

                    # print("3")
                    # print(entity)

                    if s % 1000 == 0:
                        print(s, 'images processed')

                else:
                    continue

            except:
                continue
    return


def eval_index(index2, kind):

    datastore_client = datastore.Client('adina-image-analysis')

    q = datastore_client.query(kind=kind)
    q.add_filter('Indexed(ORB)', '=', "No")
    q.keys_only()
    q_results = list(q.fetch())
    print("len_q_results", len(q_results))

    h = 0
    desc_all = []
    filenames = []
    looped_entities = []

    for key_entity in q_results:

        the_key = key_entity.key
        q2 = datastore_client.query(kind=kind)
        q2.key_filter(the_key)
        image_entity = (list(q2.fetch()))[0]

        if utils.check_if_indexed(image_entity, 'Indexed(ORB)') == True:
            continue

        desc = json.loads(image_entity['ORB Descriptors'])
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
    faiss.write_index(index2, "eval_flagged_originals_index.index")

    # write the files to the bucket
    utils.add_file_to_bucket("eval_flagged_originals_index.index")

    utils.set_indexed_Yes(looped_entities, 'Indexed(ORB)')
    print("set_indexed_Yes", h)

    return index2


def eval_index_with_pca(index2, kind, pca):

    datastore_client = datastore.Client('adina-image-analysis')

    q = datastore_client.query(kind=kind)
    q.add_filter('Indexed(ORB)', '=', "No")
    q.keys_only()
    q_results = list(q.fetch())
    print("len_q_results", len(q_results))

    h = 0
    desc_all = []
    filenames = []
    looped_entities = []

    for key_entity in q_results:

        the_key = key_entity.key
        q2 = datastore_client.query(kind=kind)
        q2.key_filter(the_key)
        image_entity = (list(q2.fetch()))[0]

        if utils.check_if_indexed(image_entity, 'Indexed(ORB)') == True:
            continue

        desc = json.loads(image_entity['ORB Descriptors'])
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
    des_orb_post_pca = pca.transform(np_desc_all)
    filenames = np.array(filenames)
    filenames = filenames.astype(int)

    try:
        index2.add_with_ids(des_orb_post_pca, filenames)
    except ValueError as e:
        print(e)
        print("filenames", filenames.shape, filenames)

    # write the index to a file
    faiss.write_index(index2, "eval_full3_index.index")

    # write the files to the bucket
    utils.add_file_to_bucket("eval_full3_index.index")

    utils.set_indexed_Yes(looped_entities, 'Indexed(ORB)')
    print("set_indexed_Yes", h)

    return index2


def eval_compute_xq_cloud(orb, img_url):
    """compute the query vector of descriptors from the image url"""

    img = utils.url_to_img(img_url)
    xq = []

    try:
        des = utils.compute_ORB(img, orb, show_image=False)
        if des.any() != None:
            for i in range(des.shape[0]):
                xq.append(utils.convert_intList_to_bit(list(des[i])))
        else:
            return None

    except AttributeError:
        None

    return np.array(xq).astype('float32')


def eval_compute_xq_cloud_with_pca(orb, img_url, pca):
    """compute the query vector of descriptors from the image url"""

    img = utils.url_to_img(img_url)
    xq = []

    try:
        des = utils.compute_ORB(img, orb, show_image=False)
        if des.any() != None:
            for i in range(des.shape[0]):
                xq.append(utils.convert_intList_to_bit(list(des[i])))
        else:
            return None

    except AttributeError:
        None

    xq_tran = pca.transform(xq)
    xq_np = np.array(xq_tran).astype('float32')
    return xq_np
    # xq_np = xq_np.reshape(1,-1)
    # return pca.transform(xq_np)
    # return np.array(pca.transform(xq)).reshape(1,-1).astype('float32')


# Calculates how many times each des_file_ids appears in I, and according to the desired N, it returns the mapping to file_ids (images' names) of the highest N in the list.
def eval_retrieve_top_n(I, n):

    indexes_list = list(I.flatten())
    distinct_indexes = list(set(indexes_list))
    filename_count = dict.fromkeys(distinct_indexes, 0)

    for index in indexes_list:
        filename_count[index] += 1

    top_n = nlargest(n, filename_count, key=filename_count.get)

    num = 1
    top_n_ids = []
    scores = []
    sources = []

    for str_i in top_n:

        top_n_ids.append(str_i)
        scores.append(filename_count[str_i])
        sources.append(ds_source(str_i))

        num += 1

    return top_n_ids, scores, sources


# # Calculates how many times each des_file_ids appears in I, and according to the desired N, it returns the mapping to file_ids (images' names) of the highest N in the list.
# def eval_retrieve_relative_top_n(I, n):
#     datastore_client = datastore.Client('adina-image-analysis')
#     kind = "Evaluation_Flagged_Original"
#
#     indexes_list = list(I.flatten())
#     distinct_indexes = list(set(indexes_list))
#     filename_count = dict.fromkeys(distinct_indexes, 0)
#
#     for index in indexes_list:
#         filename_count[index] += 1
#
#     # normalize filename_count according to the number of descriptors of each index
#     for index in filename_count:
#         key = datastore_client.key(kind, str(index))
#         print(key)
#         image_entity = datastore_client.get(key)
#         print(image_entity)
#         print(image_entity['ORB Descriptors'])
#         desc = json.loads(image_entity['ORB Descriptors'])
#         number_descriotors = len(desc)
#         filename_count[index] = filename_count[index]/number_descriotors
#
#     top_n = nlargest(n, filename_count, key=filename_count.get)
#
#     num = 1
#     top_n_ids = []
#     scores = []
#     sources = []
#
#     for str_i in top_n:
#
#         top_n_ids.append(str_i)
#         scores.append(filename_count[str_i])
#         sources.append(ds_source(str_i))
#
#         num+=1
#
#     return top_n_ids, scores, sources


# used once with 'image_urls.tsv' and stopped manually after 283 images
def add_BreakingNews_dataset(input_file):
    n = 1
    with open(input_file, "r") as urls:
        for line in urls.read().split("\n"):
            line = line.split("\t")
            url = line[1]
            try:
                urlo = urlopen(url)
                urll = image.load_img(urlo)
                imageio.imwrite('222_'+str(n)+'.jpg', np.array(urll))
                n += 1
            except:
                continue
    urls.close()
    return

# used once with 'image_urls.tsv' after the 283-rd image


def add_to_storage_BreakingNews_dataset(input_file):
    storage_client = storage.Client('adina-image-analysis')
    bucket = storage_client.get_bucket('additional_evaluation_images')
    n = 1
    with open(input_file, "r") as urls:
        for line in urls.read().split("\n"):
            if n <= 9239:
                n += 1
                continue

            if n >= 15000:
                return

            if n % 500 == 0:
                print(n)

            line = line.split("\t")
            url = line[1]

            try:
                img = io.BytesIO(requests.get(url).content)
                print(url)
                name = '222_' + str(n) + '.jpg'
                n += 1
                print(name)
                blob = storage.Blob(name, bucket)
                blob.upload_from_file(img)

            except:
                continue
    urls.close()
    return

# used once with 'mini_originals.tsv', 'mini_photoshops.tsv'


def add_p_s_battles(input_file, input_file2):
    path = 'original_p_s'
    path2 = 'photoshops_p_s'

    n = 0

    urls = open(input_file, "r")

    for line in urls.read().split("\n"):
        line = line.split("\t")
        i_d = line[0]
        url = line[1]
        try:
            urlo = urlopen(url)
            urll = image.load_img(urlo)
            n += 1
            name = '555_' + str(n) + '.jpg'
            path_and_name = os.path.join(path, name)
            imageio.imwrite(path_and_name, np.array(urll))
            k = 0
            with open(input_file2, "r") as ps_urls:
                for line2 in ps_urls.read().split("\n"):
                    line2 = line2.split("\t")
                    org = line2[1]
                    url2 = line2[2]
                    if org == i_d:
                        k += 1
                        urlo2 = urlopen(url2)
                        urll2 = image.load_img(urlo2)
                        name2 = '555_' + str(n) + '_' + str(k) + '.jpg'
                        path_and_name2 = os.path.join(path2, name2)
                        imageio.imwrite(path_and_name2, np.array(urll2))
        except:
            continue

    urls.close()
    return


# used once with 'without_begining_originals.tsv', 'without_begining_photoshops.tsv'
def add_to_storage_p_s_battles(input_file, input_file2):
    storage_client = storage.Client('adina-image-analysis')
    path = storage_client.get_bucket('additional_evaluation_images')
    path2 = storage_client.get_bucket('photoshop_battles_manipualtions')

    n = 48

    urls = open(input_file, "r")

    for line in urls.read().split("\n"):

        if n >= 3000:
            return

        line = line.split("\t")
        i_d = line[0]
        url = line[1]
        ext = line[2]

        try:
            img = io.BytesIO(requests.get(url).content)

            n += 1
            name = '555_' + str(n) + '.'

            blob = storage.Blob(name + ext, path)
            blob.upload_from_file(img)

            k = 0
            with open(input_file2, "r") as ps_urls:
                for line2 in ps_urls.read().split("\n"):
                    line2 = line2.split("\t")
                    org = line2[1]
                    url2 = line2[2]
                    ext2 = line2[3]

                    if org == i_d:
                        img2 = io.BytesIO(requests.get(url2).content)
                        # urlo2 = urlopen(url2)
                        # urll2 = image.load_img(urlo2)
                        k += 1
                        name2 = '555_' + str(n) + '_' + str(k) + '.'
                        blob = storage.Blob(name2 + ext2, path2)
                        blob.upload_from_file(img2)
                        # path_and_name2 = os.path.join(path2, name2)
                        # imageio.imwrite(path_and_name2, np.array(urll2))
        except:
            continue

    urls.close()
    return

# used once with quantity=130


def sample_social_media_storage(quantity):
    storage_client = storage.Client('adina-image-analysis')
    bucket = storage_client.get_bucket('adina-images')
    path = storage_client.get_bucket('sampled_social_media_images')

    # p = utils.count_storage('adina-images')

    l = list(range(300000, 400000))

    choseen = random.sample(l, quantity)

    i = 300000
    s = 36443

    for blob in bucket.list_blobs():

        if s >= 65000:
            return

        i += 1

        if i in choseen:

            if not (utils.ext(blob.public_url) == '.jpeg' or utils.ext(blob.public_url) == '.png' or utils.ext(
                    blob.public_url) == '.jpg'):
                continue

            if not (blob.name.startswith('4C') or blob.name.startswith('RE') or blob.name.startswith('R')):
                continue

            try:
                img = io.BytesIO(requests.get(blob.public_url).content)
                name = '333_' + str(s) + '.jpg'
                s += 1
                blob = storage.Blob(name, path)
                blob.upload_from_file(img)

                if s % 5000 == 0:
                    print(s)

            except:
                continue
    return

# a function for the copy_move dataset - rename files.
# for example -  used with s_path = 'cmbExtra2', d_path = 'generated_cm_images/new'


def rename_cm_dataset(s_path, d_path):
    names = {}
    names['barrier'] = '444_1'
    names['beach_wood'] = '444_2'
    names['berries'] = '444_3'
    names['bricks'] = '444_4'
    names['cattle'] = '444_5'
    names['central_park'] = '444_6'
    names['christmas_hedge'] = '444_7'
    names['clean_walls'] = '444_8'
    names['dark_and_bright'] = '444_9'
    names['disconnected_shift'] = '444_10'
    names['egyptian'] = '444_11'
    names['extension'] = '444_12'
    names['fisherman'] = '444_13'
    names['fountain'] = '444_14'
    names['four_babies'] = '444_15'
    names['giraffe'] = '444_16'
    names['horses'] = '444_17'
    names['japan_tower'] = '444_18'
    names['knight_moves'] = '444_19'
    names['kore'] = '444_20'
    names['malawi'] = '444_21'
    names['mask'] = '444_22'
    names['mykene'] = '444_23'
    names['red_tower'] = '444_24'
    names['sailing'] = '444_25'
    names['statue'] = '444_26'
    names['window'] = '444_27'
    names['supermarket'] = '444_28'
    names['stone_ghost'] = '444_29'
    names['ship'] = '444_30'
    names['swan'] = '444_31'
    names['tree'] = '444_32'
    names['wading'] = '444_33'
    names['motorcycle'] = '444_34'
    names['port'] = '444_35'
    names['hedge'] = '444_36'
    names['jellyfish_chaos'] = '444_37'
    names['lone_cat'] = '444_38'
    names['no_beach'] = '444_39'
    names['noise_pattern'] = '444_40'
    names['sails'] = '444_41'
    names['scotland'] = '444_42'
    names['sweets'] = '444_43'
    names['tapestry'] = '444_44'
    names['threehundred'] = '444_45'
    names['white'] = '444_46'
    names['wood_carvings'] = '444_47'
    names['writing_history'] = '444_48'

    for folder in os.listdir(s_path):
        for filename in os.listdir(os.path.join(s_path, folder)):
            if '_gt_' in filename:
                continue
            end_indexes = (filename.find('_copy_') + 5)
            end_parameters = filename[end_indexes:]
            urll = image.load_img(s_path + '/' + folder + '/' + filename)
            for key, value in names.items():
                if utils.del_ext(filename).startswith(key):
                    new_name = value
                    break
            p_and_n = os.path.join(d_path, new_name)
            imageio.imwrite(p_and_n + str(end_parameters), np.array(urll))

    return


def ds_source(filename):
    filename = str(filename)
    if filename.startswith('111'):
        return 'first_draft'
    elif filename.startswith('222'):
        return 'newspapers'
    elif filename.startswith('333'):
        return 'social_media'
    elif filename.startswith('444'):
        return 'copy_move'
    elif filename.startswith('555'):
        return 'photoshop_battles'
