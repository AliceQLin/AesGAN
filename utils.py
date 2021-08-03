import os
import errno
import numpy as np
import scipy
import scipy.misc
import json
import cv2
import tensorflow as tf
from skimage import io, img_as_float

def random_crop_and_pad_image(image, crop_h, crop_w, ignore_label=255):
    image_shape = tf.shape(image)
    pad = tf.image.pad_to_bounding_box(image, 0, 0, tf.maximum(crop_h, image_shape[0]), tf.maximum(crop_w, image_shape[1]))    
    last_image_dim = tf.shape(image)[-1]
    img_crop = tf.random_crop(pad, [crop_h,crop_w,3])
    img_crop.set_shape((crop_h, crop_w, 3))
    return img_crop

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def get_image(image_path , image_size, is_crop= True, resize_w= 64, is_grayscale= False, is_test=False):
    return transform(imread(image_path , is_grayscale), image_size, is_crop , resize_w, is_test=is_test)

def transform(image, npx = 64 , is_crop=False, resize_w=64, is_test=False):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image , npx , resize_w=resize_w, is_test=is_test)
    else:
        cropped_image = image
        cropped_image = scipy.misc.imresize(cropped_image ,
                            [resize_w , resize_w])
    return np.array(cropped_image)/127.5 - 1

def center_crop(x, crop_h , crop_w=None, resize_w=64, is_test=False):

    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))

    if not is_test:
        rate = np.random.uniform(0, 1, size=1)
        if rate < 0.5:
            x = np.fliplr(x)
    # return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
    #                            [resize_w, resize_w])
    return scipy.misc.imresize(x[20:218 - 20, 0: 178], [resize_w, resize_w])

def save_images(images, size, image_path, is_ouput=False):
    return imsave(inverse_transform(images, is_ouput), size, image_path)

def imread(path, is_grayscale=False):

    if (is_grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def merge(images, size):

    if size[0] + size[1] == 2:
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1], 3))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w: i * w + w, :] = image

    else:
        h, w = images.shape[1], images.shape[2]
        img = np.zeros(((int)(h * size[0]), (int)(w * size[1]), 3))
        for idx, image in enumerate(images):
            i = (int)(idx % size[1])
            j = (int)(idx // size[1])
            img[j * h:j * h + h, i * w: i * w + w, :] = image

    return img

def inverse_transform(image, is_ouput=False):

    if is_ouput == True:
        print (image[0])
    result = ((image + 1) * 127.5).astype(np.uint8)
    if is_ouput == True:
        print (result)
    return result

def eye_mask(eye_left, eye_right, shape):
    
    mask = np.zeros(shape=shape)
    #画左眼椭圆
    cv2.ellipse(mask, (int(eye_left['x']), int(eye_left['y'])), (25, 15), 0, 0, 360, (1.0, 1.0, 1.0), -1)
    #画右眼椭圆
    cv2.ellipse(mask, (int(eye_right['x']), int(eye_right['y'])), (25, 15), 0, 0, 360, (1.0, 1.0, 1.0), -1)
    
    return np.array(mask)


log_interval = 10

def read_image_list_for_Eyes(category):

    json_cat = category + "/data.json"
    with open(json_cat, 'r') as f:
        data = json.load(f)

    all_iden_info = []
    all_ref_info = []

    test_all_iden_info = []
    test_all_ref_info = []

    #ssim & eyescore-based
    with tf.Session() as sess:
        
        image1 = tf.placeholder(tf.float32, [None, None, None, 3])
        image2 = tf.placeholder(tf.float32, [None, None, None, 3])             
        out_ssim = tf.image.ssim(image1, image2, 255)
        
        #c: id
        #k: name of identity
        #v: details.
        for c, (k, v) in enumerate(data.items()):
    
            identity_info = []
    
            is_close = False
            is_close_id = 0
    
            if c % log_interval == 0:
                print('Processed {}/{}'.format(c, len(data)))
    
            if len(v) < 2:
                continue
    
            for i in range(len(v)):
    
                if is_close or v[i]['opened'] is None or v[i]['opened'] < 0.60:
                    is_close = True
                    is_close_id = i
    
                str_info = str(v[i]['filename']) + "_"
    
                if 'eye_left' in v[i] and v[i]['eye_left'] != None:
                    str_info += str(v[i]['eye_left']['y']) + "_"
                    str_info += str(v[i]['eye_left']['x']) + "_"
                else:
                    str_info += str(0) + "_"
                    str_info += str(0) + "_"
    
                if 'box_left' in v[i] and v[i]['box_left'] != None:
                    str_info += str(v[i]['box_left']['h']) + "_"
                    str_info += str(v[i]['box_left']['w']) + "_"
                else:
                    str_info += str(0) + "_"
                    str_info += str(0) + "_"
    
                if 'eye_right' in v[i] and v[i]['eye_right'] != None:
                    str_info += str(v[i]['eye_right']['y']) + "_"
                    str_info += str(v[i]['eye_right']['x']) + "_"
                else:
                    str_info += str(0) + "_"
                    str_info += str(0) + "_"
    
                if 'box_right' in v[i] and v[i]['box_right'] != None:
                    str_info += str(v[i]['box_right']['h']) + "_"
                    str_info += str(v[i]['box_right']['w'])
                else:
                    str_info += str(0) + "_"
                    str_info += str(0)
    
                identity_info.append(str_info)
    
            if is_close == False:
    
                for j in range(len(v)):
    
                    first_n = np.random.randint(0, len(v), size=1)[0]
    
                    middle_value = identity_info[first_n]                  
                    
                    if first_n == 0:
                        ref_picn = 1
                    else:
                        ref_picn = 0
                        
                    tmp_ssim = 0
                    flg = ref_picn
                    
                    img1 = cv2.imread(category +'/' + v[first_n]['filename'])
                    img1 = np.float32(np.expand_dims(img1, 0))
                    shape = img1.shape
                    
                    if v[first_n]['eye_left'] is None or v[first_n]['eye_right'] is None:
                        continue
                    
                    img1_mask = eye_mask(v[first_n]['eye_left'], v[first_n]['eye_right'], shape)
                    img1 = img1 * (1 - img1_mask)
                    
                    eyescore = 0
                    
                    for i in range(len(v)):
                        if i == first_n or v[i]['opened'] is None or v[i]['opened'] < 0.60 :
                            continue
                            
                        else:

                            img2 = cv2.imread(category +'/' + v[i]['filename'])
                            img2 = np.float32(np.expand_dims(img2, 0))          
                            img2 = img2 * (1 - img1_mask)
               
                            out_res = sess.run(out_ssim, feed_dict={image1: img1, image2: img2})
                                
                            if out_res[0] > tmp_ssim and v[i]['eyescore'] >= eyescore:
                                tmp_ssim = out_res[0]
                                ref_picn = i
                                eyescore = v[i]['eyescore']
                                
                    if flg == ref_picn:
                        
                        for i in range(len(v)):
                            
                            if i == first_n or v[i]['opened'] is None or v[i]['opened'] < 0.60 :
                                continue
                                
                            else:
                                img2 = cv2.imread(category +'/' + v[i]['filename'])
                                img2 = np.float32(np.expand_dims(img2, 0))          
                                img2 = img2 * (1 - img1_mask)
                   
                                out_res = sess.run(out_ssim, feed_dict={image1: img1, image2: img2})
                                    
                                if out_res[0] > tmp_ssim:
                                    tmp_ssim = out_res[0]
                                    ref_picn = i
                                    
                    second_n = ref_picn
                    all_iden_info.append(middle_value)
                    all_ref_info.append(identity_info[second_n])
    
            else:
       
                middle_value = identity_info[is_close_id]              
                
                if is_close_id == 0:
                    ref_picn = 1
                else:
                    ref_picn = 0
                    
                tmp_ssim = 0
                flg = ref_picn
                
                img1 = cv2.imread(category +'/' + v[is_close_id]['filename'])
                img1 = np.float32(np.expand_dims(img1, 0))
                shape = img1.shape
                
                if v[is_close_id]['eye_left'] is None or v[is_close_id]['eye_right'] is None:
                    continue
                
                img1_mask = eye_mask(v[is_close_id]['eye_left'], v[is_close_id]['eye_right'], shape)
                img1 = img1 * (1 - img1_mask)
                
                eyescore = 0
                
                for i in range(len(v)):
                    if i == is_close_id or v[i]['opened'] is None or v[i]['opened'] < 0.60 :
                        continue
                        
                    else:
                        img2 = cv2.imread(category +'/' + v[i]['filename'])
                        img2 = np.float32(np.expand_dims(img2, 0))          
                        img2 = img2 * (1 - img1_mask)              
                   
                        out_res = sess.run(out_ssim, feed_dict={image1: img1, image2: img2})
                            
                        if out_res[0] > tmp_ssim and v[i]['eyescore'] >= eyescore:
                            tmp_ssim = out_res[0]
                            ref_picn = i
                            eyescore = v[i]['eyescore']
                                            
                if flg == ref_picn:
                    
                    for i in range(len(v)):
                        
                        if i == is_close_id or v[i]['opened'] is None or v[i]['opened'] < 0.60 :
                            continue
                            
                        else:
                            img2 = cv2.imread(category +'/' + v[i]['filename'])
                            img2 = np.float32(np.expand_dims(img2, 0))          
                            img2 = img2 * (1 - img1_mask)
                            
                            out_res = sess.run(out_ssim, feed_dict={image1: img1, image2: img2})
                                
                            if out_res[0] > tmp_ssim:
                                tmp_ssim = out_res[0]
                                ref_picn = i
                                
                second_n = ref_picn
                test_all_iden_info.append(middle_value)
                test_all_ref_info.append(identity_info[second_n])
                
                if is_close_id == 0:
                    ref_picn = 1
                else:
                    ref_picn = 0

                tmp_ssim = 0
                flg = ref_picn
                
                img1 = cv2.imread(category +'/' + v[is_close_id]['filename'])
                img1 = np.float32(np.expand_dims(img1, 0))
                shape = img1.shape
                
                if v[is_close_id]['eye_left'] is None or v[is_close_id]['eye_right'] is None:
                    continue
                
                img1_mask = eye_mask(v[is_close_id]['eye_left'], v[is_close_id]['eye_right'], shape)
                img1 = img1 * (1 - img1_mask)
                
                eyescore = 0
                
                for i in range(len(v)):
                    if i == is_close_id or i == second_n or v[i]['opened'] is None or v[i]['opened'] < 0.60 :
                        continue
                        
                    else:
                        img2 = cv2.imread(category +'/' + v[i]['filename'])
                        img2 = np.float32(np.expand_dims(img2, 0))          
                        img2 = img2 * (1 - img1_mask)
               
                        out_res = sess.run(out_ssim, feed_dict={image1: img1, image2: img2})
                            
                        if out_res[0] > tmp_ssim and v[i]['eyescore'] >= eyescore:
                            tmp_ssim = out_res[0]
                            ref_picn = i
                            eyescore = v[i]['eyescore']
                            
                if flg == ref_picn:
                    
                    for i in range(len(v)):
                        
                        if i == is_close_id or i == second_n or v[i]['opened'] is None or v[i]['opened'] < 0.60 :
                            continue
                            
                        else:
                            img2 = cv2.imread(category +'/' + v[i]['filename'])
                            img2 = np.float32(np.expand_dims(img2, 0))          
                            img2 = img2 * (1 - img1_mask)
                            
                            out_res = sess.run(out_ssim, feed_dict={image1: img1, image2: img2})
                                
                            if out_res[0] > tmp_ssim:
                                tmp_ssim = out_res[0]
                                ref_picn = i
                                
                second_n = ref_picn
                test_all_iden_info.append(middle_value)
                test_all_ref_info.append(identity_info[second_n])

    assert len(all_iden_info) == len(all_ref_info)
    assert len(test_all_iden_info) == len(test_all_ref_info)

    print ("train_data", len(all_iden_info))
    print ("test_data", len(test_all_iden_info))
    
    if not os.path.exists('./json/'):
        os.makedirs('./json/')
    
    with open('./json/images_list.json', 'w') as f:
        json.dump(all_iden_info, f)
    with open('./json/images_ref_list.json', 'w') as f:
        json.dump(all_ref_info, f)
    with open('./json/test_images_list.json', 'w') as f:
        json.dump(test_all_iden_info, f)
    with open('./json/test_images_ref_list.json', 'w') as f:
        json.dump(test_all_ref_info, f)

    return all_iden_info, all_ref_info, test_all_iden_info, test_all_ref_info

class Eyes(object):

    def __init__(self, image_path):
        self.dataname = "Eyes"
        self.image_size = 256
        self.channel = 3
        self.image_path = image_path
        self.image_label_path = './gt_labels'
        self.dims = self.image_size*self.image_size
        self.shape = [self.image_size, self.image_size, self.channel]
        self.train_images_name, self.train_eye_pos_name, self.train_ref_images_name, self.train_ref_pos_name, self.train_labels_name, \
            self.test_images_name, self.test_eye_pos_name, self.test_ref_images_name, self.test_ref_pos_name = self.load_Eyes(image_path)

    def load_Eyes(self, image_path):

        if os.path.exists('./json/test_images_ref_list.json'):
            with open('./json/images_list.json', 'r') as f:
                images_list = json.load(f)
            with open('./json/images_ref_list.json', 'r') as f:
                images_ref_list = json.load(f)
            with open('./json/test_images_list.json', 'r') as f:
                test_images_list = json.load(f)
            with open('./json/test_images_ref_list.json', 'r') as f:
                test_images_ref_list = json.load(f)
        else:
            images_list, images_ref_list, test_images_list, test_images_ref_list = read_image_list_for_Eyes(image_path)
        
        train_images_name = []
        train_eye_pos_name = []
        train_ref_images_name = []
        train_ref_pos_name = []
        train_labels_name = []

        test_images_name = []
        test_eye_pos_name = []
        test_ref_images_name = []
        test_ref_pos_name = []

        #train
        for images_info_str in images_list:

            eye_pos = []
            image_name, left_eye_x, left_eye_y, left_eye_h, left_eye_w, right_eye_x, right_eye_y,\
                right_eye_h, right_eye_w = images_info_str.split('_', 9)
            eye_pos.append((int(left_eye_x), int(left_eye_y), int(left_eye_h), int(left_eye_w), int(right_eye_x),
                            int(right_eye_y), int(right_eye_h), int(right_eye_w)))
            labels_name = os.path.join(self.image_label_path, image_name[:-4]+'.png')
            image_name = os.path.join(self.image_path, image_name)
            
            train_images_name.append(image_name)
            train_eye_pos_name.append(eye_pos)
            train_labels_name.append(labels_name)

        for images_info_str in images_ref_list:

            eye_pos = []
            image_name, left_eye_x, left_eye_y, left_eye_h, left_eye_w, right_eye_x, right_eye_y, \
                right_eye_h, right_eye_w = images_info_str.split('_', 9)

            eye_pos.append((int(left_eye_x), int(left_eye_y), int(left_eye_h), int(left_eye_w), int(right_eye_x),
                            int(right_eye_y), int(right_eye_h), int(right_eye_w)))
            
            image_name = os.path.join(self.image_path, image_name)
            
            train_ref_images_name.append(image_name)
            train_ref_pos_name.append(eye_pos)

        for images_info_str in test_images_list:

            eye_pos = []
            image_name, left_eye_x, left_eye_y, left_eye_h, left_eye_w, right_eye_x, right_eye_y, \
            right_eye_h, right_eye_w = images_info_str.split('_', 9)
            eye_pos.append(
                (int(left_eye_x), int(left_eye_y), int(left_eye_h), int(left_eye_w), int(right_eye_x),
                 int(right_eye_y), int(right_eye_h), int(right_eye_w)))
            image_name = os.path.join(self.image_path, image_name)

            test_images_name.append(image_name)
            test_eye_pos_name.append(eye_pos)

        for images_info_str in test_images_ref_list:

            eye_pos = []
            image_name, left_eye_x, left_eye_y, left_eye_h, left_eye_w, right_eye_x, right_eye_y, \
            right_eye_h, right_eye_w = images_info_str.split('_', 9)
            eye_pos.append(
                (int(left_eye_x), int(left_eye_y), int(left_eye_h), int(left_eye_w), int(right_eye_x),
                 int(right_eye_y), int(right_eye_h), int(right_eye_w)))
            image_name = os.path.join(self.image_path, image_name)
            
            test_ref_images_name.append(image_name)
            test_ref_pos_name.append(eye_pos)

        assert len(train_images_name) == len(train_eye_pos_name) == len(train_ref_images_name) == len(train_ref_pos_name)
        assert len(test_images_name) == len(test_eye_pos_name) == len(test_ref_images_name) == len(test_ref_pos_name)

        return train_images_name, train_eye_pos_name, train_ref_images_name, train_ref_pos_name, train_labels_name, \
               test_images_name, test_eye_pos_name, test_ref_images_name, test_ref_pos_name

    def getShapeForData(self, filenames, is_test=False):

        array = [get_image(batch_file, 108, is_crop=False, resize_w=256,
                           is_grayscale=False, is_test=is_test) for batch_file in filenames]
        sample_images = np.array(array)
        return sample_images
    
    def read_labels(self, filenames):
        flg = 0
        for batch_file in filenames:
            blur = scipy.misc.imread(batch_file)
            blurPad = np.expand_dims(blur, 0)
            if flg == 0:
                Pad = blurPad
                flg = 1
            else:
                Pad = np.concatenate([Pad, blurPad], axis=0)
    
        return Pad

    def getNextBatch(self, batch_num=0, batch_size=64, is_shuffle=True):
        
        ro_num = len(self.train_images_name) // batch_size
        if batch_num % ro_num == 0 and is_shuffle:
            
            print("**********Has shuffled!*********")
            
            length = len(self.train_images_name)
            perm = np.arange(length)
            np.random.shuffle(perm)

            self.train_images_name = np.array(self.train_images_name)
            self.train_images_name = self.train_images_name[perm]

            self.train_eye_pos_name = np.array(self.train_eye_pos_name)
            self.train_eye_pos_name = self.train_eye_pos_name[perm]

            self.train_ref_images_name = np.array(self.train_ref_images_name)
            self.train_ref_images_name = self.train_ref_images_name[perm]

            self.train_ref_pos_name = np.array(self.train_ref_pos_name)
            self.train_ref_pos_name = self.train_ref_pos_name[perm]
            
            self.train_labels_name = np.array(self.train_labels_name)
            self.train_labels_name = self.train_labels_name[perm]
            

        return self.train_images_name[(int)(batch_num % ro_num) * batch_size: (int)(batch_num % ro_num + 1) * batch_size], \
               self.train_eye_pos_name[(int)(batch_num % ro_num) * batch_size: (int)(batch_num % ro_num + 1) * batch_size], \
               self.train_ref_images_name[(int)(batch_num % ro_num) * batch_size: (int)(batch_num % ro_num + 1) * batch_size], \
                self.train_ref_pos_name[(int)(batch_num % ro_num) * batch_size: (int)(batch_num % ro_num + 1) * batch_size], \
                self.train_labels_name[(int)(batch_num % ro_num) * batch_size: (int)(batch_num % ro_num + 1) * batch_size]
                

    def getTestNextBatch(self, batch_num=0, batch_size=64, is_shuffle=True):

        ro_num = len(self.test_images_name) // batch_size
        if batch_num == 0 and is_shuffle:
            
            print("**********Has shuffled!*********")

            length = len(self.test_images_name)
            perm = np.arange(length)
            np.random.shuffle(perm)

            self.test_images_name = np.array(self.test_images_name)
            self.test_images_name = self.test_images_name[perm]

            self.test_eye_pos_name = np.array(self.test_eye_pos_name)
            self.test_eye_pos_name = self.test_eye_pos_name[perm]

            self.test_ref_images_name = np.array(self.test_ref_images_name)
            self.test_ref_images_name = self.test_ref_images_name[perm]

            self.test_ref_pos_name = np.array(self.test_ref_pos_name)
            self.test_ref_pos_name = self.test_ref_pos_name[perm]
            

        return self.test_images_name[(int)(batch_num % ro_num) * batch_size: (int)(batch_num % ro_num + 1) * batch_size], \
               self.test_eye_pos_name[(int)(batch_num % ro_num) * batch_size: (int)(batch_num % ro_num + 1) * batch_size], \
               self.test_ref_images_name[(int)(batch_num % ro_num) * batch_size: (int)(batch_num % ro_num + 1) * batch_size], \
               self.test_ref_pos_name[(int)(batch_num % ro_num) * batch_size: (int)(batch_num % ro_num + 1) * batch_size]
               