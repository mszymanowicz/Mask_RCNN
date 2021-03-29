import cv2
import sys
#from time import sleep
import numpy as np

import os
import random
import math
import re
import time
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

import widerface

def rmv_imgs_with_many_faces(dataset, max_anns):
    len_annotations = []
    for i, item in enumerate(dataset.image_info):
        len_annotations.append(len(item['annotations']))

    ind_rmv = []
    exceeding_anns = []
    for i, anns in enumerate(len_annotations):
        if anns > max_anns:
            exceeding_anns.append(anns)
            ind_rmv.append(i)

    dataset._image_ids = np.delete(dataset._image_ids, ind_rmv)
    dataset.num_images = len(dataset.image_ids)

def find_bbox_indices_by_size(gt_bbox):
    ind_small=[]
    ind_medium = []
    ind_large = []
    for i, item in enumerate(gt_bbox):
        area = (gt_bbox[i][3]-gt_bbox[i][1]+1)*(gt_bbox[i][2]-gt_bbox[i][0]+1)
        if area < 32**2:
            ind_small.append(i)
        elif area > 32**2 and area < 96**2:
            ind_medium.append(i)
        elif area > 96**2:
            ind_large.append(i)
    return ind_small, ind_medium, ind_large

def compute_batch_metrics(dataset):
    APs50=[]
    APs75=[]
    APs_range=[]
    ARs50=[]
    ARs75=[]
    ARs_range=[]
    APs_range_small=[]
    APs_range_medium=[]
    APs_range_large=[]
    ARs_range_small=[]
    ARs_range_medium=[]
    ARs_range_large=[]
    '''for image_id in image_ids:
        # Load image
        image, image_meta, gt_class_id, gt_bbox =\
            modellib.load_image_gt(dataset, config,
                                   image_id, use_mini_mask=False)'''



    for image_id in dataset.image_ids:
        image = dataset.load_image(image_id)
        '''gt_bbox=[]
        gt_class_id=[]
        for dict in dataset.image_info[image_id]['annotations']:
            gt_bbox.append(dict['bbox'])
            gt_class_id.append(dict['category_id'])
            cv2.rectangle(image, (dict['bbox'][1], dict['bbox'][0]), (dict['bbox'][3], dict['bbox'][2]), (255, 0, 0), 2)

        gt_bbox = np.array(gt_bbox)
        print('gt_bbox:,', gt_bbox)
        #print('gt_bbox.shape:', gt_bbox.shape)
        gt_class_id = np.array(gt_class_id)
        #zerod = np.zeros(len(gt_class_id))
        #gt_class_id = np.vstack((gt_class_id, zerod))'''

        gt_bbox, gt_class_id = dataset.load_mask(image_id)
        '''print('gt_bbox:,', gt_bbox)
        if len(gt_bbox[0]) == 4:
            for i in range(len(gt_bbox)):
                cv2.rectangle(image, (gt_bbox[i][1], gt_bbox[i][0]), (gt_bbox[i][3], gt_bbox[i][2]), (255, 0, 0), 2)'''

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.03,
            minNeighbors=6,
            minSize=(24, 24)
        )

        # Draw a rectangle around the faces
        pred_bbox=[]
        pred_scores=[]
        pred_class_id=[]
        count = 0
        for (x, y, w, h) in faces:
            #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            pred_bbox.append([y, x, y+h, x+w])
            pred_scores.append(1.)
            pred_class_id.append(1)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            count += 1
        if count == 0:
            pred_bbox.append([])
            pred_scores.append(1.)
            pred_class_id.append(1)

        pred_bbox = np.array(pred_bbox)
        pred_scores = np.array(pred_scores)
        pred_class_id = np.array(pred_class_id)
        #zerod = np.zeros(len(pred_scores))
        #pred_scores = np.vstack((pred_scores, zerod))
        #pred_class_id = np.vstack((pred_class_id, zerod))
        #print(pred_bbox)
        r = {'rois': pred_bbox, 'class_ids': pred_class_id, 'scores': pred_scores}
        '''print("r['rois']:", r['rois'])
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Video', img)
        cv2.waitKey(0)'''
     # Run object detection
        #results = model.detect([image], verbose=0)
        # Compute AP
        #r = results[0]
        if pred_bbox.shape != (1,0):
            AP50, precisions50, recalls50, overlaps =\
                utils.compute_ap(gt_bbox, gt_class_id,
                                  r['rois'], r['class_ids'], r['scores'], config=config, iou_threshold=0.5)
            #print('AP50:', AP50)
            #print('pred_bbox.shape:', pred_bbox.shape)
            AP75, precisions75, recalls75, overlaps =\
                utils.compute_ap(gt_bbox, gt_class_id,
                                  r['rois'], r['class_ids'], r['scores'], config=config, iou_threshold=0.75)
            AP_range =\
                utils.compute_ap_range(gt_bbox, gt_class_id,
                                  r['rois'], r['class_ids'], r['scores'], config=config)

            AR50, positive_ids50 =\
                utils.compute_recall(r['rois'], gt_bbox, iou=0.5)

            AR75, positive_ids75 =\
                utils.compute_recall(r['rois'], gt_bbox, iou=0.75)

            AR_range =\
                utils.compute_ar_range(r['rois'], gt_bbox)
        else:
            AP50 = 0.; AP75 = 0.; AP_range = 0.; AR50 = 0.; AR75 = 0.; AR_range = 0.;

        #AR50 = recalls50[-2]
        #AR75 = recalls75[-2]
        #AR_range = recalls_range[-2]
        APs50.append(AP50)
        APs75.append(AP75)
        APs_range.append(AP_range)
        ARs50.append(AR50)
        ARs75.append(AR75)
        ARs_range.append(AR_range)


        gt_ind_small, gt_ind_medium, gt_ind_large = find_bbox_indices_by_size(gt_bbox)



        gt_bbox_small = gt_bbox[gt_ind_small,:]
        gt_bbox_medium = gt_bbox[gt_ind_medium,:]
        gt_bbox_large = gt_bbox[gt_ind_large,:]

        gt_class_id_small = gt_class_id[gt_ind_small]
        gt_class_id_medium = gt_class_id[gt_ind_medium]
        gt_class_id_large = gt_class_id[gt_ind_large]

        if pred_bbox.shape != (1,0):
            pred_ind_small, pred_ind_medium, pred_ind_large = find_bbox_indices_by_size(r['rois'])

            pred_bbox_small = r['rois'][pred_ind_small,:]
            pred_bbox_medium = r['rois'][pred_ind_medium,:]
            pred_bbox_large = r['rois'][pred_ind_large,:]

            pred_class_id_small = r['class_ids'][pred_ind_small]
            pred_class_id_medium = r['class_ids'][pred_ind_medium]
            pred_class_id_large = r['class_ids'][pred_ind_large]

            pred_score_small = r['scores'][pred_ind_small]
            pred_score_medium = r['scores'][pred_ind_medium]
            pred_score_large = r['scores'][pred_ind_large]



        if gt_bbox_small.size!=0:
            if pred_bbox.shape != (1,0):
                AP_range_small =\
                    utils.compute_ap_range(gt_bbox_small, gt_class_id_small,
                                      pred_bbox_small, pred_class_id_small, pred_score_small, config=config)

                AR_range_small =\
                    utils.compute_ar_range(pred_bbox_small, gt_bbox_small)
            else:
                AP_range_small = 0.
                AR_range_small = 0.

            APs_range_small.append(AP_range_small)
            ARs_range_small.append(AR_range_small)


        if gt_bbox_medium.size!=0:
            if pred_bbox.shape != (1,0):
                AP_range_medium =\
                    utils.compute_ap_range(gt_bbox_medium, gt_class_id_medium,
                                      pred_bbox_medium, pred_class_id_medium, pred_score_medium, config=config)

                AR_range_medium =\
                    utils.compute_ar_range(pred_bbox_medium, gt_bbox_medium)
            else:
                AP_range_medium = 0.
                AR_range_medium = 0.

            APs_range_medium.append(AP_range_medium)
            ARs_range_medium.append(AR_range_medium)


        if gt_bbox_large.size!=0:
            if pred_bbox.shape != (1,0):
                AP_range_large =\
                utils.compute_ap_range(gt_bbox_large, gt_class_id_large,
                                  pred_bbox_large, pred_class_id_large, pred_score_large, config=config)

                AR_range_large =\
                utils.compute_ar_range(pred_bbox_large, gt_bbox_large)
            else:
                AP_range_large = 0.
                AR_range_large = 0.

            APs_range_large.append(AP_range_large)
            ARs_range_large.append(AR_range_large)

    return APs50, APs75, APs_range, ARs50, ARs75, ARs_range, APs_range_small, APs_range_medium,\
        APs_range_large, ARs_range_small, ARs_range_medium, ARs_range_large




config = widerface.CocoConfig()
WIDERFACE_DIR = "../../datasets/widerface"  # TODO: enter value here

# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

#TEST_MODE = "inference"


dataset = widerface.CocoDataset()
dataset.load_coco(WIDERFACE_DIR, "val")

# Must call before using the dataset
dataset.prepare()



rmv_imgs_with_many_faces(dataset, 1)



cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)






#all_image_ids = dataset.image_ids
APs50, APs75, APs_range, ARs50, ARs75, ARs_range, APs_range_small, APs_range_medium,\
    APs_range_large, ARs_range_small, ARs_range_medium, ARs_range_large = compute_batch_metrics(dataset)


print('ALL images: maxDet=1')
print('mAP @IoU=50: ',np.mean(APs50))
print('mAP @IoU=75: ',np.mean(APs75))
print('mAP @IoU=50:5:95: ',np.mean(APs_range))
print('mAR @IoU=50: ',np.mean(ARs50))
print('mAR @IoU=75: ',np.mean(ARs75))
print('mAR @IoU=50:5:95: ',np.mean(ARs_range))
print('mAP @IoU=50:5:95: area=small: ',np.mean(APs_range_small))
print('mAP @IoU=50:5:95: area=medium: ',np.mean(APs_range_medium))
print('mAP @IoU=50:5:95: area=large:',np.mean(APs_range_large))
print('mAR @IoU=50:5:95: area=small: ',np.mean(ARs_range_small))
print('mAR @IoU=50:5:95: area=medium: ',np.mean(ARs_range_medium))
print('mAR @IoU=50:5:95: area=large:',np.mean(ARs_range_large))
