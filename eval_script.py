#coding=utf-8
#Author: stephon
#Time:   2018.12.06

import os
from shapely.geometry import Polygon
import numpy as np
import glob
import copy
import cv2

DRAW_DEBUG=False

def area(bbox):
    """
    get area
    """
    return bbox[2] * 1.0 * bbox[3]          # width * height

def loadAnnotation(folder_p):
    data = []
    imgs = glob.glob(folder_p + "/*.jpg")   # all images

    for img_name in imgs:
        img_name = os.path.abspath(img_name)

        m_id = os.path.basename(img_name.split(".")[0])
        txt_path = img_name.split(".")[0]+".txt"

        img_info={"m_id": m_id}

        with open(txt_path, "r") as f:
            lines=f.readlines()
            anns_info=[]
            for index,line in enumerate(lines):
                line=line.strip()
                arr=line.split(",")[:8]
                arr = map(int, arr)
                anns_info.append({"b_id":str(index),"bbox":arr})
            img_info["anns"]=anns_info

        data.append(img_info)

    return data

def intersect(bboxA, bboxB):
	"""Return a new bounding box that contains the intersection of
	'self' and 'other', or None if there is no intersection
	"""
	new_top = max(bboxA[1], bboxB[1])
	new_left = max(bboxA[0], bboxB[0])
	new_right = min(bboxA[0] + bboxA[2], bboxB[0] + bboxB[2])
	new_bottom = min(bboxA[1] + bboxA[3], bboxB[1] + bboxB[3])

	if new_top < new_bottom and new_left < new_right:
		return [new_left, new_top, new_right - new_left, new_bottom - new_top]

	return None

def iou_score(bboxA, bboxB):
    """Returns the Intersection-over-Union score, defined as the area of
	the intersection divided by the intersection over the union of
	the two bounding boxes. This measure is symmetric.
	"""

    if intersect(bboxA, bboxB):
        intersection_area = area(intersect(bboxA, bboxB))
    else:
        intersection_area = 0
    union_area = area(bboxA) + area(bboxB) - intersection_area
    if union_area > 0:
        return float(intersection_area) / float(union_area)
    else:
        return 0

def intersection(g, p):

    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter/union

def drawDebug(img_path, gt_boxes, eval_boxes):
    if DRAW_DEBUG==False:
        return

    img=cv2.imread(img_path)

    min_len=len(gt_boxes) if len(gt_boxes)<len(eval_boxes) else len(eval_boxes)

    print(min_len)
    for i in range(min_len):


        gt_box=gt_boxes[i]["bbox"]
        eval_box = eval_boxes[i]["bbox"]

        gt_pts = np.array(gt_box, np.int32)
        gt_pts = gt_pts.reshape((-1, 1, 2))

        eval_pts = np.array(eval_box, np.int32)
        eval_pts = eval_pts.reshape((-1, 1, 2))

        cv2.polylines(img, [gt_pts], True, (255, 0, 0))
        cv2.polylines(img, [eval_pts], True, (0, 255, 0))

    cv2.imshow("img",img)
    cv2.waitKey(0)


# Compute detections
# Three values: tp, fn, fp;
def getDetections(groundtruth_param, evaluation_param, img_folder_path, debug_index=0, detection_threshold=0.5):
    """
    A box is a match iff the intersection of union score is >= 0.5.
    Params
    ------
    Input dicts have the format of annotation dictionaries
    """
    groundtruth=copy.deepcopy(groundtruth_param)
    evaluation = copy.deepcopy(evaluation_param)

    # parameters
    detectRes = {}

    # results are lists of dicts {gt_id: xxx, eval_id: yyy}
    detectRes['true_positives'] = 0
    detectRes['false_negatives'] = 0
    detectRes['false_positives'] = 0

    gt_mids = []
    for obj in groundtruth:
        gt_mids.append(obj["m_id"])

    eval_mids = []
    for obj in evaluation:
        eval_mids.append(obj["m_id"])

    imgIds = gt_mids
    imgIds = imgIds if len(imgIds) > 0 else list(set(gt_mids).intersection(set(eval_mids)))

    for m_id in imgIds:
        oneImg={}
        oneImg['true_positives'] = []
        oneImg['false_negatives'] = []
        oneImg['false_positives'] = []

        gt_boxes=[]
        eval_boxes=[]
        img_path = os.path.join(img_folder_path, m_id+".jpg")
        for obj in groundtruth:             #find all boxes belong to this image
            if obj["m_id"]==m_id:
                gt_boxes=obj['anns']
                break

        for obj in evaluation:              #find all boxes belong to this image
            if obj["m_id"]==m_id:
                eval_boxes=obj['anns']
                break
        #print(m_id,len(gt_boxes),len(eval_boxes))

        drawDebug(img_path, gt_boxes, eval_boxes)
        for gt_box in gt_boxes:
            max_iou = 0.0
            match = None

            for i, eval_box in enumerate(eval_boxes):
                iou = intersection(np.asarray(gt_box['bbox']), np.asarray(eval_box['bbox']))

                if iou >= detection_threshold and iou > max_iou:
                    max_iou = iou
                    match = eval_box["b_id"]
                    match_index=i

            if match is not None:               #calculate tpr
                oneImg['true_positives'].append({'gt_id': gt_box['b_id'], 'eval_id': match})
                del(eval_boxes[match_index])
            else:                               #calculate fnr
                oneImg['false_negatives'].append({'gt_id': gt_box['b_id']})

        if len(eval_boxes) > 0:                 #calculate fpr
            oneImg['false_positives'].extend([{'eval_id': eval_box['b_id']} for eval_box in eval_boxes])


        #print(len(oneImg['true_positives']),len(oneImg['false_negatives']),len(oneImg['false_positives']))
        detectRes['true_positives']+=len(oneImg['true_positives'])
        detectRes['false_negatives']+=len(oneImg['false_negatives'])
        detectRes['false_positives']+=len(oneImg['false_positives'])

    print(detectRes, detection_threshold)
    return detectRes

if __name__ == '__main__':
    #load ground truth
    #load evaluation
    
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>first: do inference<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    os.system("python eval.py")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>first: do inference DONE!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    from eval import FLAGS

    img_gt_path =   "./train_lb/Indonesia_idcard2_3/ann3_img_label"
    assert(FLAGS.txt_output_dir == FLAGS.image_output_dir)
    #img_test_path = "./test_output/yinni_idcard/ann1"
    img_test_path =   FLAGS.txt_output_dir

    data_eval = loadAnnotation(img_test_path)
    data_gt   = loadAnnotation(img_gt_path)
    print("result num: {}".format(len(data_eval)))
    print("truth num: {}".format(len(data_gt)))
    
    #m_id: 1820_idFront
    #anns: []: each one is a dict; {bbox and b_id}

    groundtruth = data_gt
    evaluation  = data_eval

    import random
    nlist = range(1, 8)

    for i in range(len(nlist)):
        nlist[i] = float(nlist[i]) * 0.1
    print(nlist)

    #get result
    result = []
    for index, val in enumerate(nlist):
        thresh = val

        #get statistic result; get three 
        res = getDetections(groundtruth, evaluation, img_gt_path, index, detection_threshold = thresh)   #for different thresh, calculate its fpr, tpr, fnr

        #print(thresh,res)
        #for positive truths, it can separate into true_positive and false negative
        #for positive result: it can separate into true_positive and false positive
        
        found = res['true_positives']
        n_found = res['false_negatives']    
        fp = res['false_positives']   

        #if (len(inter(found + n_found, leg_mp))) > 0:
        t_recall = 100 * found * 1.0 / (found + n_found)    
        # t_recall = "%.1f" % (t_recall)
        #print('total recall: ', t_recall)

        t_precision = 100 * found * 1.0 /(found + fp)   
        #precision = "%.2f" % (t_precision)
        #print('total precision: ', t_precision)

        f_score = "%.2f" % (2 * t_recall * t_precision / (t_recall + t_precision)) if (t_recall + t_precision) > 0  else 0
        #print('f-score localization: ', f_score)

        result.append([t_recall, t_precision, f_score, thresh])   #recall, precision

    for item in result:
        print(item)

#综合来看, 选择0.8, 0.9作为重合度指标, 下降会很大.
#建议选择0.5, 0.6, 0.7;

#ann3:  97.56% (200个, 作为测试集合)
#ann2:  97.72% 
#ann1:  97.04% 

#record: 168389 (依然是200个训练样本, 收敛状态下, 可以达到多大的F值)
#ann3:  97.56% (200个, 作为测试集合)
#ann2:  97.72% 
#ann1:  97.04% 
