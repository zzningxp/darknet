# usage: python scripts/cars_eval.py <datacfg> <results_dir> <result_prefix>(optional)
import os
import numpy as np
import glob
import sys
import matplotlib.pyplot as plt

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def evaluate(class_recs, npos, results, ovthresh=0.5, use_07_metric=False):
    # split results
    splitlines = [x.strip().split(' ') for x in results]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    # image_ids = [image_ids[x] for x in sorted_ind if confidence[x] > 0.5]
    image_ids = [image_ids[x] for x in sorted_ind]
    print("# detection: %d" % len(image_ids))

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap

if __name__ == "__main__":
    workspace = os.getcwd() + '/'
    datacfg = workspace + sys.argv[1]
    resdir = workspace + sys.argv[2]
    prefix = 'comp4_det_test_'
    thresh = 0.5
    if (len(sys.argv) == 5):
        prefix = sys.argv[3]
        thresh = float(sys.argv[4])

    # parse datacfg file
    lines = open(datacfg).readlines()
    images = []
    classes = []
    for line in lines:
        if (line.split(' ')[0] == 'names'):
            names_path = line.strip().split(' ')[-1]
            if (names_path[0] != '/'):
                names_path = workspace + names_path
            lists = open(names_path).readlines()
            classes = [x.strip() for x in lists]
        if (line.split(' ')[0] == 'valid'):
            valid_path = line.strip().split(' ')[-1]
            if (valid_path[0] != '/'):
                valid_path = workspace + valid_path
            lists = open(valid_path).readlines()
            images = [x.strip() for x in lists]

    # evaluate
    aps = []
    for idx, classname in enumerate(classes):
        # load ground truth
        class_recs = {}
        npos = 0
        for image in images:
            label = image.replace('.jpg', '.txt')
            split = label.rfind('/')
            label = label[:split] + '_gt' + label[split:]
            lines = open(label).readlines()
            bbox = []
            for line in lines:
                t = line.strip().split(',')
                if (t[0] in ['p', 'f', 'w']):
                    bbox.append([int(t[2]), int(t[3]), int(t[2]) + int(t[4]), int(t[3]) + int(t[5])])
            image_id = image.split('/')[-1].replace('.jpg', '')
            npos += len(bbox)
            bbox = np.array(bbox)
            det = [False] * len(bbox)
            class_recs[image_id] = {'bbox': bbox, 'det': det}
        # load results
        results = open(resdir + prefix + classname + '.txt').readlines()
        # calc recall, precision and AP
        print("%s %d %d" % (classname, npos, len(results)))
        rec, prec, ap = evaluate(class_recs, npos, results, ovthresh=thresh)
        aps.append(ap)
        print("Max Recall: %f" % np.max(rec))
        print("Max Precision: %f" % np.max(prec))
        print("AP@%s: %f" % (classname, ap))
        # plt.plot(rec, prec, 'b')
        # plt.xlim(0, 1)
        # plt.ylim(0, 1)
        # plt.xlabel("recall")
        # plt.ylabel("precision")
        # plt.title("%s%.1f" % (prefix, thresh))
        # plt.savefig("roc_%s.jpg", classname)
        # plt.show()

    # calc mAP
    mAP = np.average(np.array(aps))
    print(mAP)
