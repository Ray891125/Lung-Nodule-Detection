import torch
import numpy as np


def train_collate(batch_samples):
    '''
        batch_samples: list contains samples*batch_size, [samples, samples, ...]
        smaples: list contains 4 sample, [sample, sample, sample]
        sample: list contains images, bbox, label
    '''
    batch = []
    for samples in batch_samples:
        for sample in samples:
            batch.append(sample)
    return single_collate(batch)

def single_collate(batch):
    batch_size = len(batch)
    inputs = torch.stack([batch[b][0] for b in range(batch_size)], 0)

    bboxes = [batch[b][1] for b in range(batch_size)]
    labels = [batch[b][2] for b in range(batch_size)]
    
    # add padding to bboxes and label to make the same size
    try:
        max_label_num = max(label.shape[0] for label in labels)
        if max_label_num > 0:
            label_padded = np.ones((len(labels), max_label_num)) * -1
            bbox_padded = np.ones((len(bboxes), max_label_num, 6)) * -1
            for idx, (label, bbox) in enumerate(zip(labels, bboxes)):
                if label.shape[0] > 0:
                    label_padded[idx, :label.shape[0]] = label
                    bbox_padded[idx, :bbox.shape[0], :] = bbox
        else:
            label_padded = np.ones((len(labels), 1)) *-1
            bbox_padded = np.ones((len(labels), 1, 6)) *-1
    except:
        print(bboxes)

    label_padded = torch.from_numpy(label_padded)
    bbox_padded = torch.from_numpy(bbox_padded)
    # if len(batch[0]) > 3:
    #     lobe_info = [batch[b][3] for b in range(batch_size)]
    #     return [inputs, bbox_padded, label_padded, lobe_info]
    return [inputs, bbox_padded, label_padded]

def eval_collate(batch):
    batch_size = len(batch)
    inputs = torch.stack([batch[b][0] for b in range(batch_size)], 0)

    bboxes = [batch[b][1] for b in range(batch_size)]
    labels = [batch[b][2] for b in range(batch_size)]
    
    # add padding to bboxes and label to make the same size
    max_label_num = max(label.shape[0] for label in labels)
    if max_label_num > 0:
        label_padded = np.ones((len(labels), max_label_num)) * -1
        bbox_padded = np.ones((len(bboxes), max_label_num, 6)) * -1
        for idx, (label, bbox) in enumerate(zip(labels, bboxes)):
            if label.shape[0] > 0:
                label_padded[idx, :label.shape[0]] = label
                bbox_padded[idx, :bbox.shape[0], :] = bbox
    else:
        label_padded = np.ones((len(labels), 1)) *-1
        bbox_padded = np.ones((len(labels), 1, 6)) *-1

    label_padded = torch.from_numpy(label_padded)
    bbox_padded = torch.from_numpy(bbox_padded)

    return [inputs, bbox_padded, label_padded]


def test_collate(batch):
    batch_size = len(batch)
    for b in range(batch_size): 
        inputs = torch.stack([batch[b][0]for b in range(batch_size)], 0)
        images = [batch[b][1] for b in range(batch_size)]

    return [inputs, images]
