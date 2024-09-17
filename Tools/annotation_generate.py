import os
import json
import pandas as pd
data_dir = r'C:\Users\BB\Desktop\training_data\My_SAnet\Data\Bbox_crop_anno'
set_name_dir = r'C:\Users\BB\Desktop\training_data\My_SAnet\Data\train.txt'
# filenames = os.listdir(data_dir)
with open(set_name_dir, "r") as f:
    filenames = f.read().splitlines()

col_names = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'w', 'h', 'd', 'diamemter_mm']
submission_path = os.path.join(data_dir, 'annotation_train_crop.csv')
labels = []
five = 0.
five_ten = 0.
ten_twenty = 0.
twenty_thirty = 0.
thirty_fifty = 0.
total = 0
for fn in filenames:
    with open(os.path.join(data_dir, '%s_nodule_count_crop.json' % (fn)), 'r') as f:
        annota= json.load(f)
        bboxes = annota['bboxes']
        if len(bboxes) > 0:
            for nodule in bboxes:  # 遍历输入列表
                top_left = nodule[0]  # 左上角坐标
                bottom_right = nodule[1]  # 右下角坐标
                z = (bottom_right[2] + top_left[2]) / 2
                y = (bottom_right[0] + top_left[0]) / 2
                x = (bottom_right[1] + top_left[1]) / 2
                d = (bottom_right[2] - top_left[2]) + 1
                h = (bottom_right[0] - top_left[0]) + 1
                w = (bottom_right[1] - top_left[1]) + 1
                labels.append([fn, x, y, z, w, h, d,max(d,h,w)]) 
                total += 1
                if max(d,h,w) < 5:
                    five += 1
                elif max(d,h,w) < 10:
                    five_ten += 1
                elif max(d,h,w) < 20:
                    ten_twenty += 1 
                elif max(d,h,w) < 30:
                    twenty_thirty += 1
                elif max(d,h,w) < 50:
                    thirty_fifty += 1
        else:
            print("No bboxes for %s" % fn) 
              
# print(five/total,five_ten/total,ten_twenty/total,twenty_thirty/total,thirty_fifty/total)
df = pd.DataFrame(labels, columns=col_names)
df.to_csv(submission_path, index=False)