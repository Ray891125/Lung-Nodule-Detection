import matplotlib.pyplot as plt
import os
import numpy as np
import my_csvTools  
import warnings
import math
class PatientData:
    def __init__(self, data_dir, patient_id=0):
        self.data_dir = data_dir
        self.patient_id = patient_id
        self.image = None
        self.pred_boxes = None
        self.gt_boxes = None
        self.fn_boxes = None

    def load_bbox(self, pred_dir=None, fn_dir=None, gt_dir=None):
        self.load_image()
        self.load_pred(pred_dir)
        self.load_gt(gt_dir)
        self.load_fn(fn_dir)

    def load_image(self):
        try:
            image_path = os.path.join(self.data_dir, f"{self.patient_id}_crop.npy")
            self.image = np.load(image_path)
            self._preprocess_image()
        except Exception as e:
            raise IOError(f"Failed to load image: {e}")

    def _preprocess_image(self):
        new_img = (self.image - -1200) / (600 - -1200)
        new_img[new_img < 0] = 0
        new_img[new_img > 1] = 1
        self.image = (new_img * 255).astype('uint8')

    def load_pred(self, pred_dir):
        try:
            self.pred_boxes = my_csvTools.read_csv_per_patient(pred_dir, self.patient_id)
        except Exception as e:
            warnings.warn(f"Failed to load pred data: {e}", Warning)

    def load_gt(self, gt_dir):
        try:
            self.gt_boxes = my_csvTools.read_csv_per_patient(gt_dir, self.patient_id)
        except Exception as e:
            warnings.warn(f"Failed to load ground truth data: {e}", Warning)
    def load_fn(self, fn_dir):
        try:
            self.fn_boxes = my_csvTools.read_csv_per_patient(fn_dir, self.patient_id)
        except Exception as e:
            warnings.warn(f"Failed to load false negative data: {e}", Warning)

import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, patient_data):
        self.patient_data = patient_data
        self.colors = {'pred': '#FF0000', 'gt': '#00FF00', 'fn': '#0000FF'}
        self.current_slice = 0
        self.current_image = plt.figure(figsize=(12, 8))
    def create_figure(self):
        plt.figure(figsize=(12, 8))
        plt.title(self.patient_data.patient_id )
        plt.xticks([])
        plt.yticks([])  
        plt.box(False)
    def draw_boxes(self, box_list,mode='pred'):
        """Draws bounding boxes on the image."""
        txt_color = '#000000'
        edge_color = self.colors.get(mode, '#FF0000')
        
        for box in box_list:
            if len(box) < 7:
                box.append(100)
            box = [my_csvTools.tryFloat(num) for num in box] 
            if type(box[-1])!=str and box[-1] < 0.9:
                continue

            x, y, z, w, h, d, _ = box
            start_slice = z - d / 2
            end_slice = z + d / 2

            if start_slice <= self.current_slice <= end_slice:
                rect = plt.Rectangle(
                    (x - w / 2, y - h / 2), w, h, fill=False, edgecolor=edge_color, linewidth=1
                )
                plt.gca().add_patch(rect)

                if mode == 'pred' :
                    #text = f"{int(w)},{int(h)},{int(d)},{math.floor(box[-1]*100)/100.0}"
                    text = f"{math.floor(box[-1]*100)/100.0}"
                    plt.text(
                        x - w / 2, y - h / 2-5, text, color=txt_color,
                        bbox={'edgecolor': edge_color, 'facecolor': edge_color, 'alpha': 0.5, 'pad': 0}
                    )

    def draw_image(self, slice_id=None, is_pred=True, is_gt=True, is_fn=True):
        """Displays the image slice with bounding boxes."""
        if slice_id is None:
            slice_id = self.current_slice
        else:
            self.current_slice = slice_id

        slice_img = self.patient_data.image[:, :, slice_id]
        plt.title(f"{self.patient_data.patient_id} - Slice {slice_id}")
        plt.imshow(slice_img, cmap='gray', vmin=0, vmax=255)
        if is_pred and self.patient_data.pred_boxes is not None:
            self.draw_boxes(self.patient_data.pred_boxes, mode='pred')
        if is_gt and self.patient_data.gt_boxes is not None:
            self.draw_boxes(self.patient_data.gt_boxes, mode='gt')
        if is_fn and self.patient_data.fn_boxes is not None:
            self.draw_boxes(self.patient_data.fn_boxes, mode='fn')

        plt.connect('key_press_event', self.key_press_callback)  # 連接按鍵事件
        plt.show()

    def key_press_callback(self, event):
        if event.key == 'n':
            plt.ion()
            plt.clf()
            plt.ioff()
            self.current_slice += 1
            if self.current_slice >= self.patient_data.image.shape[2]:
                self.current_slice = 0
            self.draw_image(self.current_slice)

def main():

    data_dir = r"C:\Users\BB\Desktop\training_data\My_SAnet\Data\Dicom_crop"
    gt_dir = r"C:\Users\BB\Desktop\training_data\nodule_eval\annotation_val.csv"
    pred_dir = r"C:\Users\BB\Desktop\training_data\nodule_eval\predict_result\submission_rpn.csv"
    fn_dir = None
    patient_id = "CHESTCT1489"

    # 創建一個 PatientData 對象並加載資料
    patient_data = PatientData(data_dir, patient_id)
    patient_data.load_bbox(pred_dir, fn_dir, gt_dir)

    # 創建一個 Visualizer 對象並使用 PatientData 對象進行繪製   
    visualizer = Visualizer(patient_data)

    # 顯示第一個切片的圖像和標註框
    slice_id = 35
    visualizer.draw_image(slice_id,is_pred = True, is_gt = True, is_fn = False)

if __name__ == "__main__":
    main()
