import numpy as np
import matplotlib.pyplot as plt
lobe_box_dir = r"C:\Users\BB\Desktop\training_data\My_SAnet\Data\Dicom_crop"
lobe_box_save_dir = r"C:\Users\BB\Desktop\training_data\My_SAnet\Data\Dicom_crop_pad"
lobe_box_info_dir = r"E:\training_data\ME_dataset\CHEST1001\npy\lobe_info.txt"
lobe_info_dir = r"E:\training_data\ME_dataset\CHEST1001\npy\CHEST1001_lobe.npz"
border = 8


def apply_mask_to_image(image: np.ndarray, mask: np.ndarray, mask_value: int = 5) -> np.ndarray:
    """
    Apply a mask to an image. Pixels in the image where the mask is 0 will be set to mask_value.

    Parameters:
    - image: np.ndarray, the input image to be masked.
    - mask: np.ndarray, the mask to be applied to the image. Should be the same shape as the image.
    - mask_value: int, the value to assign to masked out pixels in the image. Default is 0.

    Returns:
    - masked_image: np.ndarray, the masked image.
    """
    if image.shape != mask.shape:
        raise ValueError("The shape of image and mask must be the same.")
    
    # Create a copy of the image to avoid modifying the original image
    masked_image = np.copy(image)
    
    # Apply the mask: set pixels in the image to mask_value where the mask is 0
    masked_image[mask == 0] = mask_value
    
    return masked_image
# get lobe mask range
def get_lobe_range(lobe_box_info_dir: str) -> dict:
    with open( lobe_box_info_dir, "r") as file:
      lobe_box_info = file.readlines()
    # lobe_range = (y_min, y_max, x_min, x_max, z_min, z_max)
    lobe_range = np.zeros(6, dtype=int)
    lobe_range[4:] = np.array(list(map(int, lobe_box_info[3].split(","))))
    lobe_range[:4] = np.array(list(map(int, lobe_box_info[4].split(","))))
    return lobe_range


# Get dicom name will be  processed
import os
import re
#files = os.listdir(lobe_box_dir)
files = []
with open(r"C:\Users\BB\Desktop\training_data\My_SAnet\Src\Tools\output.txt", "r") as file:
      for line in file:
        files.append(line.strip())
# CHEST1649_crop.npy
for file in files:
    #-----------------------------------
    # Load the lobe box
    #-----------------------------------
    lobe_box = np.load(os.path.join(lobe_box_dir,file))
    #print(os.path.join(lobe_box_dir,file))
    temp_file = file
    # Use regular expression to extract the ID portion
    match = re.match(r"(CHESTCT\d+)_crop.npy", temp_file)
    if match:
        # Extract the ID from the match object
        id = match.group(1)
        #filenames.append(id)
        lobe_range = get_lobe_range(r"E:\training_data\ME_dataset\{id}\npy\lobe_info.txt".format(id=id))
        lobe_mask = np.load(r"E:\training_data\ME_dataset\{id}\npy\{id}_lobe.npz".format(id=id))
        lobe_box_mask = lobe_mask['image'][lobe_range[0]-border:lobe_range[1]+border, lobe_range[2]-border:lobe_range[3]+border, lobe_range[4]:lobe_range[5]]

        lobe_box_img = apply_mask_to_image(lobe_box, lobe_box_mask, mask_value=5)
        np.save(os.path.join(lobe_box_save_dir,file), lobe_box_img)
        # lobe_slice_box = lobe_box_img[:, :, 150]
        # plt.imshow(lobe_slice_box, cmap='gray')
        #plt.imshow(lobe_box[:,:,30], cmap='gray')