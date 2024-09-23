import numpy as np

from typing import List, Tuple
class SplitComb():
    def __init__(self, crop_size: List[int]=[128, 128, 128], overlap_size:List[int]=[32, 32, 32], do_padding:bool=True, pad_value:int=0):
        self.stride_size = [crop_size[0]-overlap_size[0], 
                            crop_size[1]-overlap_size[1], 
                            crop_size[2]-overlap_size[2]]
        self.overlap = overlap_size
        self.crop_size = np.array(crop_size, dtype=np.int32)
        self.pad_value = pad_value
        self.do_padding = do_padding

    def split(self, image, lobe = None, out_stride: int = 4):
        do_padding = self.do_padding
        image_splits = []
        _,d, h, w = image.shape

        # Number of splits in each dimension
        nz = int(np.ceil(float(d) / self.stride_size[0]))
        ny = int(np.ceil(float(h) / self.stride_size[1]))
        nx = int(np.ceil(float(w) / self.stride_size[2]))

        nzyx = [nz, ny, nx]
        if do_padding or (d < self.crop_size[0]) or (h < self.crop_size[1]) or (w < self.crop_size[2]): # Last 3 condition for fail lobe
            pad = [[0, 0],[0, int(nz * self.stride_size[0] + self.overlap[0] - d)],
                    [0, int(ny * self.stride_size[1] + self.overlap[1] - h)],
                    [0, int(nx * self.stride_size[2] + self.overlap[2] - w)]]
            image = np.pad(image, pad, 'constant', constant_values=self.pad_value)
            do_padding = True
            
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    start_z = int(iz * self.stride_size[0])
                    end_z = start_z + self.crop_size[0]
                    if end_z > d and not do_padding:
                        start_z = d - self.crop_size[0]
                        end_z = d
                        
                    start_y = int(iy * self.stride_size[1])
                    end_y = start_y + self.crop_size[1]
                    if end_y > h and not do_padding:
                        start_y = h - self.crop_size[1]
                        end_y = h
                    
                    start_x = int(ix * self.stride_size[2])
                    end_x = start_x + self.crop_size[2]
                    if end_x > w and not do_padding:
                        start_x = w - self.crop_size[2]
                        end_x = w

                    image_split = image[np.newaxis, :, start_z:end_z, start_y:end_y, start_x:end_x]

                    image_splits.append(image_split)

        image_splits = np.concatenate(image_splits, 0)  
        return image_splits, nzyx, [d, h, w]

    def combine(self, output, nzhw, image_shape: Tuple[int, int, int]):
        nz, nh, nw = nzhw
        idx = 0
        for iz in range(nz): 
            for ih in range(nh):
                for iw in range(nw):
                    sz = int(iz * self.stride_size[0])
                    sh = int(ih * self.stride_size[1])
                    sw = int(iw * self.stride_size[2])
                    
                    if sz + self.crop_size[0] > image_shape[0] and not self.do_padding:
                        sz = image_shape[0] - self.crop_size[0]
                        
                    if sh + self.crop_size[1] > image_shape[1] and not self.do_padding:
                        sh = image_shape[1] - self.crop_size[1]
                        
                    if sw + self.crop_size[2] > image_shape[2] and not self.do_padding:
                        sw = image_shape[2] - self.crop_size[2]
                    
                    # [N, 7]
                    # 7->  prob, z_min, y_min, x_min, d, h, w 
                    output[idx][:, 1] += sz
                    output[idx][:, 2] += sh
                    output[idx][:, 3] += sw
                    idx += 1
        return output