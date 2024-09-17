import numpy as np

class SplitComb():
    def __init__(self, side_len, max_stride, stride, margin, pad_value):
        self.side_len = side_len
        self.max_stride = max_stride
        self.stride = stride
        self.margin = margin
        self.pad_value = pad_value

    # def split(self, bboxes, data, idx, num_divisions=4, result_cube_size=[128,128,128]):
    #     centers = self._calculate_centers(data.shape[1:], num_divisions, result_cube_size)
    #     center = centers[idx]
    #     sz, ez = int(center[0] - result_cube_size[0] / 2), int(center[0] + result_cube_size[0] / 2)
    #     sh, eh = int(center[1] - result_cube_size[1] / 2), int(center[1] + result_cube_size[1] / 2)
    #     sw, ew = int(center[2] - result_cube_size[2] / 2), int(center[2] + result_cube_size[2] / 2)

    #     split = data[:, sz:ez, sh:eh, sw:ew]

    #     new_bbox = []
    #     for bbox in bboxes:
    #         if sz <= bbox[0] < ez and sh <= bbox[1] < eh and sw <= bbox[2] < ew:
    #             new_bbox.append([bbox[0] - sz, bbox[1] - sh, bbox[2] - sw, bbox[3], bbox[4], bbox[5]])

    #     label = np.ones(len(new_bbox)) if new_bbox else np.zeros(1)
    #     if not new_bbox:
    #         new_bbox = np.zeros((1, 6))
    #     if split.shape != (1, result_cube_size[0], result_cube_size[1], result_cube_size[2]):
    #         draw_image(split)
    #     return np.array(split), np.array(new_bbox), np.array(label), np.array([center[0], center[1], center[2]])
    def split(self, data, num_divisions=4 ):    
        # 获取数据的形状
        # 假设 data 是一个形状不固定的3D立方体数据
        # 获取数据的形状
        _, depth, height, width = data.shape

        # 计算需要填充的数量，使得深度可以被4整除
        pad_depth = (4 - depth % 4) % 4

        # 在数据的z轴方向上进行填充
        padded_data = np.pad(data, ((0, 0), (0, pad_depth), (0, 0), (0, 0)), mode='constant',constant_values=pad_value)

        # 填充后的新深度
        new_depth = padded_data.shape[1]

        # 每个立方体在z轴的深度
        cube_depth = new_depth // 4

        # 初始化分割后的立方体和起始坐标的列表
        cubes = []
        start_coords = []

        # 按照z轴分割立方体
        for i in range(4):
            start_z = i * cube_depth
            end_z = start_z + cube_depth
            cube = np.array(padded_data[:, start_z:end_z, :, :])
            cubes.append(cube)
            start_coords.append((0, start_z, 0, 0))

        return np.array(cubes), np.array(start_coords)

    def combine(self, output, nzhw=None, side_len=None, stride=None, margin=None):
        if side_len is None:
            side_len = self.side_len
        if stride is None:
            stride = self.stride
        if margin is None:
            margin = self.margin
        if nzhw is None:
            nz = self.nz
            nh = self.nh
            nw = self.nw
        else:
            nz, nh, nw = nzhw

        assert (side_len % stride == 0)
        assert (margin % stride == 0)
        side_len //= stride
        margin //= stride

        splits = []
        for i in range(len(output)):
            splits.append(output[i])

        output = -1000000 * np.ones((
            nz * side_len,
            nh * side_len,
            nw * side_len,
            splits[0].shape[3],
            splits[0].shape[4]), np.float32)

        idx = 0
        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz = iz * side_len
                    ez = (iz + 1) * side_len
                    sh = ih * side_len
                    eh = (ih + 1) * side_len
                    sw = iw * side_len
                    ew = (iw + 1) * side_len

                    split = splits[idx][margin:margin + side_len, margin:margin + side_len, margin:margin + side_len]
                    output[sz:ez, sh:eh, sw:ew] = split
                    idx += 1

        return output
    def _calculate_centers(self,big_cube_sizes, num_divisions,result_cube_size):
        start_z = result_cube_size[0]/2
        start_y = result_cube_size[1]/2
        start_x = result_cube_size[2]/2

        end_z = big_cube_sizes[0] - result_cube_size[0]/2
        end_y = big_cube_sizes[1] - result_cube_size[1]/2
        end_x = big_cube_sizes[2] - result_cube_size[2]/2

        margin_z = np.ceil((end_z - start_z) / (num_divisions-1))
        margin_y = np.ceil((end_y - start_y) / (num_divisions-1))
        margin_x = np.ceil((end_x - start_x) / (num_divisions-1))
        
        centers = []
        for i in range(num_divisions):
            for j in range(num_divisions):
                for k in range(num_divisions):
                    center_z = int(start_z+i * margin_z)
                    center_y = int(start_y+j * margin_y)
                    center_x = int(start_x+k * margin_x)
                    if k == num_divisions-1:
                        center_x = int(end_x)
                    if j == num_divisions-1:
                        center_y = int(end_y)
                    if i == num_divisions-1:
                        center_z = int(end_z)
                    centers.append([center_z, center_y, center_x])
        
        return centers 
    
def draw_image(image):
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('TkAgg')
    plt.figure(figsize=(6, 6))
    plt.clf()
    plt.imshow(image[0][20,:,:], cmap='gray')
    plt.axis('off')
    plt.show()
# # 測試 SplitComb 類
# side_len = 128
# max_stride = 16
# stride = 4
# margin = 0
# pad_value = 3

# split_comb = SplitComb(side_len, max_stride, stride, margin, pad_value)

# # 創建一個隨機的 512x512x512 的影像
# input_data = np.random.rand(1, 512, 512, 512).astype(np.float32)

# # 分割影像
# splits, nzhw,new_center = split_comb.split(input_data)

# # 確認分割後的小影像尺寸
# for split in splits:
#     print(split.shape)  # 每個小影像的尺寸應該是 (1, 128, 128, 128)

# # 組合影像
# # output_data = split_comb.combine(splits, nzhw)

# # # 確認組合後的影像與原始影像相同
# # print(np.allclose(input_data, output_data))  # 應該輸出 True
