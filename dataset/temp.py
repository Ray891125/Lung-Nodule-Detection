import math
def calculate_centers(big_cube_sizes, num_divisions,result_cube_size):
    start_z = result_cube_size[0]/2
    start_y = result_cube_size[1]/2
    start_x = result_cube_size[2]/2

    margin_z = math.ceil((big_cube_sizes[0] - result_cube_size[0]) / (num_divisions-1))
    margin_y = math.ceil((big_cube_sizes[1] - result_cube_size[1]) / (num_divisions-1))
    margin_x = math.ceil((big_cube_sizes[2] - result_cube_size[2]) / (num_divisions-1))
    
    centers = []
    for i in range(num_divisions):
        for j in range(num_divisions):
            for k in range(num_divisions):
                center_z = start_z+i * margin_z
                center_y = start_y+j * margin_y
                center_x = start_x+k * margin_x
                centers.append((center_z, center_y, center_x))
    
    return centers

big_cube_sizes = (300, 400, 500)  # The side lengths of the big cube along x, y, z axes
num_divisions = 4                 # Number of divisions along each dimension
cube = (128, 128, 128)            # The side lengths of the smaller cube along x, y, z axes
centers = calculate_centers(big_cube_sizes, num_divisions,cube)

for idx, center in enumerate(centers):
    print(f"Center {idx + 1}: {center}")
