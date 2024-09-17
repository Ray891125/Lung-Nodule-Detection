import torch
from typing import Tuple

def box_center_dist(boxes1: torch.Tensor, boxes2: torch.Tensor, euclidean: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Distance of center points between two sets of boxes

    Arguments:
        boxes1: boxes; (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
        boxes2: boxes; (x1, y1, x2, y2, (z1, z2))[M, dim * 2]
        euclidean: computed the euclidean distance otherwise it uses the l1
            distance

    Returns:
        Tensor: the NxM matrix containing the pairwise
            distances for every element in boxes1 and boxes2; [N, M]
    """
    center1 = boxes1
    center2 = boxes2

    if euclidean:
        print(center1[:, None] )
        print(center2[None]) 
        dists = (center1[:, None] - center2[None]).pow(2).sum(-1).sqrt()
    else:
        # before sum: [N, M, dims]
        dists = (center1[:, None] - center2[None]).abs().sum(-1)
    return dists

# Example usage:
boxes1 = torch.tensor([[1, 1, 4], [2, 2, 5], [3, 3, 6]])
boxes2 = torch.tensor([[2, 2, 5], [3, 3, 6]])
euclidean_distances = box_center_dist(boxes1, boxes2, euclidean=True)
l1_distances = box_center_dist(boxes1, boxes2, euclidean=False)

print("Euclidean distances:")
print(euclidean_distances)

print("\nL1 distances:")
print(l1_distances)
