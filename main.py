import cv2
import numpy as np


def compute_combined_energy(saliency_map, depth_map , alpha , beta):
    pass

def find_seam(energy):
    rows, cols = energy.shape
    cost = np.zeros_like(energy, dtype=np.float64)
    path = np.zeros_like(energy, dtype=np.int64)

    cost[0, :] = energy[0, :]

    for i in range(1, rows):
        for j in range(cols):
            min_cost = cost[i-1, j]
            path[i, j] = j

            if j > 0 and cost[i-1, j-1] < min_cost:
                min_cost = cost[i-1, j-1]
                path[i, j] = j-1
            if j < cols - 1 and cost[i-1, j+1] < min_cost:
                min_cost = cost[i-1, j+1]
                path[i, j] = j+1

            cost[i, j] = energy[i, j] + min_cost

    seam = np.zeros(rows, dtype=np.int64)
    seam[-1] = np.argmin(cost[-1])
    for i in range(rows-2, -1, -1):
        seam[i] = path[i+1, seam[i+1]]

    return seam

def remove_seam(image, seam):
    rows, cols = image.shape[:2]
    if len(image.shape) == 3:
        output = np.zeros((rows, cols - 1, 3), dtype=image.dtype)
        for i in range(rows):
            j = seam[i]
            output[i, :, 0] = np.delete(image[i, :, 0], j)
            output[i, :, 1] = np.delete(image[i, :, 1], j)
            output[i, :, 2] = np.delete(image[i, :, 2], j)
    else:
        output = np.zeros((rows, cols - 1), dtype=image.dtype)
        for i in range(rows):
            j = seam[i]
            output[i, :] = np.delete(image[i, :], j)
    return output


def seam_carving(image, saliency_map, depth_map, scale_percent, alpha , beta):
    height, width, _ = image.shape
    new_width = int(width * scale_percent)

    energy = compute_combined_energy(saliency_map, depth_map, alpha , beta)

    while width > new_width:
        seam = find_seam(energy)
        image = remove_seam(image, seam)
        energy = remove_seam(energy, seam)
        width -= 1

    return image