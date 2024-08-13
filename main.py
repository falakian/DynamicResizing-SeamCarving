import cv2
import numpy as np

# Normalize energy matrix to the range [0, 255] and convert to uint8
def normalize_energy(energy):
    image = (energy - energy.min()) / (energy.max() - energy.min()) * 255
    image = image.astype(np.uint8)
    return image

# Apply thresholds to the sum of saliency and depth maps to generate the result matrix
def apply_thresholds(image_sum, threshold, s_map):
    # Create masks based on the threshold
    mask_sum_low = cv2.inRange(image_sum, 0, threshold)
    mask_sum_high = cv2.inRange(image_sum, threshold + 1, 255)

    # Initialize result matrix with zeros
    result = np.zeros_like(image_sum)

    # Assign values based on the masks
    result[mask_sum_high > 0] = 255
    result[mask_sum_low > 0] = s_map[mask_sum_low > 0]

    return result

# Compute the combined energy matrix using saliency and depth maps with given weights
def compute_combined_energy(saliency_map, depth_map, alpha, beta):
    # Combine saliency and depth maps
    sum_senergy = depth_map + saliency_map

    # Calculate the threshold using Otsu's method
    threshold, _ = cv2.threshold(sum_senergy, sum_senergy.min(), sum_senergy.max(), cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply thresholds and normalize the combined energy map
    combined_energy = apply_thresholds(sum_senergy, threshold, saliency_map)
    combined_energy = normalize_energy(combined_energy).astype(np.uint16)
    depth_map = normalize_energy(depth_map).astype(np.uint16)
    saliency_map = normalize_energy(saliency_map).astype(np.uint16)

    # Compute the final energy matrix with given weights
    sum_energy = (alpha) * combined_energy + (beta) * depth_map + (1 - alpha - beta) * saliency_map
    
    return sum_energy

# Find the seam with the lowest energy cost
def find_seam(energy):
    rows, cols = energy.shape
    cost = np.zeros_like(energy, dtype=np.float64)
    path = np.zeros_like(energy, dtype=np.int64)

    # Initialize the cost of the first row
    cost[0, :] = energy[0, :]

    # Calculate the cost for each pixel in the image
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

    # Trace back the seam from the last row to the first
    seam = np.zeros(rows, dtype=np.int64)
    seam[-1] = np.argmin(cost[-1])
    for i in range(rows-2, -1, -1):
        seam[i] = path[i+1, seam[i+1]]

    return seam

# Remove the identified seam from the image
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

# Main function to perform seam carving
def seam_carving(image, saliency_map, depth_map, scale_percent, alpha, beta):
    height, width, _ = image.shape
    new_width = int(width * scale_percent)

    # Compute the initial energy map
    energy = compute_combined_energy(saliency_map, depth_map, alpha, beta)

    # Iteratively remove seams until the desired width is reached
    while width > new_width:
        seam = find_seam(energy)
        image = remove_seam(image, seam)
        energy = remove_seam(energy, seam)
        width -= 1

    return image

# Parameters
scale_percent = 0.5  # Scaling factor
alpha = 0.15  # Weight for the combined energy
beta = 0.7    # Weight for the depth map

# Process the "Snowman" image
image = cv2.imread('./Snowman/Snowman.png')
saliency_map = cv2.imread('./Snowman/Snowman_SMap.png', cv2.IMREAD_GRAYSCALE)
depth_map = cv2.imread('./Snowman/Snowman_DMap.png', cv2.IMREAD_GRAYSCALE)
result = seam_carving(image, saliency_map, depth_map, scale_percent, alpha, beta)
cv2.imshow('Seam Carved Image Snowman', result)
cv2.imwrite('./Snowman.png', result)

# Process the "Diana" image
image = cv2.imread('./Diana/Diana.png')
saliency_map = cv2.imread('./Diana/Diana_SMap.png', cv2.IMREAD_GRAYSCALE)
depth_map = cv2.imread('./Diana/Diana_DMap.png', cv2.IMREAD_GRAYSCALE)
result = seam_carving(image, saliency_map, depth_map, scale_percent, alpha, beta)
cv2.imshow('Seam Carved Image Diana', result)
cv2.imwrite('./Diana.png', result)

# Process the "Dolls" image
image = cv2.imread('./Dolls/Dolls.png')
saliency_map = cv2.imread('./Dolls/Dolls_SMap.png', cv2.IMREAD_GRAYSCALE)
depth_map = cv2.imread('./Dolls/Dolls_DMap.png', cv2.IMREAD_GRAYSCALE)
result = seam_carving(image, saliency_map, depth_map, scale_percent, alpha, beta)
cv2.imshow('Seam Carved Image Dolls', result)
cv2.imwrite('./Dolls.png', result)

# Process the "Baby" image
image = cv2.imread('./Baby/Baby.png')
saliency_map = cv2.imread('./Baby/Baby_SMap.png', cv2.IMREAD_GRAYSCALE)
depth_map = cv2.imread('./Baby/Baby_DMap.png', cv2.IMREAD_GRAYSCALE)
result = seam_carving(image, saliency_map, depth_map, scale_percent, alpha, beta)
cv2.imshow('Seam Carved Image Baby', result)
cv2.imwrite('./Baby.png', result)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
