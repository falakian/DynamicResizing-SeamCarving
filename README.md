# Seam Carving Project

This repository contains the implementation of a seam carving algorithm for content-aware image resizing. Seam carving is an advanced image processing technique that allows for the resizing of images by dynamically removing or adding pixels along the least important seams. This approach preserves the visual content of the image while adjusting its dimensions, making it particularly useful for applications where maintaining the integrity of the main subject is crucial.

# Steps
## 1. Calculate energy map: 
### Step 1: Combining Saliency and Depth Maps

The first step in the energy function calculation is to combine the saliency and depth maps. This approach ensures that the algorithm considers both visual prominence and spatial depth when determining the importance of each pixel.

- **Saliency Map:** Identifies the visually prominent regions of the image.
- **Depth Map:** Provides information about the spatial depth of the image.

By combining these two maps, we get a more accurate representation of pixel importance. The combination is done using simple addition:
```python
sum_map = saliency_map + depth_map
```
### Step 2: Calculating the Threshold and Intermediate Energy Matrix
Next, a threshold (T) is computed using Otsu's method on the sum_map. This threshold helps in distinguishing between significant and less significant pixels. Based on this threshold, an intermediate energy matrix (result) is calculated as follows:
```python
if sum_map[i, j] > T:
    result[i, j] = 255
else:
    result[i, j] = saliency_map[i, j]
```

### Step 3: Calculating the Final Energy Matrix
Finally, the enhanced energy matrix (Sum_energy) is computed by combining the intermediate energy matrix (result), depth map, and saliency map using the following weighted sum:
```python
Sum_energy = 0.15 * result + 0.7 * depth_map + 0.15 * saliency_map
```
These weights have been determined through extensive experimentation to ensure optimal image resizing results

## 2. Build accumulated cost matrix using forward energy: 
This step is implemented with dynamic programming. The value of each pixel is equal to its corresponding value in the energy map added to the minimum new neighbor energy introduced by removing one of its three top neighbors (top-left, top-center, and top-right)

## 3. Find and remove minimum seam from top to bottom edge: 
Backtracking from the bottom to the top edge of the accumulated cost matrix to find the minimum seam. All the pixels in each row after the pixel to be removed are shifted over one column to the left if it has index greater than the minimum seam.

## 4. Repeat step 1 - 3 until achieving targeting width 

# Result
### Result of the Diana
<div align=center>
  <table>
    <tr>
      <th>Orginal Image</th>
      <th>40% reduction</th>
      <th>50% reduction</th>
      <th>60% reduction</th>
    </tr>
    <tr>
      <td><img src="https://github.com/falakian/DynamicResizing-SeamCarving/blob/main/Diana/Diana.png" alt="Orginal Image" /></td>
      <td><img src="https://github.com/falakian/DynamicResizing-SeamCarving/blob/main/60per/Diana.png" alt="40% reduction" /></td>
      <td><img src="https://github.com/falakian/DynamicResizing-SeamCarving/blob/main/50per/Diana.png" alt="50% reduction" /></td>
      <td><img src="https://github.com/falakian/DynamicResizing-SeamCarving/blob/main/40per/Diana.png" alt="60% reduction" /></td>
    </tr>
  </table>
</div>

### Result of the Snowman
<div align=center>
  <table>
    <tr>
      <th>Orginal Image</th>
      <th>40% reduction</th>
      <th>50% reduction</th>
      <th>60% reduction</th>
    </tr>
    <tr>
      <td><img src="https://github.com/falakian/DynamicResizing-SeamCarving/blob/main/Snowman/Snowman.png" alt="Orginal Image" /></td>
      <td><img src="https://github.com/falakian/DynamicResizing-SeamCarving/blob/main/60per/Snowman.png" alt="40% reduction" /></td>
      <td><img src="https://github.com/falakian/DynamicResizing-SeamCarving/blob/main/50per/Snowman.png" alt="50% reduction" /></td>
      <td><img src="https://github.com/falakian/DynamicResizing-SeamCarving/blob/main/40per/Snowman.png" alt="60% reduction" /></td>
    </tr>
  </table>
</div>

### Result of the Dolls
<div align=center>
  <table>
    <tr>
      <th>Orginal Image</th>
      <th>40% reduction</th>
      <th>50% reduction</th>
      <th>60% reduction</th>
    </tr>
    <tr>
      <td><img src="https://github.com/falakian/DynamicResizing-SeamCarving/blob/main/Dolls/Dolls.png" alt="Orginal Image" /></td>
      <td><img src="https://github.com/falakian/DynamicResizing-SeamCarving/blob/main/60per/Dolls.png" alt="40% reduction" /></td>
      <td><img src="https://github.com/falakian/DynamicResizing-SeamCarving/blob/main/50per/Dolls.png" alt="50% reduction" /></td>
      <td><img src="https://github.com/falakian/DynamicResizing-SeamCarving/blob/main/40per/Dolls.png" alt="60% reduction" /></td>
    </tr>
  </table>
</div>

### Result of the Baby
<div align=center>
  <table>
    <tr>
      <th>Orginal Image</th>
      <th>40% reduction</th>
      <th>50% reduction</th>
      <th>60% reduction</th>
    </tr>
    <tr>
      <td><img src="https://github.com/falakian/DynamicResizing-SeamCarving/blob/main/Baby/Baby.png" alt="Orginal Image" /></td>
      <td><img src="https://github.com/falakian/DynamicResizing-SeamCarving/blob/main/60per/Baby.png" alt="40% reduction" /></td>
      <td><img src="https://github.com/falakian/DynamicResizing-SeamCarving/blob/main/50per/Baby.png" alt="50% reduction" /></td>
      <td><img src="https://github.com/falakian/DynamicResizing-SeamCarving/blob/main/40per/Baby.png" alt="60% reduction" /></td>
    </tr>
  </table>
</div>
