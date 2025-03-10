#!/usr/bin/env python
import nibabel as nib
import numpy as np
import trimesh
from scipy import ndimage
import os
from skimage import measure, filters, morphology, segmentation
import matplotlib.pyplot as plt

# Load the MRI file
mri_file = '101.nii'
print(f"Loading MRI file: {mri_file}")
img = nib.load(mri_file)
data = img.get_fdata()

# Normalize the data
data_norm = (data - data.min()) / (data.max() - data.min())

# Let's try a more robust brain extraction approach
# First, let's find a more suitable slice to analyze

# Find axial slices with the most content (middle of the brain)
# Sum across x and y axes for each z slice to find slices with most content
slice_sums = np.sum(data_norm > 0.1, axis=(1, 2))
best_z = np.argmax(slice_sums)
print(f"Best z-slice: {best_z}")

# Find best y-slice (coronal)
y_slice_sums = np.sum(data_norm > 0.1, axis=(0, 2))
best_y = np.argmax(y_slice_sums)
print(f"Best y-slice: {best_y}")

# Find best x-slice (sagittal)
x_slice_sums = np.sum(data_norm > 0.1, axis=(0, 1))
best_x = np.argmax(x_slice_sums)
print(f"Best x-slice: {best_x}")

# Save diagnostic images showing the best slices
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(data_norm[best_z, :, :], cmap='gray')
plt.title('Best Axial (z) Slice')
plt.subplot(132)
plt.imshow(data_norm[:, best_y, :], cmap='gray')
plt.title('Best Coronal (y) Slice')
plt.subplot(133)
plt.imshow(data_norm[:, :, best_x], cmap='gray')
plt.title('Best Sagittal (x) Slice')
plt.tight_layout()
plt.savefig('best_slices.png')
plt.close()

# Use a boundary-based approach to detect the brain-skull interface

# 1. Start with basic thresholding to get a rough segmentation
threshold = filters.threshold_otsu(data_norm)
print(f"Otsu threshold: {threshold}")

# 2. Create a gradient magnitude image to highlight boundaries
# This will show the CSF-skull and brain-CSF boundaries clearly
from scipy import ndimage
gradient = ndimage.gaussian_gradient_magnitude(data_norm, sigma=1.0)

# 3. Normalize the gradient for better visualization
gradient_norm = gradient / gradient.max()

# Advanced skull stripping approach with multiple layers of refinement

# Create a gradient magnitude image to highlight tissue boundaries
gradient = ndimage.gaussian_gradient_magnitude(data_norm, sigma=1.5)
gradient_norm = gradient / gradient.max()

# Set thresholds for different tissue types
# In T1 MRI: CSF is dark, gray matter is mid-intensity, white matter is bright
csf_thresh = threshold * 0.7
brain_thresh_min = threshold * 1.2
brain_thresh_max = 0.95
skull_thresh = threshold * 1.3  # Higher intensity for skull/bone
print(f"CSF threshold: {csf_thresh}")
print(f"Brain threshold range: {brain_thresh_min} to {brain_thresh_max}")
print(f"Skull threshold: {skull_thresh}")

# Generate masks for tissue types 
csf_mask = data_norm < csf_thresh
brain_tissue = (data_norm > brain_thresh_min) & (data_norm < brain_thresh_max)
potential_skull = data_norm > skull_thresh  # Bone tends to be brighter

# Initial head mask
head_mask = data_norm > (threshold * 0.4)
head_mask = ndimage.binary_closing(head_mask, iterations=3)
head_mask = ndimage.binary_fill_holes(head_mask)

# Create an erosion mask (well inside the brain)
eroded_head = ndimage.binary_erosion(head_mask, iterations=18)  # More aggressive erosion
eroded_head = morphology.remove_small_objects(eroded_head, min_size=5000)

# Get a conservative initial brain region
initial_brain = eroded_head & (data_norm > brain_thresh_min * 0.9)
initial_brain = morphology.remove_small_objects(initial_brain, min_size=1000)

# Create a more detailed brain boundary mask using edge detection
edge_mask = gradient_norm > 0.15
edge_mask = morphology.remove_small_objects(edge_mask, min_size=20)

# Create a skull boundary mask 
skull_boundary = edge_mask & potential_skull
skull_boundary = ndimage.binary_dilation(skull_boundary, iterations=2)

# Create a brain-expand mask that avoids skull regions
expansion_mask = ~skull_boundary & head_mask

# Expand the initial brain mask to fill the brain cavity, avoiding skull regions
expanded_brain = initial_brain.copy()
for i in range(5):  # Controlled expansion with skull avoidance
    expanded_brain = ndimage.binary_dilation(expanded_brain, iterations=1)
    expanded_brain = expanded_brain & expansion_mask
    expanded_brain = ndimage.binary_fill_holes(expanded_brain)

# Create a mask just for high-intensity brain tissue (white matter)
brain_tissue_high = (data_norm > brain_thresh_min) & (data_norm < brain_thresh_max) & expanded_brain

# Dilate this high-intensity brain tissue to include adjacent gray matter
dilated_brain_tissue = ndimage.binary_dilation(brain_tissue_high, iterations=3)
dilated_brain_tissue = dilated_brain_tissue & expanded_brain

# Apply intensity constraints to avoid non-brain tissue like CSF
brain_mask = dilated_brain_tissue & (data_norm > csf_thresh * 0.8)

# One more time with a simpler approach to create a solid brain model
# Start with a very conservative brain region that we're confident is inside the brain
inner_brain_mask = ndimage.binary_erosion(head_mask, iterations=20)
inner_brain_mask = morphology.remove_small_objects(inner_brain_mask, min_size=5000)

# Use a multi-threshold approach for finding brain-skull boundary
# T1 MRI characteristics: CSF is dark, gray matter is darker than white matter, skull is bright
# Prepare threshold masks for all tissues of interest
csf_mask = data_norm < (threshold * 0.8)  # CSF appears darker
white_matter = data_norm > (threshold * 1.1)  # White matter appears brighter
skull_bone = data_norm > (threshold * 1.3)  # Skull bone is even brighter

# Highlight the boundaries
skull_boundary = edge_mask & (skull_bone | csf_mask) 
skull_boundary = ndimage.binary_dilation(skull_boundary, iterations=2)

# Create a search region that avoids bone/skull
search_region = head_mask & (~skull_boundary)

# One last try for a solid brain mask
# Start with a more conservative threshold to focus on just the solid parts
midline_mask = (data_norm > threshold * 0.8) & (data_norm < threshold * 1.7)
midline_mask = midline_mask & head_mask  # Constrain to head region

# Erode the head mask to get well inside the brain
eroded = ndimage.binary_erosion(head_mask, iterations=18)
eroded = morphology.remove_small_objects(eroded, min_size=5000)

# Combine with midline mask to get robust brain regions
brain_core = eroded & midline_mask
brain_core = morphology.remove_small_objects(brain_core, min_size=1000)

# Now expand with aggressive hole filling
brain_mask = brain_core.copy()
for i in range(15):
    # Dilate by a small amount
    dilated = ndimage.binary_dilation(brain_mask, iterations=1)
    # Only expand into head regions
    brain_mask = dilated & head_mask
    # Fill holes frequently
    if i % 2 == 0:
        brain_mask = ndimage.binary_fill_holes(brain_mask)

# Stop expansion at the skull boundary
brain_mask = brain_mask & (~skull_boundary)

# Very aggressive hole filling and closing for a solid model
brain_mask = ndimage.binary_fill_holes(brain_mask)
# Use multiple passes of closing with different structuring elements
for _ in range(3):
    brain_mask = ndimage.binary_closing(brain_mask, iterations=5)
    brain_mask = ndimage.binary_fill_holes(brain_mask)

# Very minimal opening to preserve shape
brain_mask = ndimage.binary_opening(brain_mask, iterations=1)

# Ensure we still have a good size object
brain_mask = morphology.remove_small_objects(brain_mask, min_size=10000)

# 10. Get the largest connected component (should be the brain)
labels, num_labels = ndimage.label(brain_mask)
if num_labels > 0:
    sizes = ndimage.sum(brain_mask, labels, range(1, num_labels + 1))
    largest_label = np.argmax(sizes) + 1
    brain_mask = labels == largest_label
    print(f"Keeping largest connected component ({np.sum(brain_mask)} voxels)")
else:
    print("No distinct components found")

# 11. Clean up with morphological operations
brain_mask = ndimage.binary_closing(brain_mask, iterations=2)
brain_mask = ndimage.binary_opening(brain_mask, iterations=1)

# This is the final brain mask - convert to float for marching cubes
binary_brain = brain_mask.astype(np.float32)

# Save diagnostic images showing the skull detection process
plt.figure(figsize=(15, 5))
plt.subplot(141)
plt.imshow(data_norm[best_z, :, :], cmap='gray')
plt.title('Original MRI')
plt.subplot(142)
plt.imshow(edge_mask[best_z, :, :], cmap='gray')
plt.title('Edge Detection')
plt.subplot(143)
plt.imshow(skull_boundary[best_z, :, :], cmap='gray')
plt.title('Skull Boundary')
plt.subplot(144)
plt.imshow(brain_mask[best_z, :, :], cmap='gray')
plt.title('Brain Mask')
plt.tight_layout()
plt.savefig('skull_boundary_detection.png')
plt.close()

# Save diagnostic images showing the expansion mask that avoids skull
plt.figure(figsize=(15, 5))
plt.subplot(141)
plt.imshow(data_norm[best_z, :, :], cmap='gray')
plt.title('Original MRI')
plt.subplot(142)
plt.imshow(initial_brain[best_z, :, :], cmap='gray')
plt.title('Initial Brain')
plt.subplot(143)
plt.imshow(expansion_mask[best_z, :, :], cmap='gray')
plt.title('Expansion Mask')
plt.subplot(144)
plt.imshow(brain_mask[best_z, :, :], cmap='gray')
plt.title('Final Brain Mask')
plt.tight_layout()
plt.savefig('skull_avoidance_process.png')
plt.close()

# Overlay the brain mask on the original MRI for better visualization
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(data_norm[best_z, :, :], cmap='gray')
plt.title('Original MRI')

# Create a color overlay of the skull boundary
plt.subplot(132)
overlay = np.zeros((*data_norm[best_z].shape, 3))
overlay[..., 0] = data_norm[best_z]  # red channel = original image
overlay[..., 1] = data_norm[best_z]  # green channel = original image
overlay[..., 2] = data_norm[best_z]  # blue channel = original image
# Add red overlay for skull
skull_overlay = skull_boundary[best_z].astype(float) * 0.8
overlay[..., 0] = np.maximum(overlay[..., 0], skull_overlay)
overlay[..., 1] = np.where(skull_overlay > 0, overlay[..., 1] * 0.2, overlay[..., 1])
overlay[..., 2] = np.where(skull_overlay > 0, overlay[..., 2] * 0.2, overlay[..., 2])
plt.imshow(overlay)
plt.title('Skull Boundary (Red)')

# Create a color overlay of the brain mask
plt.subplot(133)
overlay = np.zeros((*data_norm[best_z].shape, 3))
overlay[..., 0] = data_norm[best_z]  # red channel = original image
overlay[..., 1] = data_norm[best_z]  # green channel = original image
overlay[..., 2] = data_norm[best_z]  # blue channel = original image
# Add green overlay for brain
brain_overlay = brain_mask[best_z].astype(float) * 0.7
overlay[..., 1] = np.maximum(overlay[..., 1], brain_overlay)
overlay[..., 0] = np.where(brain_overlay > 0, overlay[..., 0] * 0.3, overlay[..., 0])
overlay[..., 2] = np.where(brain_overlay > 0, overlay[..., 2] * 0.3, overlay[..., 2])
plt.imshow(overlay)
plt.title('Brain Mask (Green)')

plt.tight_layout()
plt.savefig('brain_skull_overlay.png')
plt.close()

# Generate a sagittal view of the brain mask to verify it's working in 3D
plt.figure(figsize=(10, 10))
plt.subplot(221)
plt.imshow(data_norm[best_z, :, :], cmap='gray')
plt.title('Axial Slice (Original)')
plt.subplot(222)
plt.imshow(brain_mask[best_z, :, :], cmap='gray')
plt.title('Axial Slice (Brain Mask)')
plt.subplot(223)
plt.imshow(data_norm[:, :, best_x], cmap='gray')
plt.title('Sagittal Slice (Original)')
plt.subplot(224)
plt.imshow(brain_mask[:, :, best_x], cmap='gray')
plt.title('Sagittal Slice (Brain Mask)')
plt.tight_layout()
plt.savefig('brain_mask_3d_view.png')
plt.close()

print("Saved diagnostic images")

# This is the final brain mask
binary_brain = brain_mask

# Extract surface mesh using marching cubes
from skimage import measure
verts, faces, normals, values = measure.marching_cubes(binary_brain, level=0.5)

# Create a mesh object
mesh = trimesh.Trimesh(vertices=verts, faces=faces, normals=normals)

# Get the affine transform from the MRI to correctly position the mesh
affine = img.affine
# Apply the transform to the vertices
verts_transformed = np.dot(verts, affine[:3, :3].T) + affine[:3, 3]
mesh.vertices = verts_transformed

# Smooth the mesh to make it more visually appealing
# Fix the deprecation warning by using trimesh.graph.smooth_shade
trimesh.graph.smooth_shade(mesh)

# Save the mesh as STL and OBJ formats with brain-specific naming
output_stl = os.path.splitext(mri_file)[0] + "_brain_solid.stl"
output_obj = os.path.splitext(mri_file)[0] + "_brain_solid.obj"

mesh.export(output_stl)
mesh.export(output_obj)

print(f"Brain-only mesh saved as {output_stl} and {output_obj}")
print(f"Mesh has {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")