############################################################################################################################################################################°°
# These packages were missing and neede to install for me:
# pip install scikit-image
# pip install opencv-python


#For a quick intro to image analysis and the explanation of the metrics computed by this code, please refer to the provided slides.

# EDIT THE PATH OF THE FOLDER CONTAINING THE ORIGINAL IMAGES
folder_path = "./Images/" 

#1. You can rename images as you wish. Only files with the following extensions inside the folder will be processed: .png, .jpg, .jpeg, .bmp
#2. The images will be preprocessed by converting them to grayscale, increasing the contrast, and segmenting them using Otsu's thresholding.
#3. In the binary image, only the 4 largest connected components will be kept, namely corresponding to the 4 samples in the image.
#4. The samples will be sorted based on their centroids, in a grid-like fashion, indexing the 4 connected regions as top-left, bottom-left, top-right and bottom-right.

# EDIT THE PATH OF THE FOLDER IN WHICH YOU WANT TO OUTPUT THE STATISTICS AND THE LABELED IMAGES
output_folder = "./Processed dataset/df old/"

#In the output folder, the following files will be generated:

#1. A CSV file named "image_statistics.csv" containing the statistics of the samples and their voids. 
	#The voids in each sample are labeled from 1 to n, where n is the number of voids in the sample.

#2. Segmented images of the samples

#3.Labeled images of the voids in each sample, with the voids numbered from 1 to n, where n is the number of voids in the sample.
   #These images are necessary to understand the correspondence between the voids in each sample and their statistics in the CSV file.

############################################################################################################################################################################


import os
import re
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.io import imsave

# Get a list of all the image files in the folder, ignoring hidden files and directories
file_names = [f for f in os.listdir(folder_path) if not f.startswith(('.', '_', '-')) and os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

# Initialize an empty list to store the statistics
statistics = []

# Define a function to get the quadrant of a point
def get_quadrant(x, y, avg_x, avg_y):
	if y < avg_y and x < avg_x:
		return 0  # Top-left
	elif y >= avg_y and x < avg_x:
		return 1  # Bottom-left
	elif y < avg_y and x >= avg_x:
		return 2  # Top-right
	else:
		return 3  # Bottom-right

# Define a function to get the quadrant name of a point
def get_quadrant_name(x, y, avg_x, avg_y):
	if y < avg_y and x < avg_x:
		return "top_left"  # Top-left
	elif y >= avg_y and x < avg_x:
		return "bottom_left"  # Bottom-left
	elif y < avg_y and x >= avg_x:
		return "top_right"  # Top-right
	else:
		return "bottom_right"  # Bottom-right

# Iterate over the image files and convert them to a list of arrays
for image_file in file_names:
	# Open the image file
	image_path = os.path.join(folder_path, image_file)
	image = Image.open(image_path)
	
	# Convert the image to grayscale
	image = image.convert('L')

	# Increase the contrast
	enhancer = ImageEnhance.Contrast(image)
	image = enhancer.enhance(5)

	# Get dimensions
	width, height = image.size
	v_crop = 800
	o_crop = 1000
	
	# Check if the image is large enough to crop
	if width >= o_crop and height >= v_crop:
		# Determine the coordinates for a 1000x1000 square centered in the image
		left = (width - o_crop)/2
		top = (height - v_crop)/2
		right = (width + o_crop)/2
		bottom = (height + v_crop)/2

		# Crop the image and convert it to a numpy array
		image = image.crop((left, top, right, bottom))
		preprocessed_image = np.array(image)
	   
		# Segment the image using Otsu's thresholding
		otsu_thresh = threshold_otsu(preprocessed_image)
		segmented_image = preprocessed_image > otsu_thresh

		# Label the connected components in the segmented image
		labeled_segmented_image = label(segmented_image)

		# Calculate the size of each component
		regions_segmented = regionprops(labeled_segmented_image)
		sizes = [r.area for r in regions_segmented]

		# Sort the components by size and keep only the largest ones
		largest_components = sorted([(i, s) for i, s in enumerate(sizes)], key=lambda x: x[1], reverse=True)[:4]
		largest_labels = [x[0] + 1 for x in largest_components]  # labels start from 1
		segmented_image = np.isin(labeled_segmented_image, largest_labels)

		# Extract the largest connected regions
		image_regions = []
		for lbl in largest_labels:
			region = (labeled_segmented_image == lbl)
			image_regions.append(region)

		# Get the centroid for each region
		image_regions_centroids = [regionprops(region.astype(int))[0].centroid for region in image_regions]

		# Calculate the average x and y coordinates
		avg_x = sum(c[1] for c in image_regions_centroids) / len(image_regions_centroids)
		avg_y = sum(c[0] for c in image_regions_centroids) / len(image_regions_centroids)

		# Sort the indices based on the quadrant
		sorted_indices = sorted(range(len(image_regions_centroids)), key=lambda k: get_quadrant(image_regions_centroids[k][1], image_regions_centroids[k][0], avg_x, avg_y))

		# Sort the regions and centroids based on the sorted indices
		image_regions = [image_regions[i] for i in sorted_indices]
		image_regions_centroids = [image_regions_centroids[i] for i in sorted_indices]

		# Assign names to the sorted regions
		for idx, img in enumerate(image_regions):

			# Get the quadrant name
			quadrant = get_quadrant_name(image_regions_centroids[idx][1], image_regions_centroids[idx][0], avg_x, avg_y)
 
			# Convert the boolean image to an 8-bit unsigned integer image
			img = (img.astype(np.uint8) * 255)
			# Use regionprops to get properties of the region
			part_props = regionprops(img)[0]  # Get the first (and only) region
			# Crop the image with margin around the bounding box
			minr, minc, maxr, maxc = part_props.bbox
			minr_crop = max(minr - 15, 0)
			minc_crop = max(minc - 15, 0)
			maxr_crop = min(maxr + 15, img.shape[0])
			maxc_crop = min(maxc + 15, img.shape[1])
			padded_img = img[minr_crop:maxr_crop, minc_crop:maxc_crop]
			# Save the padded image
			filename = f"{image_file}_{quadrant}_segmented.png"
			output_path = os.path.join(output_folder, filename)
			imsave(output_path, padded_img)

			# Invert the padded image
			inv_padded_img = 255 - padded_img
			# Apply labeling
			labels = label(inv_padded_img, background=0)
			props_voids = regionprops(labels)
			# Find the biggest region
			biggest_region = max(props_voids, key=lambda region: region.area)
			# Set the pixels in the biggest region to 0
			labels[labels == biggest_region.label] = 0
			# Repeat labeling
			labels = label(labels, background=0)
			props_voids = regionprops(labels)
			# Create a figure and axes
			fig, ax = plt.subplots()
			# Display the image
			ax.imshow(labels, cmap='nipy_spectral')
			# Annotate label numbers
			for region in regionprops(labels):
				# Get the coordinates of the centroid
				y, x = region.centroid
				# Annotate the label number at the centroid
				ax.text(x, y, str(region.label), color='white')
			# Save the labeled image
			labeled_filename = f"{image_file}_{quadrant}_labeled.png"
			labeled_output_path = os.path.join(output_folder, labeled_filename)
			plt.savefig(labeled_output_path)
			# Close the figure to free up memory
			plt.close(fig)

			# For each statistic in regionprops, create a row with Image name, Position, Region type, ID, and each metric
			statistics.append({
				"Image name": image_file,
				"Position": quadrant,
				"Region type": "part",
				"ID": 0,
				"Area [pixels]": round(part_props.area, 3),
				"Perimeter [pixels]": round(part_props.perimeter, 3),
				"Eccentricity": round(part_props.eccentricity, 3),
				"Orientation [radians]": round(part_props.orientation, 3),
				"Solidity": round(part_props.solidity, 3),
				"Extent": round(part_props.extent, 3),
				"Major Axis Length [pixels]": round(part_props.major_axis_length, 3),
				"Minor Axis Length [pixels]": round(part_props.minor_axis_length, 3),
				"Equivalent Diameter [pixels]": round(part_props.equivalent_diameter, 3)
			})

			# Loop over each void
			for i, prop in enumerate(props_voids, start=1):
				# For each statistic in regionprops, create a row with Image name, Position, Region type, ID, and each metric
				statistics.append({
					"Image name": image_file,
					"Position": quadrant,
					"Region type": "void",
					"ID": i,
					"Area [pixels]": round(prop.area, 3),
					"Perimeter [pixels]": round(prop.perimeter, 3),
					"Eccentricity": round(prop.eccentricity, 3),
					"Orientation [radians]": round(prop.orientation, 3),
					"Solidity": round(prop.solidity, 3),
					"Extent": round(prop.extent, 3),
					"Major Axis Length [pixels]": round(prop.major_axis_length, 3),
					"Minor Axis Length [pixels]": round(prop.minor_axis_length, 3),
					"Equivalent Diameter [pixels]": round(prop.equivalent_diameter, 3)
				})

# Convert the list of dictionaries to a DataFrame
statistics_df = pd.DataFrame(statistics)
# Export the DataFrame to a CSV file in the output_folder
output_file = os.path.join(output_folder, "image_statistics.csv")
statistics_df.to_csv(output_file, index=False)


