import numpy as np
from scipy import ndimage
from skimage.morphology import medial_axis
from skimage.morphology import skeletonize
from skimage.transform import hough, probabilistic_hough
import matplotlib.pyplot as plt
import pickle

import Whale as w

spec_data = w.get_spectrogram(5)
munged_data = ndimage.gaussian_filter(spec_data, sigma=1.0)
black_white = munged_data > 2.0 * munged_data.mean()
data = black_white

# Compute the medial axis (skeleton) and the distance transform
skel, distance = medial_axis(data, return_distance=True)
skel1 = skeletonize(data)

# Distance to the background for pixels of the skeleton
dist_on_skel = distance * skel
dist_on_skel1 = distance * skel1

h, theta, d = hough(skel1)
lines = probabilistic_hough(skel1, threshold=6, line_length=5, line_gap=3)
lines = sorted(lines)

plt.figure(figsize=(8, 4))
plt.subplot(141)
plt.imshow(data, cmap=plt.cm.gray, interpolation='nearest')
plt.axis('off')
plt.subplot(142)
plt.imshow(dist_on_skel, cmap=plt.cm.spectral, interpolation='nearest')
plt.contour(data, [0.5], colors='w')
plt.axis('off')
plt.subplot(143)
plt.imshow(dist_on_skel1, cmap=plt.cm.spectral, interpolation='nearest')
plt.contour(data, [0.5], colors='w')
plt.axis('off')

plt.subplot(144)
plt.imshow(skel1 * 0)

for line in lines:
    p0, p1 = line
    plt.plot((p0[0], p1[0]), (p0[1], p1[1]))

plt.title('Lines found with PHT')
plt.axis('image')

plt.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0,
                    right=1)
plt.show()

