# Saves FITS PNG image
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

filepath = sys.argv[1]

# Open fits image
image_file = open(filepath, "rb")

hdu_list = fits.open(image_file)
hdu_list.info()
image_data = hdu_list[0].data

print(type(image_data))
print("Resolution: ", image_data.shape)

fig = plt.imshow(image_data, cmap='gray')
#plt.colorbar()

# Save full image
print(np.nanmin(image_data), np.nanmax(image_data))
plt.imsave(f"{filepath}.png", image_data, cmap='gray', vmin=np.nanmin(image_data), vmax=np.nanmax(image_data))

hdu_list.close()
