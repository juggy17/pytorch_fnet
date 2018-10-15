import sys
sys.path.insert(0, '../../czifile/czifile/')

from czifile import CziFile
import matplotlib.pyplot as plt
import numpy as np

with CziFile('data/3500000427_100X_20170120_F05_P27.czi') as czi:
    image_arrays = czi.asarray()
print(image_arrays.shape)

images = [image_arrays[0,0,0,idx] for idx in range(39)]

for i in range(3):
    plt.figure(i+1)
    
    ax = plt.subplot(311)
    plt.tight_layout()
    ax.set_title('sample #{:d}'.format(i*3))
    ax.axis('off')
    plt.imshow(np.squeeze(images[i*3]))
    
    ax = plt.subplot(312)
    plt.tight_layout()
    ax.set_title('sample #{:d}'.format(i*3+1))
    ax.axis('off')
    plt.imshow(np.squeeze(images[i*3+1]))
        
    ax = plt.subplot(313)
    plt.tight_layout()
    ax.set_title('sample #{:d}'.format(i*3+2))
    ax.axis('off')    
    plt.imshow(np.squeeze(images[i*3+2]))

plt.show(block=True)
