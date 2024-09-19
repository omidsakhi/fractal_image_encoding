import numpy as np

def downsample(block, scale):
    if scale == 1:
        return block
    return block.reshape(block.shape[0]//scale, scale, block.shape[1]//scale, scale).mean(axis=(1,3))

def create_domain_pyramid(image):
    pyramid = []
    dn2 = downsample(image, 2)
    dn4 = downsample(dn2, 2)
    dn8 = downsample(dn4, 2)
    pyramid.append(dn2)
    pyramid.append(np.rot90(dn2))
    pyramid.append(np.rot90(dn2, 2))
    pyramid.append(np.rot90(dn2, 3))
    pyramid.append(np.flipud(dn2))
    pyramid.append(np.rot90(np.flipud(dn2)))
    pyramid.append(np.rot90(np.flipud(dn2), 2))
    pyramid.append(np.rot90(np.flipud(dn2), 3))
    pyramid.append(dn4)
    pyramid.append(np.rot90(dn4))
    pyramid.append(np.rot90(dn4, 2))
    pyramid.append(np.rot90(dn4, 3))
    pyramid.append(np.flipud(dn4))
    pyramid.append(np.rot90(np.flipud(dn4)))
    pyramid.append(np.rot90(np.flipud(dn4), 2))
    pyramid.append(np.rot90(np.flipud(dn4), 3))
    pyramid.append(dn8)
    pyramid.append(np.rot90(dn8))
    pyramid.append(np.rot90(dn8, 2))
    pyramid.append(np.rot90(dn8, 3))
    pyramid.append(np.flipud(dn8))
    pyramid.append(np.rot90(np.flipud(dn8)))
    pyramid.append(np.rot90(np.flipud(dn8), 2))
    pyramid.append(np.rot90(np.flipud(dn8), 3))
    return pyramid
