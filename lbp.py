import numpy as np


def circular_lbp(img, radius=1, neighbors=8):
    '''Circular LBP implementation

    Ref:
    Philipp Wagner
    https://github.com/bytefish/facerec/blob/master/py/facerec/lbp.py
    '''
    # make sure that you always have one
    # neighbor and no more than 31
    neighbors = max(min(neighbors, 31), 1)
    rows, cols = img.shape

    # get the circle
    angles = 2 * np.pi / neighbors
    theta = np.arange(0, 2*np.pi, angles)

    # get sample point locations around center
    sample_points = np.array([-np.sin(theta), np.cos(theta)]).T
    sample_points *= radius

    # Get bounding box of samples
    miny = min(sample_points[:, 0])
    maxy = max(sample_points[:, 0])
    minx = min(sample_points[:, 1])
    maxx = max(sample_points[:, 1])

    # calculate block size, each LBP code is
    # computed within a block of size bsizey*bsizex
    blocksizey = np.ceil(max(maxy, 0)) - np.floor(min(miny, 0)) + 1
    blocksizex = np.ceil(max(maxx, 0)) - np.floor(min(minx, 0)) + 1

    # Staring point
    origy = 0 - np.floor(min(miny, 0))
    origx = 0 - np.floor(min(minx, 0))

    # output image size
    dx = cols - blocksizex + 1
    dy = rows - blocksizey + 1

    # Center points
    C = np.asarray(img[origy:origy+dy, origx:origx+dx], dtype=np.uint8)

    # image with LBPs
    result = np.zeros((dy, dx), dtype=np.int32)

    # loop through every sample and
    # compare with all center points in image
    for i, p in enumerate(sample_points):
        # get coordinate of sample points
        y, x = p + (origy, origx)

        # Calculate floors, ceils and rounds for the x and y.
        # these are points used to interpolate
        #
        #  fx,fy|--------| cx,fy
        #       | *(x,y) |
        #       |        |
        #  fx,cy|--------| cx,cy
        #
        fx = np.floor(x)
        fy = np.floor(y)
        cx = np.ceil(x)
        cy = np.ceil(y)

        # calculate fractional part if there is one
        ty = y - fy
        tx = x - fx

        # calculate  bilinear interpolation weights
        # for the sample point
        w1 = (1 - tx) * (1 - ty)
        w2 = tx * (1 - ty)
        w3 = (1 - tx) * ty
        w4 = tx * ty

        # calculate interpolated image
        N = w1 * img[fy:fy + dy, fx:fx + dx]    # top left
        N += w2 * img[fy:fy + dy, cx:cx + dx]   # top right
        N += w3 * img[cy:cy + dy, fx:fx + dx]   # bottom left
        N += w4 * img[cy:cy + dy, cx:cx + dx]   # bottom right

        # update LBP codes
        D = N >= C
        result += (1 << i) * D
    return result
