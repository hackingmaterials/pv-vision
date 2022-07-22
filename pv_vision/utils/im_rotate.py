import cv2 as cv


def im_rotate(im_path, rotate_angle):
    """ rotate input images through 90 or 180 degrees

    Parameters
    ----------
    im_path: str or pathlib.PosixPath
        The file path of the original image.

    rotate_angle: int, one of [0, 1, 2]
        The angle of rotation.
        0 = 90-clockwise,
        1 = 180
        2 = 90-counter-clockwise

    Returns
    -------
    image_r: array
    The rotated image.
    """
    image = cv.imread(str(im_path))
    image_r = cv.rotate(image, rotate_angle)

    return image_r
