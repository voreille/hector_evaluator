import warnings

import numpy as np
import SimpleITK as sitk
from scipy.interpolate import RegularGridInterpolator


def dice(y_true, y_pred):
    return 2 * np.sum(np.logical_and(
        y_true, y_pred)) / (np.sum(y_true) + np.sum(y_pred))


def compute_scores(y_true, y_pred):
    if np.sum(y_pred) == 0:
        return 0, 0, 0
    else:
        fp = np.sum(np.logical_and(np.logical_not(y_true), y_pred))
        tp = np.sum(np.logical_and(y_true, y_pred))
        fn = np.sum(np.logical_and(np.logical_not(y_pred), y_true))
        tn = np.sum(
            np.logical_and(np.logical_not(y_true), np.logical_not(y_pred)))
        sensitivity = tp / (tp + fn)
        precision = tp / (tp + fp)
        return dice(y_true, y_pred), sensitivity, precision


def get_np_volume_from_sitk(sitk_image):
    trans = (2, 1, 0)
    pixel_spacing = sitk_image.GetSpacing()
    image_position_patient = sitk_image.GetOrigin()
    np_image = sitk.GetArrayFromImage(sitk_image)
    np_image = np.transpose(np_image, trans)
    return np_image, pixel_spacing, image_position_patient


def grid_from_spacing(start, spacing, n):
    return np.asarray([start + k * spacing for k in range(n)])


def resample(np_volume, origin, current_pixel_spacing, resampling_px_spacing,
             bounding_box):

    x_old = grid_from_spacing(origin[0], current_pixel_spacing[0],
                              np_volume.shape[0])
    y_old = grid_from_spacing(origin[1], current_pixel_spacing[1],
                              np_volume.shape[1])
    z_old = grid_from_spacing(origin[2], current_pixel_spacing[2],
                              np_volume.shape[2])

    output_shape = (np.ceil([
        bounding_box[3] - bounding_box[0],
        bounding_box[4] - bounding_box[1],
        bounding_box[5] - bounding_box[2],
    ]) / resampling_px_spacing).astype(int)

    x_new = grid_from_spacing(bounding_box[0], resampling_px_spacing[0],
                              output_shape[0])
    y_new = grid_from_spacing(bounding_box[1], resampling_px_spacing[1],
                              output_shape[1])
    z_new = grid_from_spacing(bounding_box[2], resampling_px_spacing[2],
                              output_shape[2])
    interpolator = RegularGridInterpolator((x_old, y_old, z_old),
                                           np_volume,
                                           method='nearest',
                                           bounds_error=False,
                                           fill_value=0)
    x, y, z = np.meshgrid(x_new, y_new, z_new, indexing='ij')
    pts = np.array(list(zip(x.flatten(), y.flatten(), z.flatten())))

    return interpolator(pts).reshape(output_shape)
