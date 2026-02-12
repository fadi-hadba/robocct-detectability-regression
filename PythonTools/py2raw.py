
from __future__ import annotations

from pathlib import Path

import numpy as np

from .ezrt_header import EzrtHeader

def py2raw(image_array: np.ndarray, input_header: EzrtHeader | bytes | None = None,
           switch_order: bool = False, savepath: Path | str = None):
    """
    Save an internal Python projection representation (2-dim np array) as a .raw projection file with an specified
    header.

    :param image_array: 2D numpy input array to save
    :param input_header: EzrtHeader to use (if None, a default header is generated)
    :param switch_order: toggle order of numpy array shape (True: (H, W); False (W, H))
    :param compression: compression algorithm to use (currently supported: None (=uncompressed), lz4, gzip)
    :param kwargs: further arguments for compression algorithm
    """
    if len(image_array.shape) != 2:
        raise ValueError(f'input image array has wrong dimensions ({len(image_array.shape)})')

    if switch_order:
        image_array = image_array.reshape(tuple(reversed(image_array.shape)))

    if input_header is not None:
        if isinstance(input_header, EzrtHeader):
            header = input_header
        elif isinstance(input_header, bytes) or isinstance(input_header, bytearray):
            header = EzrtHeader.frombuffer(input_header)
        else:
            raise ValueError('input header data type not supported')
    else:
        converted_bitdepth = EzrtHeader.convert_to_ezrt_bitdepth(image_array.dtype)
        header = EzrtHeader(image_array.shape[0], image_array.shape[1], converted_bitdepth, 1)

    if savepath is not None:
        data = header.tobytes() + image_array.ravel().tobytes()
        with open(savepath, 'wb') as outfile:
            outfile.write(data)

    return header, image_array  # Hier wird der Header und das Bildarray zurückgegeben



#
#
# def py2raw(image_array: np.ndarray, input_header: bytes | None = None, switch_order: bool = False):
#     """
#     Convert a 2D numpy array to .raw file format and return the raw data.
#
#     :param image_array: 2D numpy input array to convert
#     :param input_header: Header data to use (if None, a default header is generated)
#     :param switch_order: Toggle order of numpy array shape (True: (H, W); False: (W, H))
#     :return: Raw data as bytes
#     """
#     if len(image_array.shape) != 2:
#         raise ValueError(f'input image array has wrong dimensions ({len(image_array.shape)})')
#
#     if switch_order:
#         image_array = image_array.reshape(tuple(reversed(image_array.shape)))
#
#     if input_header is not None:
#         if not isinstance(input_header, bytes):
#             raise ValueError('input header data type not supported')
#         header_data = input_header
#     else:
#         converted_bitdepth = EzrtHeader.convert_to_ezrt_bitdepth(image_array.dtype)
#         header = EzrtHeader(image_array.shape[0], image_array.shape[1], converted_bitdepth, 1)
#         header_data = header.tobytes()
#
#     image_data = image_array.ravel().tobytes()
#
#     return image_data
