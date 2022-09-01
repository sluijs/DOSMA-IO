from typing import List, Tuple, Sequence, Dict, Optional, Any
from collections.abc import MutableMapping
import json

import numpy as np
import pydicom
from pydicom.datadict import tag_for_keyword, dictionary_VR

from dosma.core import orientation as stdo
from dosma.defaults import AFFINE_DECIMAL_PRECISION, SCANNER_ORIGIN_DECIMAL_PRECISION

__all__ = ["to_RAS_affine", "DatasetProxy", "compress_headers", "decompress_header"]


def _concat_headers(headers: List[Dict]):
    out = {}

    for header in headers:
        for key in header:
            # remove empty tags and inline binaries
            if not "Value" in header.get(key):
                continue

            # essential elements
            vr = header.get(key).get("vr")
            value = header.get(key).get("Value")

            # create a new tag in the compressed header
            if not key in out:
                out[key] = { "vr": vr, "Value": [] }

            # add the value to the tag in the compressed header
            out[key]["Value"].append(value)

    return out


def _compress_value(value: List):
    val = np.array([json.dumps(v) for v in value], dtype="object")

    if np.all(val == val[0]):
        return [value[0]]

    return value


def compress_headers(headers: List[dict]):
    """Top-level header compression: a compression algorithm for DICOM+JSON headers.

    A utility function to compress top-level attributes from a list of DICOM+JSON headers. Only top-
    level attributes are compressed, as sequences are arrays with undetermined order in JSON.

    NB: attributes without "Value" properties are not preserved (eg, empty items, inlineBinary),
    making this a lossy compression algorithm.

    Args:
        headers (List[Dict]): Headers in DICOM+JSON format.

    Returns:
        Dict: a compressed representation of a list of DICOM+JSON headers.
    """
    out = _concat_headers(headers)

    for key in out:
        out[key]["Value"] = _compress_value(out[key]["Value"])

    out["__compressor__"] = "tlc"
    out["__len__"] = len(headers)

    return out


def _decompress_value(value: List, repeats: int):
    if len(value) == 1:
        return value * repeats

    return value


def decompress_header(compressed_header: Dict) -> List[Dict]:
    """Decompress a top-level compressed header into a list of DICOM+JSON headers.

    Args:
        compressed_header (Dict): top-level compressed header.

    Returns:
        headers (Lists[Dict]): Headers in DICOM+JSON format.
    """

    # create empty headers
    n_headers = compressed_header.get("__len__")
    headers = [{} for _ in range(n_headers)]

    for key in compressed_header:
        if key in ["__len__", "__compressor__"]:
            continue

        # essential elements
        vr = compressed_header.get(key).get("vr")
        value = compressed_header.get(key).get("Value")

        for i in range(n_headers):
            idx = 0 if len(value) == 1 else i
            print(value[idx])
            headers[i][key] = { "vr": vr, "Value": value[idx]}

    return headers


class DatasetProxy:
    """Performant partial implementation of the pydicom.FileDataset metadata interface.

    This class is a wrapper around a DICOM+JSON dict that acts as pydicom.FileDataset. It supports
    accessing attributes by DICOM keywords.

    Example:
    >>> DatasetProxy(dicom_json_header).RescaleIntercept

    Args:
        header (Dict): Header in DICOM+JSON format.
    """

    def __init__(self, header: Dict):
        super().__init__()

        self._dict: MutableMapping[str, Dict] = header

    def _format_json_tag(self, keyword: str) -> str:
        """Convert a DICOM tag's keyword to its corresponding hex value."""

        tag = tag_for_keyword(keyword)
        if tag is None:
            raise AttributeError(f"Keyword `{keyword} was not found in the data dictionary.")

        json_tag = hex(tag)[2:].zfill(8).upper()
        if not json_tag in self._dict:
            raise AttributeError(f"Tag `{json_tag}` was not found in this header.")

        return json_tag

    def _set_json_tag(self, keyword: str, value):
        """Set the value of a DICOM+JSON tag."""

        json_tag = self._format_json_tag(keyword)

        vr = dictionary_VR(keyword)
        value = value if isinstance(value, list) else [value]
        self._dict[json_tag] = { "vr": vr, "Value": value }

    def __contains__(self, keyword: str):
        try:
            _ = self._format_json_tag(keyword)
            return True
        except AttributeError:
            return False

    def __getattr__(self, keyword: str):
        json_tag = self._format_json_tag(keyword)
        value = self._dict[json_tag]['Value']

        if isinstance(value, list):
            if len(value) == 1:
                return value[0]

        return value

    def __getitem__(self, keyword: str):
        try:
            return self.__getattr__(keyword)
        except AttributeError:
            raise KeyError

    def __delitem__(self, keyword: str):
        try:
            json_tag = self._format_json_tag(keyword)
            del self._dict[json_tag]

        except AttributeError:
            raise KeyError

    def __iter__(self):
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def __setitem__(self, keyword: str, value):
        self._set_json_tag(keyword, value)

    def __array__(self):
        return np.asarray(self._dict)

    def get(self, keyword: str, value: Optional[Any] = None):
        try:
            return self.__getitem__(keyword)
        except KeyError:
            return value

    def __repr__(self):
        return f"{self.__class__.__name__}(elements={len(self)})"


def to_RAS_affine(headers: List[DatasetProxy], default_ornt: Tuple[str, str] = None):
    """Convert from LPS+ orientation (default for DICOM) to RAS+ standardized orientation.

    Args:
        headers (list[pydicom.FileDataset]): Headers for DICOM files to reorient.
            Files should correspond to single volume.

    Returns:
        np.ndarray: Affine matrix.
    """
    try:
        im_dir = headers[0].ImageOrientationPatient
    except AttributeError:
        im_dir = _decode_inplane_direction(headers, default_ornt=default_ornt)
        if im_dir is None:
            raise RuntimeError("Could not determine in-plane directions from headers.")
    try:
        in_plane_pixel_spacing = headers[0].PixelSpacing
    except AttributeError:
        try:
            in_plane_pixel_spacing = headers[0].ImagerPixelSpacing
        except AttributeError:
            raise RuntimeError(
                "Could not determine in-plane pixel spacing from headers. "
                "Neither attribute 'PixelSpacing' nor 'ImagerPixelSpacing' found."
            )

    orientation = np.zeros([3, 3])

    # Determine vector for in-plane pixel directions (i, j).
    i_vec, j_vec = (
        np.asarray(im_dir[:3]).astype(np.float64),
        np.asarray(im_dir[3:]).astype(np.float64),
    )  # unique to pydicom, please revise if using different library to load dicoms
    i_vec, j_vec = (
        np.round(i_vec, AFFINE_DECIMAL_PRECISION),
        np.round(j_vec, AFFINE_DECIMAL_PRECISION),
    )
    i_vec = i_vec * in_plane_pixel_spacing[0]
    j_vec = j_vec * in_plane_pixel_spacing[1]

    # Determine vector for through-plane pixel direction (k).
    # Compute difference in patient position between consecutive headers.
    # This is the preferred method to determine the k vector.
    # If single header, take cross product between i/j vectors.
    # These actions are done to avoid rounding errors that might result from float subtraction.
    if len(headers) > 1:
        k_vec = np.asarray(headers[1].ImagePositionPatient).astype(np.float64) - np.asarray(
            headers[0].ImagePositionPatient
        ).astype(np.float64)
    else:
        slice_thickness = headers[0].get("SliceThickness", 1.0)
        i_norm = 1 / np.linalg.norm(i_vec) * i_vec
        j_norm = 1 / np.linalg.norm(j_vec) * j_vec
        k_norm = np.cross(i_norm, j_norm)
        k_vec = k_norm / np.linalg.norm(k_norm) * slice_thickness
        if hasattr(headers[0], "SpacingBetweenSlices") and headers[0].SpacingBetweenSlices < 0:
            k_vec *= -1
    k_vec = np.round(k_vec, AFFINE_DECIMAL_PRECISION)

    orientation[:3, :3] = np.stack([j_vec, i_vec, k_vec], axis=1)
    scanner_origin = headers[0].get("ImagePositionPatient", np.zeros((3,)))
    scanner_origin = np.asarray(scanner_origin).astype(np.float64)
    scanner_origin = np.round(scanner_origin, SCANNER_ORIGIN_DECIMAL_PRECISION)

    affine = np.zeros([4, 4])
    affine[:3, :3] = orientation
    affine[:3, 3] = scanner_origin
    affine[:2, :] = -1 * affine[:2, :]
    affine[3, 3] = 1

    affine[affine == 0] = 0

    return affine


def _decode_inplane_direction(headers: Sequence[pydicom.FileDataset], default_ornt=None):
    """Helper function to decode in-plane direction from header(s).

    Recall the direction in dicoms are in cartesian order ``(x,y)``,
    but numpy/dosma are in matrix order ``(y,x)``. When adding new
    methods, make sure to account for this.

    Returns:
        np.ndarray: 6-element LPS direction array where first 3 elements define
            direction for x-direction (columns) and second 3 elements define
            direction for y-direction (rows)
    """
    _patient_ornt_to_nib = {"H": "S", "F": "I"}

    if (
        len(headers) == 1
        and hasattr(headers[0], "PatientOrientation")
        and headers[0].PatientOrientation
    ):
        # Decoder: patient orientation.
        # Patient orientation is only decoded along principal direction (e.g. "FR" -> "F").
        ornt = [
            _patient_ornt_to_nib.get(k[:1], k[:1]) for k in headers[0].PatientOrientation
        ]  # (x,y)
        ornt = stdo.orientation_nib_to_standard(ornt)
        affine = stdo.to_affine(ornt)
        affine[:2, :] = -1 * affine[:2, :]
        return np.concatenate([affine[:3, 0], affine[:3, 1]], axis=0)

    if default_ornt:
        affine = stdo.to_affine(default_ornt)
        affine[:2, :] = -1 * affine[:2, :]
        return np.concatenate([affine[:3, 0], affine[:3, 1]], axis=0)

    return None
