"""
URL I/O.

This module contains HTTP input/output helpers.

"""

from io import BytesIO
import zipfile
from typing import Tuple, Union, List, Dict, Collection

import numpy as np
import requests
from tqdm.auto import tqdm

from dosma.core.io.dicom_io import DicomReader
from dosma.core.io.format_io import DataReader
from dosma.core.io.http_io_utils import is_url, extract_boundary, extract_part

__all__ = ["HttpReader"]


_MIME_TYPES_ZIP = [
    "application/zip",
    "application/x-zip-compressed",
    "multipart/x-zip",
    "application/dicom+zip"
]


class HttpReader(DataReader):
    """A class for reading DICOMs from HTTP requests.

    Attributes:
        verbose (bool, optional): If ``True``, show loading progress bar.
        block_size (int, optional): Block size for reading data.
        **kwargs: Keyword arguments for :class:`DicomReader`.

    Examples:
        >>> from dosma.core.io import HttpReader
        >>> reader = HttpReader()
        >>> reader.read("https://server.com/dicom.zip")
    """
    def __init__(
        self,
        verbose: bool = False,
        block_size: int = 10 ** 6,
        **kwargs,
    ):
        self.verbose = verbose
        self.block_size = block_size
        self.session = requests.Session()
        self.kwargs = kwargs

    def _read_multipart_stream(self, res: requests.Response, content_info: str, pbar: tqdm) -> List[bytes]:
        """Read multipart stream.

        Args:
            res (requests.Response): Response object.
            content_info (str): Content info.
            pbar (tqdm): Progress bar.
        """
        boundary = extract_boundary(content_info)
        blob, parts = bytes(), []

        for block in res.iter_content(self.block_size):
            pbar.update(len(block))

            blob += block
            while boundary in blob:
                part, blob = blob.split(boundary, maxsplit=1)
                content = extract_part(part)

                if content is not None:
                    parts.append(content)

        content = extract_part(blob)
        if content is not None:
            parts.append(content)

        pbar.close()
        return parts

    def _read_stream(self, res: requests.Response, pbar: tqdm) -> bytes:
        """Read stream.

        Args:
            res (requests.Response): Response object.
            pbar (tqdm): Progress bar.
        """

        blob = bytes()
        for block in res.iter_content(self.block_size):
            pbar.update(len(block))
            blob += block

        pbar.close()
        return blob

    def _read_dicom(self, buffers: List[bytes], **kwargs):
        """Read DICOMs from data.

        Args:
            buffers (List[bytes]): List of bytes objects.
            **kwargs: Keyword arguments for :class:`DicomReader`.
        """

        # do not pass verbose to reader, as files are already opened
        dr = DicomReader(**{**self.kwargs, **kwargs})
        return dr.read([BytesIO(buffer) for buffer in buffers])

    def load(
        self,
        url: str,
        params: Union[Dict, List[Tuple], bytes] = np._NoValue,
        **kwargs,
    ):
        """Load data from HTTP request.

        Args:
            url (str): URL.
            params (Union[Dict, List[Tuple], bytes], optional): Parameters to send with the request.
            **kwargs: Keyword arguments for :class:`DicomReader`.
        """

        if not is_url(url):
            raise IOError(f"Invalid URL: {url}.")

        params = params if params != np._NoValue else self.session.params
        with self.session.get(url, params=params, stream=True) as res:
            content_length = res.headers.get("Content-Length", 0)
            content_type = res.headers.get("Content-Type", "application/octet-stream").lower()

            pbar = tqdm(
                total=content_length,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                disable=not self.verbose,
            )

            # multipart/related
            if content_type.startswith("multipart/related;"):
                _, *content_info = [part.strip() for part in content_type.split(";")]
                parts = self._read_multipart_stream(res, content_info, pbar)
                return self._read_dicom(parts, **kwargs)

            # application/zip
            if content_type in _MIME_TYPES_ZIP:
                blob = self._read_stream(res, pbar)
                z = zipfile.ZipFile(BytesIO(blob))
                parts = [z.read(zinfo) for zinfo in z.infolist() if zinfo.file_size > 0]
                return self._read_dicom(parts, **kwargs)

            # fallback to single file
            blob = self._read_stream(res, pbar)
            return self._read_dicom([blob], **kwargs)

    def close(self):
        """Close the current HTTP session."""

        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def __serializable_variables__(self) -> Collection[str]:
        return self.__dict__.keys()

    read = load  # pragma: no cover
