"""
URL I/O.

This module contains HTTP input/output helpers.

"""

from io import BytesIO
import zipfile
from typing import Tuple, Union, List, Dict, Collection, Optional, Iterator

import numpy as np
import requests
from tqdm.auto import tqdm

from dosma.core.io.dicom_io import DicomReader
from dosma.core.io.format_io import DataReader
from dosma.core.io.container_io import ImageContainerFormat
from dosma.core.io.container_io_utils import get_reader
from dosma.core.io.http_io_utils import is_url

__all__ = ["HttpReader"]


_MIME_TYPES_ZIP = [
    "application/zip",
    "application/x-zip-compressed",
    "multipart/x-zip",
    "application/dicom+zip"
]


def _extract_part(part: bytes) -> Union[bytes, None]:
    if part in [b"", b"--", b"\r\n"] or part.startswith(b"--\r\n"):
        return None

    idx = part.index(b"\r\n\r\n")
    if idx > -1:
        return part[idx + 4:]

    raise ValueError("Part is not CRLF CRLF terminated.")

def extract_boundary(content_info: List[str]) -> Optional[bytes]:
    for item in content_info:
        if '=' not in item:
            continue

        key, value = item.split('=', maxsplit=1)
        if key.lower() == "boundary":
            return b"--" + value.strip('"').encode("utf-8")

    return None

def read_multipart(
    res: requests.Response,
    boundary: bytes,
) -> Iterator[bytes]:

    delimiter = b"\r\n" + boundary
    data = b""

    j = 0
    it = res.iter_content(chunk_size=chunk_size)

    for i, chunk in enumerate(it):
        data += chunk
        while delimiter in data:
            part, data = data.split(delimiter, maxsplit=1)
            content = _extract_part(part)

            j += 1
            if content is not None:
                yield content

    content = _extract_part(data)
    if content is not None:
        yield content


class HttpReader(DataReader):
    """A class for reading DICOMs from HTTP requests.

    Attributes:
        verbose (bool, optional): If ``True``, show loading progress bar.

    Examples:

    """

    def __init__(
        self,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Args:
            verbose (bool, optional): If ``True``, show loading progress bar.
        """

        self.verbose = verbose
        self.session = requests.Session()
        self.kwargs = kwargs

    def _read_dicom(self, buffers: List[bytes], **kwargs):
        """Read DICOMs from data."""

        dr = DicomReader(**{**self.kwargs, "verbose": self.verbose, **kwargs})
        return dr.read([BytesIO(buffer) for buffer in buffers])

    def load(
        self,
        url: str,
        params: Union[Dict, List[Tuple], bytes] = np._NoValue,
        block_size: int = 10 ** 6,
        **kwargs,
    ):
        """Load data from HTTP request.

        Args:
            url (str): URL.
            params (Union[Dict, List[Tuple], bytes], optional): Parameters to send with the request.
            **kwargs: Additional keyword arguments passed to the DicomReader.
        """

        if not is_url(url):
            raise IOError(f"Invalid URL: {url}.")

        params = params if params != np._NoValue else self.session.params
        with self.session.get(url, params=params, stream=True) as res:
            content_length = res.headers.get("Content-Length", 0)
            content_type = res.headers.get("Content-Type", "application/octet-stream").lower()

            progress_bar = tqdm(
                total=content_length,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                disable=not self.verbose,
            )

            if content_type.startswith("multipart/related;"):
                _, *content_info = [part.strip() for part in content_type.split(";")]
                boundary = extract_boundary(content_info)
                blob, parts = bytes(), []

                for block in res.iter_content(block_size):
                    progress_bar.update(len(block))

                    blob += block
                    while boundary in blob:
                        part, blob = blob.split(boundary, maxsplit=1)
                        content = _extract_part(part)

                        if content is not None:
                            parts.append(content)

                content = _extract_part(blob)
                if content is not None:
                    parts.append(content)

                progress_bar.close()
                return self._read_dicom(parts, **kwargs)

            if content_type in _MIME_TYPES_ZIP:
                blob = BytesIO()
                for block in res.iter_content(block_size):
                    progress_bar.update(len(block))
                    blob.write(block)

                z = zipfile.ZipFile(blob)
                parts = [z.read(zinfo) for zinfo in z.infolist() if zinfo.file_size > 0]

                progress_bar.close()
                return self._read_dicom(parts, **kwargs)

            blob = bytes()
            for block in res.iter_content(block_size):
                progress_bar.update(len(block))
                blob += block

            return self._read_dicom([blob], **kwargs)

    # def load(
    #     self,
    #     url: str,
    #     params: Union[Dict, List[Tuple], bytes] = np._NoValue,
    #     **kwargs,
    # ):
    #     """Load data from HTTP request.

    #     Args:
    #         url (str): URL.
    #         params (Union[Dict, List[Tuple], bytes], optional): Parameters to send with the request.
    #         **kwargs: Additional keyword arguments passed to the DicomReader.
    #     """

    #     if not is_url(url):
    #         raise IOError(f"Invalid URL: {url}.")

    #     params = params if params != np._NoValue else self.session.params
    #     with self.session.get(url, params=params, stream=True) as res:
    #         content_length = res.headers.get("Content-Length", 0)
    #         content_type = res.headers.get("Content-Type", "application/octet-stream")

    #         if self.verbose:
    #             res.raw.read = partial(res.raw.read, decode_content=True)
    #             with tqdm.wrapattr(res.raw, "read", total=content_length) as raw:
    #                 container_or_file = BytesIO()
    #                 shutil.copyfileobj(raw, container_or_file)

    #                 return self._handle_blob(container_or_file, content_type, **kwargs)
    #         else:
    #             container_or_file = BytesIO()
    #             shutil.copyfileobj(res.raw, container_or_file)

    #             return self._handle_blob(container_or_file, content_type, **kwargs)

    def close(self):
        """Close the current HTTP session."""

        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self):
        self.close()

    def __serializable_variables__(self) -> Collection[str]:
        return self.__dict__.keys()

    read = load  # pragma: no cover
