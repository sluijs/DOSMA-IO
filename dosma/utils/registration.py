import itertools
import multiprocessing as mp
import os
import shutil
import uuid
from functools import partial
from typing import Dict, Sequence, Union

from nipype.interfaces.elastix import ApplyWarp, Registration
from nipype.interfaces.elastix.registration import RegistrationOutputSpec
from tqdm import tqdm

from dosma import file_constants as fc
from dosma.data_io.nifti_io import NiftiWriter, NiftiReader
from dosma.data_io.med_volume import MedicalVolume

MedVolOrPath = Union[MedicalVolume, str]


def register(
    target: MedVolOrPath,
    moving: Union[MedVolOrPath, Sequence[MedVolOrPath]],
    parameters: Union[str, Sequence[str]],
    output_path: str,
    target_mask: MedVolOrPath = None,
    moving_masks: Union[MedVolOrPath, Sequence[MedVolOrPath]] = None,
    sequential: bool = False,
    collate: bool = True,
    num_workers: int = 0,
    num_threads: int = 1,
    show_pbar: bool = False,
    return_volumes: bool = False,
    rtype: type = dict,
    **kwargs,
):
    """Register moving image(s) to the target.

    `MedVolOrPath` is a shorthand for `MedicalVolume` or `str`. It indicates the argument 
    can be either a `MedicalVolume` or a `str` path to a nifti file.

    Args:
        target (`MedicalVolume` or `str`): The target/fixed image.
        moving (`MedicalVolume`(s) or `str`(s)): The moving/source image(s).
        parameters (`str(s)`): Elastix parameter files to use.
        output_path (`str`): Output directory to store files.
        target_mask (`MedicalVolume` or `str`, optional): The target/fixed mask.
        moving_masks (`MedicalVolume`(s) or `str`(s), optional): The moving mask(s).
            If only one specified, the mask will be used for all moving images.
        sequential (bool, optional): If `True`, apply parameter files sequentially.
        collate (bool, optional): If `True`, will collate outputs from sequential registration
            into single RegistrationOutputSpec instance. If `sequential=False`, this argument
            is ignored.
        num_workers (int, optional): Number of workers to use for reading/writing data and f
            or registration. Note this is not used for registration, which is done via 
            multiple threads - see `num_threads` for more details.
        num_threads (int, optional): Number of threads to use for registration. If `None`, defaults to 1.
        show_pbar (bool, optional): If `True`, show progress bar during registration. Note the progress bar
            will not be shown for intermediate reading/writing.
        return_volumes (bool, optional): If `True`, registered volumes will also be returned.
            By default, only the output namespaces (RegistrationOutputSpec) of the registrations are
            returned.
        rtype (type, optional): The return type. Either `dict` or `tuple`.
        kwargs: Keyword arguments used to initialize `nipype.interfaces.elastix.Registration`.

    Returns:
        Dict or Tuple: Type specified by `rtype`. If dict, with keys 'outputs' (registration outputs) and 'volumes' (final volumes)
            if `return_volumes=True`). If tuple, order is (`outputs`, `volumes` or `None`).
            Length of `outputs` and `volumes` depends on number of images specified in `moving`.

            outputs (Sequence[RegistrationOutputSpec]): The output objects from 
                elastix registration, one for each moving image. Each object is effectively 
                a namespace with four main attributes:
                    - 'transform' (List[str]): Paths to transform files produced using registration.
                    - 'warped_file' (str): Path to the final registered image.
                    - 'warped_files' (List[str]): Paths to all intermediate images created if multiple
                        parameter files used.
            volumes (Sequence[MedicalVolume]): Registered volumes.
    """
    assert issubclass(rtype, (Dict, Sequence))  # `rtype` must be dict or tuple
    has_output_path = bool(output_path)
    if not output_path:
        output_path = os.path.join(fc.TEMP_FOLDER_PATH, "register")

    moving = [moving] if isinstance(moving, (MedicalVolume, str)) else moving
    moving_masks = [moving_masks] if moving_masks is None or isinstance(moving_masks, (MedicalVolume, str)) else moving_masks
    if len(moving_masks) > 1 and len(moving) != len(moving_masks):
        raise ValueError("Got {} moving images but {} moving masks".format(len(moving), len(moving_masks)))

    files = [target, target_mask] + moving + moving_masks

    # Write medical volumes (if any) to nifti file for use with elastix.
    tmp_dir = os.path.join(output_path, "tmp")
    default_files = ["target", "target-mask"] + [f"moving-{idx}" for idx in range(len(moving))] + [f"moving-mask-{idx}" for idx in range(len(moving_masks))]  #noqa
    assert len(default_files) == len(files), default_files  # should be 1-to-1 with # args provided
    vols = [(idx, v) for idx, v in enumerate(files) if isinstance(v, MedicalVolume)]
    idxs, vols = [x[0] for x in vols], [x[1] for x in vols]
    if len(vols) > 0:
        filepaths = [os.path.join(tmp_dir, f"{default_files[idx]}.nii.gz") for idx in idxs]
        if num_workers > 0:
            with mp.Pool(min(num_workers, len(vols))) as p:
                out = p.starmap_async(_write, zip(vols, filepaths))
                out.wait()
        else:
            for vol, fp in zip(vols, filepaths):
                _write(vol, fp)
        for idx, fp in zip(idxs, filepaths):
            files[idx] = fp
    
    # Assign file paths to respective variables.
    target, moving = files[0], files[2:2+len(moving)]
    target_mask, moving_masks = files[1], files[2+len(moving):]
    if len(moving_masks) == 1:
        moving_masks = moving_masks * len(moving)
    
    all_outputs = {}

    # Perform registration.
    out = []
    for idx, (mvg, mvg_mask) in tqdm(
        enumerate(zip(moving, moving_masks)), disable=not show_pbar, total=len(moving)
    ):
        out_path = os.path.join(output_path, f"moving-{idx}")
        _out = _elastix_register(
            target, mvg, parameters, out_path, target_mask, 
            mvg_mask, sequential, collate, num_threads, **kwargs,
        )
        out.append(_out)
    all_outputs["outputs"] = tuple(out)

    # Load volumes.
    if return_volumes:
        filepaths = [x[-1].warped_file if isinstance(x, Sequence) else x.warped_file for x in out]
        if num_workers > 0:
            with mp.Pool(min(num_workers, len(filepaths))) as p:
                vols = p.map(_read, filepaths)
        else:
            for fp in filepaths:
                vols = _read(fp)
        all_outputs["volume"] = tuple(vols)
    
    # Clean up.
    for _dir in [tmp_dir, output_path if not has_output_path else None]:
        if not _dir or not os.path.isdir(_dir):
            continue
        shutil.rmtree(_dir)
    
    if issubclass(rtype, dict):
        out = rtype(all_outputs)
    elif issubclass(rtype, Sequence):
        out = rtype([all_outputs["outputs"], all_outputs.get("volume", None)])
    else:
        assert False  # Should have type checking earlier.

    return out


def apply_warp(
    moving: MedVolOrPath,
    transform: Union[str, Sequence[str]] = None,
    out_registration: RegistrationOutputSpec = None,
    output_path: str = None,
    rtype: type = MedicalVolume,
    num_threads: int = 1,
    show_pbar: bool = False,
) -> MedVolOrPath:
    """Apply transform(s) to moving image using transformix.

    Use transformix to apply a transform on an input image. The transform(s) is/are
    specified in the transform-parameter file(s).

    Args:
        moving (MedicalVolume(s) or str(s)): The moving/source image to transform.
        transform (str(s)): Paths to transform files to be used by transformix.
            If multiple files provided, transforms will be applied sequentially.
            If `None`, will be determined by `out_registration.transform`.
        out_registration (RegistrationOutputSpec(s)): Outputs from elastix registration
            using nipype. Must be specified if `transform` is None.
        output_path (str): Output directory to store files.
        rtype (type, optional): Return type - either `MedicalVolume` or `str`.
            If `str`, `output_path` must be specified. Defaults to `MedicalVolume`.
        num_threads (int, optional): Number of threads to use for registration. If `None`, defaults to 1.
        show_pbar (bool, optional): If `True`, show progress bar when applying transforms. 

    Return:
        MedVolOrPath: The medical volume or nifti file corresponding to the volume.
            See `rtype` for details.
    """
    assert rtype in [MedicalVolume, str], rtype  # rtype must be MedicalVolume or str
    has_output_path = bool(output_path)
    if rtype == str and not has_output_path:
        raise ValueError("`output_path` must be specified when `rtype=str`")
    if not output_path:
        output_path = os.path.join(fc.TEMP_FOLDER_PATH, f"apply_warp-{str(uuid.uuid1())}")
    os.makedirs(output_path, exist_ok=True)

    if not transform:
        transform = out_registration.transform
    elif not isinstance(transform, Sequence):
        transform = [transform]

    mv_filepath = os.path.join(output_path, "moving.nii.gz")
    if isinstance(moving, MedicalVolume):
        NiftiWriter().save(moving, mv_filepath)
        moving = mv_filepath
    
    for tf in tqdm(transform, disable=not show_pbar):
        reg = ApplyWarp()
        reg.inputs.moving_image = moving
        reg.inputs.transform_file = tf
        reg.inputs.output_path = output_path
        reg.terminal_output = fc.NIPYPE_LOGGING
        reg.inputs.num_threads = num_threads
        reg_output = reg.run()

        moving = reg_output.outputs.warped_file

    if rtype == MedicalVolume:
        out = NiftiReader().load(moving)
    else:
        out = moving

    if os.path.isfile(mv_filepath):
        os.remove(mv_filepath)
    if not has_output_path:
        shutil.rmtree(output_path)

    return out


def _elastix_register(
    target: str, moving: str, parameters: Sequence[str], output_path: str,
    target_mask: str = None, moving_mask: str=None, sequential=False, collate=True,
    num_threads=None, use_mask: Sequence[bool] = None, **kwargs,
):
    def _register(_moving, _parameters, _output_path, _use_mask=None):
        if isinstance(_parameters, str):
            _parameters = [_parameters]
        if _use_mask is None:
            _use_mask = target_mask is not None or moving_mask is not None

        os.makedirs(_output_path, exist_ok=True)
        reg = Registration()
        reg.inputs.fixed_image = target
        reg.inputs.moving_image = _moving
        reg.inputs.parameters = _parameters
        reg.inputs.output_path = _output_path
        reg.terminal_output = fc.NIPYPE_LOGGING
        if num_threads:
            reg.inputs.num_threads = num_threads
        if _use_mask and target_mask is not None:
            reg.inputs.fixed_mask = target_mask
        if _use_mask and moving_mask is not None:
            reg.inputs.target_mask = moving_mask
        for k, v in kwargs.items():
            setattr(reg.inputs, k, v)
        
        return reg.run().outputs
    
    def _collate_outputs(_outs):
        """Concatenates fields that are sequential and takes final output for fields that are not."""
        if len(_outs) == 1:
            return _outs[0]

        _result = _outs[0]
        fields = list(_outs[0].__dict__.keys())
        for _fld in fields:
            _res_val = getattr(_result, _fld)
            if not isinstance(_res_val, str) and isinstance(_res_val, Sequence):
                val = list(itertools.chain.from_iterable([getattr(x, _fld) for x in _outs]))
            else:
                val = getattr(_outs[-1], _fld)
            setattr(_result, _fld, val)
        return _result

    if use_mask is not None:
        assert sequential  # use_mask can only be specified when sequential is specified
    if sequential:
        outs, mvg = [], moving
        for idx, param in enumerate(parameters):
            _use_mask = None if use_mask is None else use_mask[idx]
            _out = _register(mvg, param, os.path.join(output_path, f"param{idx}"), _use_mask)
            outs.append(_out)
            mvg = _out.warped_file
        out = _collate_outputs(outs) if collate else outs
        return out
    else:
        return _register(moving, parameters, output_path)

def _write(vol: MedicalVolume, path: str):
    """Extracted out for multiprocessing purposes."""
    NiftiWriter().save(vol, path)


def _read(path: str):
    return NiftiReader().load(path)
