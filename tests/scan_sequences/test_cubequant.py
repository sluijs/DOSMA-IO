import os
import shutil
import unittest

import numpy as np

import dosma.file_constants as fc
from dosma.data_io import DicomReader, ImageDataFormat, NiftiReader
from dosma.scan_sequences import CubeQuant, QDess
from dosma.tissues.femoral_cartilage import FemoralCartilage
from dosma.utils.registration import apply_warp, register

from .. import util

# target mask path used to register Cubequant volume to qDESS volume
QDESS_ECHO1_PATH = util.get_read_paths(util.get_scan_dirpath(QDess.NAME), ImageDataFormat.nifti)[0]
TARGET_MASK_PATH = os.path.join(util.get_scan_dirpath(CubeQuant.NAME), "misc/fc.nii.gz")


class CubeQuantTest(util.ScanTest):
    SCAN_TYPE = CubeQuant

    def test_interregister_no_mask(self):
        """Register Cubequant scan to qDESS scan without a target mask"""
        scan = self.SCAN_TYPE.from_dicom(self.dicom_dirpath, num_workers=util.num_workers())

        # Register to first echo of QDess without a mask
        scan.interregister(target_path=QDESS_ECHO1_PATH)

    def test_interregister_mask(self):
        """Register Cubequant scan to qDESS scan with a target mask (mask for femoral cartilage)"""
        scan = self.SCAN_TYPE.from_dicom(self.dicom_dirpath, num_workers=util.num_workers())
        scan.interregister(target_path=QDESS_ECHO1_PATH, target_mask_path=TARGET_MASK_PATH)

    def test_t1_rho_map(self):
        scan = self.SCAN_TYPE.from_dicom(self.dicom_dirpath, num_workers=util.num_workers())
        scan.interregister(target_path=QDESS_ECHO1_PATH, target_mask_path=TARGET_MASK_PATH)

        # run analysis with femoral cartilage, without mask
        tissue = FemoralCartilage()
        map1 = scan.generate_t1_rho_map(tissue, TARGET_MASK_PATH, num_workers=util.num_workers())
        assert map1 is not None, "map should not be None"

        # add mask to femoral cartilage and run
        nr = NiftiReader()
        tissue.set_mask(nr.load(TARGET_MASK_PATH))
        map2 = scan.generate_t1_rho_map(tissue, num_workers=util.num_workers())
        assert map2 is not None, "map should not be None"

        # map1 and map2 should be identical
        assert map1.volumetric_map.is_identical(map2.volumetric_map)

    def test_interregister_upgrade_no_mask(self):
        """Verify cubequant interregistering using new registration.

        To be deleted once Cubequant registration is upgraded
        (https://github.com/ad12/DOSMA/issues/55).
        """
        nr = NiftiReader()
        data_dir = os.path.join(fc.TEMP_FOLDER_PATH, "test-interregister-no-mask")

        scan = self.SCAN_TYPE.from_dicom(self.dicom_dirpath, num_workers=util.num_workers())
        intraregistered_vols = sorted(scan.volumes, key=lambda x: x.get_metadata("EchoTime", float))

        scan.interregister(target_path=QDESS_ECHO1_PATH)

        # Inter-register
        out_path = os.path.join(data_dir, "inter")
        base, moving = intraregistered_vols[0], intraregistered_vols[1:]
        out_reg, _ = register(
            QDESS_ECHO1_PATH,
            base,
            parameters=[fc.ELASTIX_RIGID_PARAMS_FILE, fc.ELASTIX_AFFINE_PARAMS_FILE],
            output_path=out_path,
            sequential=True,
            collate=True,
            num_workers=util.num_workers(),
            num_threads=2,
            return_volumes=False,
            rtype=tuple,
            show_pbar=True,
        )
        out_reg = out_reg[0]

        reg_vols = [nr.load(out_reg.warped_file)]
        for mvg in moving:
            reg_vols.append(apply_warp(mvg, out_reg.transform))

        for idx, (vol, subvol) in enumerate(zip(reg_vols, scan.volumes)):
            assert np.allclose(vol.volume, subvol.volume), idx

        shutil.rmtree(data_dir)

    def test_interregister_upgrade_mask(self):
        """Verify cubequant interregistering using new registration.

        To be deleted once Cubequant registration is upgraded
        (https://github.com/ad12/DOSMA/issues/55).
        """
        nr = NiftiReader()
        data_dir = os.path.join(fc.TEMP_FOLDER_PATH, "test-interregister-mask")

        scan = self.SCAN_TYPE.from_dicom(self.dicom_dirpath, num_workers=util.num_workers())
        intraregistered_vols = sorted(scan.volumes, key=lambda x: x.get_metadata("EchoTime", float))

        scan.interregister(target_path=QDESS_ECHO1_PATH, target_mask_path=TARGET_MASK_PATH)

        # Inter-register
        out_path = os.path.join(data_dir, "inter")
        base, moving = intraregistered_vols[0], intraregistered_vols[1:]
        mask_path = scan.__dilate_mask__(TARGET_MASK_PATH, out_path)
        out_reg, _ = register(
            QDESS_ECHO1_PATH,
            base,
            parameters=[
                fc.ELASTIX_RIGID_INTERREGISTER_PARAMS_FILE,
                fc.ELASTIX_AFFINE_INTERREGISTER_PARAMS_FILE,
            ],
            output_path=out_path,
            sequential=True,
            collate=True,
            num_workers=util.num_workers(),
            num_threads=2,
            return_volumes=False,
            target_mask=mask_path,
            use_mask=[False, True],
            rtype=tuple,
            show_pbar=True,
        )
        out_reg = out_reg[0]

        reg_vols = [nr.load(out_reg.warped_file)]
        for mvg in moving:
            reg_vols.append(apply_warp(mvg, out_reg.transform))

        for idx, (vol, subvol) in enumerate(zip(reg_vols, scan.volumes)):
            assert np.allclose(vol.volume, subvol.volume), idx

        shutil.rmtree(data_dir)

    def test_intraregister_upgrade(self):
        """Verify cubequant intraregistering using new registration.

        To be deleted once Cubequant registration is upgraded
        (https://github.com/ad12/DOSMA/issues/55).
        """
        scan = self.SCAN_TYPE.from_dicom(self.dicom_dirpath, num_workers=util.num_workers())
        scan.intraregister()

        vols = DicomReader(num_workers=util.num_workers()).load(self.dicom_dirpath)
        out_path = os.path.join(fc.TEMP_FOLDER_PATH, "test-intraregister")
        _, reg_vols = register(
            vols[0],
            vols[1:],
            fc.ELASTIX_AFFINE_PARAMS_FILE,
            out_path,
            num_workers=util.num_workers(),
            num_threads=2,
            return_volumes=True,
            rtype=tuple,
            show_pbar=True,
        )
        reg_vols = [vols[0]] + list(reg_vols)

        for idx, (vol, subvol) in enumerate(zip(reg_vols, scan.volumes)):
            assert np.allclose(vol.volume, subvol.volume), idx

        shutil.rmtree(out_path)

    def test_cmd_line(self):
        # Intraregister
        cmdline_str = "--d %s --s %s cubequant intraregister" % (
            self.dicom_dirpath,
            self.data_dirpath,
        )
        self.__cmd_line_helper__(cmdline_str)

        # Generate segmentation mask for femoral cartilage via command line
        cmdline_str = "--l %s cubequant --fc interregister --tp %s --tm %s" % (
            self.data_dirpath,
            QDESS_ECHO1_PATH,
            TARGET_MASK_PATH,
        )
        self.__cmd_line_helper__(cmdline_str)

        # Generate T1rho map for femoral cartilage, tibial cartilage, and meniscus via command line
        cmdline_str = "--l %s cubequant --fc t1_rho --mask_path %s" % (
            self.data_dirpath,
            TARGET_MASK_PATH,
        )
        self.__cmd_line_helper__(cmdline_str)


if __name__ == "__main__":
    unittest.main()
