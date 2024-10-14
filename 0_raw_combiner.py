# ======== libraries ================================================================= #
import sys
import os
import os.path as path
import numpy as np
from numpy.typing import NDArray
# from typing import List

# ======== natural sorting =========================================================== #
import re  # needed for sorting


def atof(text: str) -> float | str:
    """Try to convert string to float."""
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval


def natural_keys(text: str) -> list[str | float]:
    """Sort list of string in human order."""
    # return [atof(c) for c in re.split(r"[_n.]", text)]
    return [atof(c) for c in re.split(r"[_n.]", text.lower())]  # make it case insensetive


# ======== CLI arguments ============================================================= #
cli_args = {each_arg.split("=")[0]: each_arg.split("=")[1] for each_arg in sys.argv[1:] if each_arg.count("=") == 1}


def bad_cli_args(err):
    print("use: python3 ./0_raw_combiner.py ddir=/path/to/data/directory mtype=t1")
    if err == 0:
        print("Wrong CLI usage: no ddir")
    if err == 1:
        print("Wrong CLI usage: no measurment mtype")
        print("mtype=t1")
        print("mtype=t2")
        print("mtype=t3")
    if err == 2:
        print("Data directory doesn't exist")
    sys.exit(1)


# ======== data combiner class definition ============================================= #
class DataCombiner:
    angl90offset = np.arange(2.567, 2.567 - 1 / 17.53 * 90, -1 / 17.53).reshape(1, 90)
    wave_offset = 107
    fov_offset = 230

    def __init__(self, indir: str, outdir: str, mtype: str, has_2nd_dev: bool = False) -> None:
        self.indir = indir
        self.outdir = outdir
        self.mtype = mtype
        self.has_2nd_dev = has_2nd_dev
        self.num_M = 0

        # ---- files for t1 ------------------------------------------------------------
        self.t1_im0_npy_files = []
        self.t1_im2_npy_files = []
        self.t1_ori_npy_files = []
        # ---- files in t2 or t3 -------------------------------------------------------
        self.t2_or_t3_npz_files = []

        # ---- checking input and output directory--------------------------------------
        assert mtype in ("t1", "t2", "t3"), bad_cli_args(1)
        assert os.path.isdir(self.indir), bad_cli_args(0)
        if not os.path.isdir(outdir):  # check the output directory exists
            print(f"creating direcotory: {os.path.isdir(outdir)}")
            os.makedirs(outdir, exist_ok=True)

        if mtype == "t3":
            self.fov_offset = 40

        if mtype == "t1":
            self.wave_offset = 107 + 15
        elif mtype == "t2":
            self.wave_offset = 109 + 15
        elif mtype == "t3":
            self.wave_offset = 34 + 15

    def organize_filenames_t1(self) -> None:
        files = os.listdir(self.indir)
        if not self.mtype == "t1":
            print("wrong usage")

        self.t1_ori_npy_files = sorted(
            (path.join(self.indir, file) for file in files if "ori" in file),
            key=natural_keys,
        )
        self.t1_im0_npy_files = sorted(
            (path.join(self.indir, file) for file in files if "img_el0" in file),
            key=natural_keys,
        )
        if self.has_2nd_dev:
            self.t1_im2_npy_files = sorted(
                (path.join(self.indir, file) for file in files if "img_el2" in file),
                key=natural_keys,
            )

    def organize_filenames_t2_or_t3(self) -> None:
        files = os.listdir(self.indir)
        self.t2_or_t3_npz_files = sorted(
            (path.join(self.indir, file) for file in files if ".npz" in file),
            key=natural_keys,
        )
        #print(self.t2_or_t3_npz_files)

    def count_M_whole_measurement_frame_num_t1(self) -> int:
        """Read all measuremetn data for prepping numpy-array later"""
        m = 0
        for i, ori_file_path in enumerate(self.t1_ori_npy_files):
            tmp_m = np.load(ori_file_path).shape[0]
            m += tmp_m
            print(f"{i}\t{ori_file_path} has {tmp_m} frames\ttotal={m}")
        return m

    def count_M_whole_measurement_frame_num_t2_or_t3(self) -> int:
        m = 0
        for i, each_npz_file_path in enumerate(self.t2_or_t3_npz_files):
            tmp_m = np.load(each_npz_file_path)["orient"].shape[0]
            m += tmp_m
            print(f"{i}\t{each_npz_file_path} has {tmp_m} frames\ttotal={m}")
        return m

    def output_array_creation(self) -> None:
        self.elaz = np.zeros((self.num_M, 90, 2), dtype=np.float64)
        self.data = np.zeros((self.num_M, 90, 230), dtype=np.uint8)
        self.gray = np.zeros((self.num_M, 1, 230), dtype=np.uint8)
        self.wbcm = np.zeros((self.num_M, 480, 640), dtype=np.uint8)

        print(f"{ self.elaz.shape=}")
        print(f"{ self.data.shape=}")
        print(f"{ self.gray.shape=}")
        print(f"{ self.wbcm.shape=}")


    def load_array_orient(self, ith: int) -> NDArray[np.uint8]:
        if self.mtype == "t1":
            return np.load(self.t1_ori_npy_files[ith])
        else:
            return np.load(self.t2_or_t3_npz_files[ith])["orient"]

    def load_array_webcam(self, ith: int) ->  NDArray[np.uint8]:
        if self.mtype == "t1":
            return np.load(self.t1_im2_npy_files[ith])
        else:
            return np.load(self.t2_or_t3_npz_files[ith])["webcam"]

    def load_array_spectr(self, ith: int) ->  NDArray[np.uint8]:
        if self.mtype == "t1":
            #print(f"{self.t1_im0_npy_files[ith]=}")
            return np.load(self.t1_im0_npy_files[ith]).swapaxes(1, 2)
        else:
            return np.load(self.t2_or_t3_npz_files[ith])["spectr"].swapaxes(1, 2)

    def combine(self) -> None:
        if self.mtype == "t1":
            self.organize_filenames_t1()
        elif self.mtype in ("t2", "t3"):
            self.organize_filenames_t2_or_t3()

        if (self.mtype == "t1"):
            self.num_M = self.count_M_whole_measurement_frame_num_t1()
        else:
            self.num_M = self.count_M_whole_measurement_frame_num_t2_or_t3()

        self.output_array_creation()

        mcursor = 0
        num_rotation_files = len(self.t1_ori_npy_files) if (self.mtype == "t1") else len(self.t2_or_t3_npz_files)

        print('1', self.t1_ori_npy_files)
        for i in range(num_rotation_files):
            arr_orient = self.load_array_orient(i)
            #arr_webcam = self.load_array_webcam(i)
            arr_spectr = self.load_array_spectr(i)
            #arr_spectr = arr_spectr

            m = arr_orient.shape[0]  # frame count i-th npy-file (or i-th rotation)
            print(f"{i},{m}, {self.elaz.shape}")

            # elevation & azimuth
            self.elaz[mcursor : mcursor + m, :, 0] = arr_orient[:, 0].reshape(m, 1) + self.angl90offset
            self.elaz[mcursor : mcursor + m, :, 1] = arr_orient[:, 1].reshape(m, 1) - 0.857 * (i % 2 == 0)
            # tmp_im0 = np.load(im0[i]) #.reshape(tmp_ori_m, 480, 200)
            # tmp_im0 = tmp_im0.swapaxes(1, 2)

            # jun10 iwa
            # data[mcursor : mcursor + tmp_ori_m, :, :] = .....[:, 40:40+90,  27:27+230]  #jul8 shiz
            self.data[mcursor : mcursor + m, :, :] = arr_spectr[:,         self.fov_offset:self.fov_offset+90, self.wave_offset:self.wave_offset+230]
            self.gray[mcursor : mcursor + m, 0, :] = np.mean(arr_spectr[:, 330:350,                            self.wave_offset:self.wave_offset+230], axis=1)

            # jul8 iwa
            # gray[mcursor : mcursor + tmp_ori_m, 0, :] = np.mean(tmp_im0[:, 140:140 + 20, 27:27+230], axis=1)
            # self.wbcm[mcursor : mcursor + m, :, :] = arr_webcam[:, :] if self.has_2nd_dev else 0

            mcursor += m

    def save_output(self) -> None:
        if not os.path.isdir(self.outdir):
            os.mkdir(self.outdir)
        np.save(path.join(self.outdir, "data_m-90-230.npy"), self.data)
        print(f"saved: {path.join(self.outdir, 'data_m-90-230.npy')}")

        np.save(path.join(self.outdir, "gray_m-1-230.npy"), self.gray)
        print(f"saved: {path.join(self.outdir, "gray_m-1-230.npy")}")

        np.save(path.join(self.outdir, "wbcm_m-480-640.npy"), self.wbcm)
        np.save(path.join(self.outdir, "elaz.npy"), self.elaz)
        print(f"saved: {path.join(self.outdir, "elaz.npy")}")


# ==================================================================================== #
if __name__ == "__main__":
    data_path_dir = cli_args["ddir"] if "ddir" in cli_args else bad_cli_args(0)
    m_type = cli_args["mtype"] if "mtype" in cli_args else bad_cli_args(1)
    assert os.path.isdir(data_path_dir), bad_cli_args(2)

    dc = DataCombiner(
        indir=data_path_dir,
        outdir="./processed_data/",
        mtype=m_type,
        has_2nd_dev=False,
    )
    dc.combine()
    dc.save_output()

    sys.exit(0)
