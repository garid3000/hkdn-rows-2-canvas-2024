## ======== libraries ================================================================= #
#import sys
import os
import os.path as path
import numpy as np
#from numpy.typing import NDArray

class hkdn_data_rows_2_a_bulk:
    angl90offset = np.arange(2.567, 2.567 - 1 / 17.53 * 90, -1 / 17.53).reshape(1, 90)
    wave_offset = 107
    fov_offset = 230

    def __init__(self, indir: str, outdir: str) -> None:
        self.indir = indir
        self.outdir = outdir
        #self.num_M = self.count_all_M()
        self.m = len([x for x in os.listdir(self.indir) if ".npz" in x ])

        assert path.isdir(indir), f"input data path {indir} doesn't exits"

        self.data_m_90_480


    #def count_all_M(self) -> int:
    #    tmp = 0
    #    for each_npz in os.listdir(self.indir):
    #        tmp += np.load(path.join(self.indir, each_npz))["angl"].shape[0]
    #    return tmp


# ==================================================================================== #
if __name__ == "__main__":
    dc = hkdn_data_rows_2_a_bulk(
        indir="/mnt/usb/Hokuden_gimbal/data_copied_data_part",
        outdir="./processed_data/",
    )
