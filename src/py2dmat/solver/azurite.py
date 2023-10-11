import numpy as np
import ctypes
import py2dmat
import py2dmat.solver.function

class Solver(py2dmat.solver.function.Solver):
    """Function Solver with pre-defined benchmark functions"""

    x: np.ndarray
    fx: float

    def __init__(self, info: py2dmat.Info) -> None:
        """
        Initialize the solver.

        Parameters
        ----------
        info: Info
        """
        super().__init__(info)
        self._name = "azurite"
        # Get target data
        info_s = info.solver
        _path_to_target_data = info_s.get("path_to_reference", "exp_J.csv")
        data = np.genfromtxt(_path_to_target_data, delimiter=',', encoding="utf-8-sig")[1:, :]
        self.B_target = data[:, 0]
        self.m_experiment = data[:, 1:]
        self._func = self._exp_diff
        print(info.algorithm)
        #set parameters
        #fix parameters
        self.muB = 0.05788381  # (meV/T)
        self.gval = 2.06  # g value
        #read parameters from toml file
        self.size = info_s.get("size", 4)
        self.sweep_dim = info_s.get("sweep_dim", [10])
        self.eps_cut = info_s.get("eps_cut", 1.0e-12)
        #TODO: set dmrg parameters

        #Load library
        _path_to_dmrg_library = info_s.get("path_to_dmrg_library", "dmrg.so")
        self.dmrg_lib = ctypes.CDLL(_path_to_dmrg_library)

    def _calc_dmrg(self, J_all,size,calc_type,conv_const,read_H, ini_cnt, fin_cnt, d_cnt, int_seq, int_read, int_write):
        # [s] for dmrg calc by ITensor
        dmrg_lib = self.dmrg_lib
        dmrg_lib.compute_energy.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64), ctypes.c_int,
                                            np.ctypeslib.ndpointer(dtype=np.float64),
                                            np.ctypeslib.ndpointer(dtype=np.float64),
                                            ctypes.c_int, ctypes.c_int, ctypes.c_double,
                                            np.ctypeslib.ndpointer(dtype=np.int64),
                                            ctypes.c_int]
        dmrg_lib.compute_energy.restype = ctypes.c_double
        mag_H = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # Hx,Hy,Hz
        result = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)  # ene,mx,my,mz
        all_result = np.zeros((len(read_H), 2), dtype=np.float64)
        read_mag = self.m_experiment
        out_file = "%s_size%d.dat" % (calc_type, size)
        with open(out_file, "w") as f:
            print("# Hz ene m_x m_y m_z read_mag")
        tmp_cnt = 0
        for cnt in range(ini_cnt, fin_cnt, d_cnt):
            Hz = (-1.0) * (conv_const) * read_H[cnt]
            mag_H[2] = Hz
            print(calc_type, size, mag_H[2])
            if tmp_cnt > 0 and int_seq == 1:
                int_read = 1
            dmrg_lib.compute_energy(J_all, size, mag_H, result, int_read, int_write, self.eps_cut, self.sweep_dim, len(self.sweep_dim))
            all_result[cnt][0] = result[0]  # ene
            all_result[cnt][1] = result[3]  # mz
            with open(out_file, "a") as f:
                print(" %f %f %f %f %f %f" % (mag_H[2], result[0], result[1], result[2], result[3], read_mag[cnt]),
                      file=f)
            tmp_cnt += 1
        return all_result

    def _calc_GS(self, xs: np.ndarray):
        cnt_num = len(self.B_target)
        J_all = xs
        abs_J1 = np.abs(J_all[0])  # unit = meV
        conv_const = self.gval * self.muB / abs_J1  # Tesla ->  meV
        J_normalized = J_all/abs_J1
        read_H = self.B_target
        read_mag = self.m_experiment
        results = {}
        for calc_type in ["rand", "seq", "invseq"]:
            if calc_type == "rand":
                ini_cnt = 0
                fin_cnt = len(read_H)
                int_seq = 0
                d_cnt = 1
                int_read = 0
                int_write = 0
            elif calc_type == "seq":
                ini_cnt = 0
                fin_cnt = len(read_H)
                int_seq = 1
                d_cnt = 1
                int_read = 0
                int_write = 1
            elif calc_type == "invseq":
                ini_cnt = len(read_H) - 1
                fin_cnt = -1
                d_cnt = -1
                int_seq = 1
                int_read = 0
                int_write = 1
            results[calc_type] = self._calc_dmrg(J_normalized, self.size, calc_type, conv_const, self.B_target, ini_cnt, fin_cnt, d_cnt, int_seq, int_read, int_write)

        # [s] find the GS
        rand_result = results["rand"]
        seq_result = results["seq"]
        invseq_result = results["invseq"]
        GS_result = np.zeros((cnt_num, 3), dtype=np.float64)
        for cnt in range(len(read_H)):
            if rand_result[cnt][0] < seq_result[cnt][0]:
                if rand_result[cnt][0] < invseq_result[cnt][0]:
                    GS_result[cnt][0] = rand_result[cnt][0]
                    GS_result[cnt][1] = rand_result[cnt][1]
                    GS_result[cnt][2] = 0
                else:
                    GS_result[cnt][0] = invseq_result[cnt][0]
                    GS_result[cnt][1] = invseq_result[cnt][1]
                    GS_result[cnt][2] = 2
            else:
                if seq_result[cnt][0] < invseq_result[cnt][0]:
                    GS_result[cnt][0] = seq_result[cnt][0]
                    GS_result[cnt][1] = seq_result[cnt][1]
                    GS_result[cnt][2] = 1
                else:
                    GS_result[cnt][0] = invseq_result[cnt][0]
                    GS_result[cnt][1] = invseq_result[cnt][1]
                    GS_result[cnt][2] = 2
        with open("final_result_%d.dat" % (self.size), "w") as f:
            for cnt in range(len(read_H)):
                print(" %f %f %f %f %f " % (read_H[cnt], GS_result[cnt][0], GS_result[cnt][1] * self.gval, GS_result[cnt][2], read_mag[cnt]), file=f)
        return GS_result
        # [e] find the GS

    def _exp_diff(self, xs: np.ndarray) -> float:
        GS_result = self._calc_GS(xs)
        #print("Debug")
        #print(GS_result)
        #print(self.m_experiment[:, 0])
        delta = np.sqrt(np.mean((self.m_experiment[:,0]-GS_result[:,1])**2))
        return delta
