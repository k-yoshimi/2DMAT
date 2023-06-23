import numpy as np

import py2dmat
import py2dmat.solver.function

class Solver(py2dmat.solver.function.Solver):

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
        self._name = "crystal_field"
        self.info_s = info.solver
        for type_info in self.info_s["type"][:]:
            if type_info["func"] not in ["specific", "sus", "mag"]:
                raise ValueError("type must be specific, sus or mag")
        self._type = self.info_s["type"]
        self.weight = self.info_s["weight"]
        #Get target data
        self.target_data = {}
        for type_info in self._type:
            _path_to_target_data = type_info["path_to_reference"]
            data = np.genfromtxt(_path_to_target_data, delimiter=',', encoding="utf-8-sig")[1:,:3]
            self.target_data[type_info["func"]] = data

        self._func = self._diff
        print(info.algorithm)

    def _diff(self, xs: np.ndarray) -> float:
        #Calculate the difference between the target data and the calculated data
        #Update the hamiltonian
        self.info_s["system"]["ham"]={"B20":xs[0], "B40":xs[1], "B44":xs[2]}
        self.simulator = PhysSimulator(self.info_s)
        fx = 0.0
        for idx, type_info in enumerate(self._type):
            func = type_info["func"]
            if self.weight[idx] == 0.0:
                continue
            else:
                weight = self.weight[idx]
            target_data = self.target_data[func]
            magdir = type_info["H_dir"]
            T = type_info.get("T", None)
            magfield = type_info.get("H", None)
            if func == "specific":
                fx += weight*self._diff_specific(target_data, magfield, magdir)
            elif func == "sus":
                fx += weight*self._diff_sus(target_data, magfield, magdir)
            elif func == "mag":
                fx += weight*self._diff_mag(target_data, T, magdir)
        #need normalized procedure ?
        return fx

    def _diff_specific(self, target_data, magfield, magdir ) -> float:
        Temp_table = target_data[:,0]
        SpcHeat_table = target_data[:, 1]
        # Prepare a list of specific heat for each temperature
        SpcHeat_Temp = []
        Entropy_Temp = []
        self.simulator.update_maginfo(magfield, magdir)
        #This process should be modified to use numpy array
        for Temp in Temp_table:
            SpcHeat, Entropy = self.simulator.get_cs(Temp)
            SpcHeat_Temp.append(SpcHeat)
            Entropy_Temp.append(Entropy)
        #Check: entropy is not used in this function
        SpcHeat_Temp = np.array(SpcHeat_Temp)
        delta = np.sqrt(np.mean((SpcHeat_table-SpcHeat_Temp)**2))
        return delta

    def _diff_sus(self, xs: np.ndarray, target_data, magfield, magdir ) -> float:
        Temp_table = target_data[:,0]
        Sus_table = target_data[:, 1]
        chi_Temp = np.zeros(Temp_table.shape[0])
        self.simulator.update_maginfo(magfield, magdir)
        for idx, Temp in enumerate(Temp_table):
            Jmag, chi = self.simulator.get_chi(Temp)
            chi_Temp[idx] = chi
        delta = np.sqrt(np.mean((Sus_table-chi_Temp)**2))
        return delta

    def _diff_mag(self, xs: np.ndarray, target_data, T, magdir ) -> float:
        magfield_Table = target_data[:,0]
        Jmag_table = target_data[:, 1]
        Jmag_MF = []
        for magfield in magfield_Table:
            self.simulator.update_maginfo(magfield, magdir)
            Jmag = self.simulator.get_Jmag(Temp)
            Jmag_MF.append(Jmag)
        delta = np.sqrt(np.mean((Jmag_Table-Jmag_MF)**2))
        return delta

#Crystal Field calculator
class CrystalBase:
    def __init__(self, system_info):
        self.name = system_info["crystal_structure"]
        self.n4f = system_info["n4f"]
        self.n_state = 6
        L = system_info["L"]
        S = system_info["S"]
        system_info["ham"] = system_info.get("ham", {"B20": 0.0, "B40": 0.0, "B44": 5.0})
        self.J = np.abs(L - S)
        self.g = 1 + (self.J * (self.J + 1) + S * (S + 1) - L * (L + 1)) / (2 * self.J * (self.J + 1))
        self.Jz = np.arange(-self.J, self.J + 1)[::-1]
        self.Onn = self._calc_Onn(system_info["ham"])

    def _calc_Onn(self, ham_info):
        return None

    def _get_O20(self, B20):
        Jz = self.Jz
        n_state = self.n_state
        J = self.J
        O20 = np.zeros((6, 6))
        for n in range(n_state):
            O20[n, n] = (3 * Jz[n] * Jz[n] - J * (J + 1)) * B20
        return O20

    def _get_O40(self, B40):
        Jz = self.Jz
        n_state = self.n_state
        J = self.J
        O40 = np.zeros((6, 6))
        for n in range(n_state):
            O40[n, n] = (35 * Jz[n] ** 4 - 30 * J * (J + 1) * Jz[n] ** 2
                         + 25 * Jz[n] ** 2 - 6 * J * (J + 1) + 3 * J ** 2
                         * (J + 1) ** 2) * B40
        return O40

    def _get_O44(self, B44):
        Jz = self.Jz
        n_state = self.n_state
        J = self.J
        O44 = np.zeros((6, 6))
        for n in range(2):
            n_inv = n + 4
            J2 = 1.0
            for j in range(4):
                J2 *= (J + Jz[n] - j) * (J - (Jz[n] - j) + 1)
            O44[n_inv, n] = np.sqrt(J2) * B44 / 2.0
            O44[n, n_inv] = np.conjugate(O44[n_inv, n])
        return O44

    def symmetry(self):
        return self.name

    def get_Onn(self):
        return self.Onn

class CrystalTetra(CrystalBase):
    def _calc_Onn(self, ham_info):
        # O20, O40, O44の行列要素(6x6個)を用意する。それぞれのパラメータはB20,B40,B44である。
        # |J,Jz> J=5/2, Jz=5/2, 3/2, 1/2, -1/2, -3/2, -5/2の空間で考える
        O20 = self._get_O20(ham_info["B20"])
        O40 = self._get_O40(ham_info["B40"])
        O44 = self._get_O44(ham_info["B44"])
        Onn = O20 + O40 + O44
        return Onn

class CrystalCubic(CrystalBase):
    def _calc_Onn(self, ham_info):
        O40 = self._get_O40(ham_info["B40"])
        O44 = self._get_O40(ham_info["B44"])
        Onn = O40 + O44
        return Onn


class Crystal:
    def __init__(self, system_info):
        self.system_info = system_info
        self.name = system_info["crystal_structure"]
    def get(self):
        if self.name == "tetragonal":
            return CrystalTetra(self.system_info)
        elif self.name == "cubic":
            return CrystalCubic(self.system_info)
        else:
            return None


class PhysSimulator:
    def __init__(self, info, magfield=0, magdir=[0, 0, 1]):
        self.crystal = Crystal(info["system"]).get()
        self.Onn = self.crystal.get_Onn()
        self.magfield = magfield
        self.magdir = magdir
        self.nor_magdir = magdir / np.linalg.norm(magdir, ord=2)
        self.Ham, self.eigval, self.eigvec = self._calc_Ham()

    def update_maginfo(self, magfield, magdir):
        self.magfield = magfield
        self.magdir = magdir
        self.nor_magdir = magdir / np.linalg.norm(magdir, ord=2)
        self.Ham, self.eigval, self.eigvec = self._calc_Ham()

    def _calc_Ham(self):
        Hmag = self._calc_Hmag()
        Ham = Hmag + self.Onn
        H_eig = np.linalg.eigh(Ham)
        eigval = H_eig[0] - np.min(H_eig[0])
        eigvec = H_eig[1]
        return Ham, eigval, eigvec

    def _calc_Hmag(self):
        magdir = self.magdir
        magfield = self.magfield
        J = self.crystal.J
        Jz = self.crystal.Jz
        nor_magdir = self.nor_magdir
        n_state = self.crystal.n_state
        g = self.crystal.g

        from scipy.constants import physical_constants
        bohr_unit = physical_constants["Bohr magneton in K/T"][0]

        # 磁場ハミルトニアンの行列要素
        Hmag = np.zeros((n_state, n_state), dtype=complex)
        # 対角要素
        mag_constant = magfield * g * bohr_unit
        for n in range(n_state):
            Hmag[n, n] = Jz[n] * nor_magdir[2] * mag_constant

        # 非対角要素
        mag_term = (magdir[0] + magdir[1] * 1j) * mag_constant / 2
        for n in range(n_state - 1):
            Hmag[n, n + 1] = np.sqrt((J + Jz[n]) * (J - Jz[n] + 1)) * mag_term
            Hmag[n + 1, n] = np.conjugate(Hmag[n, n + 1])
        return Hmag

    def get_Jmag(self, T):
        eigval = self.eigval
        eigvec = self.eigvec
        J = self.crystal.J
        Jz = self.crystal.Jz
        g = self.crystal.g
        n_state = self.crystal.n_state
        nor_magdir = self.nor_magdir

        Each_Ene = np.exp(- eigval / T)
        Za = np.sum(Each_Ene)

        mag = np.zeros((3, n_state), dtype=complex)
        # 磁化のベクトル成分を求める
        for i in range(n_state):
            for n in range(n_state - 1):
                mag_n_np = np.conjugate(eigvec[n, i]) * eigvec[n + 1, i] * np.sqrt(
                    (J - Jz[n + 1]) * (J + Jz[n + 1] + 1))
                mag_np_n = np.conjugate(eigvec[n + 1, i]) * eigvec[n, i] * np.sqrt((J + Jz[n]) * (J - Jz[n] + 1))
                mag[0][i] += (mag_n_np + mag_np_n) / 2
                mag[1][i] += (-mag_n_np + mag_np_n) * (-1J) / 2
                mag[2][i] += np.conjugate(eigvec[n, i]) * eigvec[n, i] * Jz[n]
            factor = np.exp(- eigval[i] / T) / Za
            mag[:, i] *= factor
        Jmag = np.sum(np.sum(mag, axis=1) * g * (-1) * nor_magdir)
        return Jmag

    def get_chi(self, T):
        Jmag = self.get_Jmag(T)
        chi = Jmag / self.magfield * 0.5585
        return Jmag, chi

    def get_cs(self, Temp):
        eigval = self.eigval

        # 分配関数
        Each_Ene = np.exp(-eigval / Temp)
        Z0 = np.sum(Each_Ene)
        # print('Z0',Z0)
        Each_Z1 = eigval / Temp ** 2 * np.exp(-eigval / Temp)
        Z1 = np.sum(Each_Z1)
        # print('Z1',Z1)
        Each_Z2 = (-eigval * 2 / Temp ** 3 + (eigval / Temp ** 2) ** 2) * np.exp(-eigval / Temp)
        Z2 = np.sum(Each_Z2)
        # print('Z2',Z2)

        # 比熱
        from scipy.constants import physical_constants
        R_constant = physical_constants["molar gas constant"][0]
        SpcHeat = (2 * Temp / Z0 * Z1 - Temp ** 2 / Z0 ** 2 * Z1 ** 2 + Temp ** 2 / Z0 * Z2) * R_constant
        # エントロピー
        Entropy = (np.log(Z0) + Temp / Z0 * Z1) * R_constant

        return SpcHeat, Entropy
