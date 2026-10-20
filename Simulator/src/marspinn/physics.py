"""Physical constants and the hydrostatic / ideal-gas relations used everywhere.

Vertical coordinate: geopotential height Z = Phi / g0 (km). In this coordinate hydrostatic
balance is exact:  d lnP / dZ = -m g0 / (k_B T) = -1 / H(T).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

K_B = 1.380649e-23        # J/K
N_A = 6.02214076e23       # 1/mol
G0 = 3.71                 # m s^-2, conventional constant defining geopotential height
T_REF = 200.0             # K, reference temperature for non-dimensionalisation
M_CO2_ATM = 43.49e-3 / N_A  # kg per molecule, typical retrieval constant for Mars


@dataclass
class Physics:
    """Constants fixed after the physics audit of the data."""
    m: float = M_CO2_ATM          # mean molecular mass (kg per molecule)
    g0: float = G0
    T_ref: float = T_REF
    floor: float = float("nan")   # hydrostatic residual of the reference data (RMS, median)

    @property
    def H_ref_km(self) -> float:
        """Reference scale height k_B T_ref / (m g0), in km."""
        return K_B * self.T_ref / (self.m * self.g0) / 1e3

    @property
    def m_gmol(self) -> float:
        return self.m * N_A * 1e3


def scale_height_km(T, phys: Physics):
    return K_B * np.asarray(T) / (phys.m * phys.g0) / 1e3


def number_density(P, T):
    """Ideal-gas law  n = P / (k_B T)."""
    return np.asarray(P) / (K_B * np.asarray(T))


def hydro_residual_np(z_km, T, lnP, phys: Physics):
    """Dimensionless hydrostatic residual  H_ref dlnP/dZ + T_ref/T  (0 when exact)."""
    z_km, T, lnP = map(np.asarray, (z_km, T, lnP))
    o = np.argsort(z_km)
    return phys.H_ref_km * np.gradient(lnP[o], z_km[o]) + phys.T_ref / T[o]


def integrate_lnP(z_km, T, lnP0, phys: Physics):
    """Integrate hydrostatic balance upward from lnP0 at z_km[0] (trapezoidal rule)."""
    z_km, T = np.asarray(z_km, float), np.asarray(T, float)
    integrand = -phys.T_ref / (phys.H_ref_km * T)
    I = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(z_km))])
    return lnP0 + I


def implied_mass(T, lnP, Phi):
    """Mean molecular mass implied by consecutive levels of a hydrostatic profile."""
    T, lnP, Phi = map(np.asarray, (T, lnP, Phi))
    dPhi = np.diff(Phi)
    ok = dPhi > 0
    m = K_B * 0.5 * (T[1:] + T[:-1])[ok] * (-np.diff(lnP))[ok] / dPhi[ok]
    return float(np.median(m))
