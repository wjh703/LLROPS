"""IERS 2010 subdaily Earth-orientation corrections.

This is an independent NumPy implementation of the models published in
IERS Conventions (2010), Tables 5.1a, 5.1b and Chapter 8.  It follows the
algorithms of the reference PMSDNUT2, UTLIBR, ORTHO_EOP and CNMTX routines,
but exposes values in arcseconds and seconds and uses LLROPS naming.
"""
from __future__ import annotations

from dataclasses import dataclass

import erfa
import numpy as np

_TWO_PI = 2.0 * np.pi
_MJD_J2000 = 51_544.5


@dataclass(frozen=True, slots=True)
class HighFrequencyEopCorrection:
    xp_arcsec: float
    yp_arcsec: float
    ut1_sec: float


# h_s, phase and frequency for the 71 Cartwright-Tayler-Edden lines used by
# the IERS ORTHO_EOP/CNMTX ocean-tide model.  Rows 0:41 are diurnal (m=1),
# and rows 41:71 are semidiurnal (m=2).
_OCEAN_LINES = np.array(
    [
        [-1.94, 9.0899831, 5.18688050],
        [-1.25, 8.8234208, 5.38346657],
        [-6.64, 12.1189598, 5.38439079],
        [-1.51, 1.4425700, 5.41398343],
        [-8.02, 4.7381090, 5.41490765],
        [-9.47, 4.4715466, 5.61149372],
        [-50.20, 7.7670857, 5.61241794],
        [-1.80, -2.9093042, 5.64201057],
        [-9.54, 0.3862349, 5.64293479],
        [1.52, -3.1758666, 5.83859664],
        [-49.45, 0.1196725, 5.83952086],
        [-262.21, 3.4152116, 5.84044508],
        [1.70, 12.8946194, 5.84433381],
        [3.43, 5.5137686, 5.87485066],
        [1.94, 6.4441883, 6.03795537],
        [1.37, -4.2322016, 6.06754801],
        [7.41, -0.9366625, 6.06847223],
        [20.62, 8.5427453, 6.07236095],
        [4.14, 11.8382843, 6.07328517],
        [3.94, 1.1618945, 6.10287781],
        [-7.14, 5.9693878, 6.24878055],
        [1.37, -1.2032249, 6.26505830],
        [-122.03, 2.0923141, 6.26598252],
        [1.02, -1.7847596, 6.28318449],
        [2.89, 8.0679449, 6.28318613],
        [-7.30, 0.8953321, 6.29946388],
        [368.78, 4.1908712, 6.30038810],
        [50.01, 7.4864102, 6.30131232],
        [-1.08, 10.7819493, 6.30223654],
        [2.93, 0.3137975, 6.31759007],
        [5.25, 6.2894282, 6.33479368],
        [3.95, 7.2198478, 6.49789839],
        [20.62, -0.1610030, 6.52841524],
        [4.09, 3.1345361, 6.52933946],
        [3.42, 2.8679737, 6.72592553],
        [1.69, -4.5128771, 6.75644239],
        [11.29, 4.9665307, 6.76033111],
        [7.23, 8.2620698, 6.76125533],
        [1.51, 11.5576089, 6.76217955],
        [2.16, 0.6146566, 6.98835826],
        [1.38, 3.9101957, 6.98928248],
        [1.80, 20.6617051, 11.45675174],
        [4.67, 13.2808543, 11.48726860],
        [16.01, 16.3098310, 11.68477889],
        [19.32, 8.9289802, 11.71529575],
        [1.30, 5.0519065, 11.73249771],
        [-1.02, 15.8350306, 11.89560406],
        [-4.51, 8.6624178, 11.91188181],
        [120.99, 11.9579569, 11.91280603],
        [1.13, 8.0808832, 11.93000800],
        [22.98, 4.5771061, 11.94332289],
        [1.06, 0.7000324, 11.96052486],
        [-1.90, 14.9869335, 12.11031632],
        [-2.18, 11.4831564, 12.12363121],
        [-23.58, 4.3105437, 12.13990896],
        [631.92, 7.6060827, 12.14083318],
        [1.92, 3.7290090, 12.15803515],
        [-4.66, 10.6350594, 12.33834347],
        [-17.86, 3.2542086, 12.36886033],
        [4.47, 12.7336164, 12.37274905],
        [1.97, 16.0291555, 12.37367327],
        [17.20, 10.1602590, 12.54916865],
        [294.00, 6.2831853, 12.56637061],
        [-2.46, 2.4061116, 12.58357258],
        [-1.02, 5.0862033, 12.59985198],
        [79.96, 8.3817423, 12.60077620],
        [23.83, 11.6772814, 12.60170041],
        [2.59, 14.9728205, 12.60262463],
        [4.47, 4.0298682, 12.82880334],
        [1.95, 7.3254073, 12.82972756],
        [1.17, 9.1574019, 13.06071921],
    ],
    dtype=float,
)

_ORTHOTIDE_FACTORS = np.array(
    [
        [0.0298, 0.1408, 0.0805, 0.6002, 0.3025, 0.1517],
        [0.0200, 0.0905, 0.0638, 0.3476, 0.1645, 0.0923],
    ],
    dtype=float,
)
_ORTHOWEIGHTS = np.array(
    [
        [-6.77832, -14.86323, 0.47884, -1.45303, 0.16406, 0.42030,
         0.09398, 25.73054, -4.77974, 0.28080, 1.94539, -0.73089],
        [14.86283, -6.77846, 1.45234, 0.47888, -0.42056, 0.16469,
         15.30276, -4.30615, 0.07564, 2.28321, -0.45717, -1.62010],
        [-1.76335, 1.03364, -0.27553, 0.34569, -0.12343, -0.10146,
         -0.47119, 1.28997, -0.19336, 0.02724, 0.08955, 0.04726],
    ],
    dtype=float,
)

_PM_ARGUMENTS = np.array(
    [
        [1, -1, 0, -2, 0, -1], [1, -1, 0, -2, 0, -2],
        [1, 1, 0, -2, -2, -2], [1, 0, 0, -2, 0, -1],
        [1, 0, 0, -2, 0, -2], [1, -1, 0, 0, 0, 0],
        [1, 0, 0, -2, 2, -2], [1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, -1], [1, 1, 0, 0, 0, 0],
    ],
    dtype=float,
)
_PM_COEFFICIENTS = np.array(
    [
        [-0.4, 0.3, -0.3, -0.4], [-2.3, 1.3, -1.3, -2.3],
        [-0.4, 0.3, -0.3, -0.4], [-2.1, 1.2, -1.2, -2.1],
        [-11.4, 6.5, -6.5, -11.4], [0.8, -0.5, 0.5, 0.8],
        [-4.8, 2.7, -2.7, -4.8], [14.3, -8.2, 8.2, 14.3],
        [1.9, -1.1, 1.1, 1.9], [0.8, -0.4, 0.4, 0.8],
    ],
    dtype=float,
)

_UT_ARGUMENTS = np.array(
    [
        [2, -2, 0, -2, 0, -2], [2, 0, 0, -2, -2, -2],
        [2, -1, 0, -2, 0, -2], [2, 1, 0, -2, -2, -2],
        [2, 0, 0, -2, 0, -1], [2, 0, 0, -2, 0, -2],
        [2, 1, 0, -2, 0, -2], [2, 0, -1, -2, 2, -2],
        [2, 0, 0, -2, 2, -2], [2, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, -1],
    ],
    dtype=float,
)
_UT_COEFFICIENTS = np.array(
    [
        [0.05, -0.03], [0.06, -0.03], [0.35, -0.20],
        [0.07, -0.04], [-0.07, 0.04], [1.75, -1.01],
        [-0.05, 0.03], [0.04, -0.03], [0.76, -0.44],
        [0.21, -0.12], [0.06, -0.04],
    ],
    dtype=float,
)


def _fundamental_arguments(mjd: float) -> np.ndarray:
    t = (float(mjd) - _MJD_J2000) / 36_525.0
    gmst_sec = np.fmod(
        67_310.54841
        + t * (3_164_400_184.812866 + t * (0.093104 - t * 0.0000062)),
        86_400.0,
    )
    return np.array(
        [
            np.fmod(gmst_sec * _TWO_PI / 86_400.0 + np.pi, _TWO_PI),
            erfa.fal03(t),
            erfa.falp03(t),
            erfa.faf03(t),
            erfa.fad03(t),
            erfa.faom03(t),
        ],
        dtype=float,
    )


def ocean_tide_correction(mjd: float) -> HighFrequencyEopCorrection:
    """Return IERS ORTHO_EOP ocean-tide corrections."""
    amplitudes = np.zeros((2, 3, 2), dtype=float)
    hs, phase, frequency = _OCEAN_LINES.T
    orders = np.concatenate((np.ones(41, dtype=int), np.full(30, 2, dtype=int)))
    for k_index, k in enumerate((-1, 0, 1)):
        dt60 = float(mjd) - 2.0 * k - 37_076.5
        pinm = np.mod(2 + orders, 2) * np.pi / 2.0
        alpha = np.fmod(phase - pinm, _TWO_PI) + np.fmod(frequency * dt60, _TWO_PI)
        for order in (1, 2):
            selected = orders == order
            amplitudes[order - 1, k_index, 0] = np.sum(hs[selected] * np.cos(alpha[selected]))
            amplitudes[order - 1, k_index, 1] = -np.sum(hs[selected] * np.sin(alpha[selected]))

    h = np.empty(12, dtype=float)
    for order_index in range(2):
        a = amplitudes[order_index, :, 0]
        b = amplitudes[order_index, :, 1]
        ap, am = a[2] + a[0], a[2] - a[0]
        bp, bm = b[2] + b[0], b[2] - b[0]
        sp = _ORTHOTIDE_FACTORS[order_index]
        p = (sp[0] * a[1], sp[1] * a[1] - sp[2] * ap, sp[3] * a[1] - sp[4] * ap + sp[5] * bm)
        q = (sp[0] * b[1], sp[1] * b[1] - sp[2] * bp, sp[3] * b[1] - sp[4] * bp - sp[5] * am)
        offset = order_index * 6
        h[offset : offset + 6] = np.ravel(np.column_stack((p, q)))

    dx_microas, dy_microas, dut1_microsec = _ORTHOWEIGHTS @ h
    return HighFrequencyEopCorrection(dx_microas * 1.0e-6, dy_microas * 1.0e-6, dut1_microsec * 1.0e-6)


def libration_correction(mjd: float) -> HighFrequencyEopCorrection:
    """Return IERS tidal-gravitation libration corrections."""
    arguments = _fundamental_arguments(mjd)
    pm_angles = np.fmod(_PM_ARGUMENTS @ arguments, _TWO_PI)
    pm_sin = np.sin(pm_angles)
    pm_cos = np.cos(pm_angles)
    dx_microas = np.sum(_PM_COEFFICIENTS[:, 0] * pm_sin + _PM_COEFFICIENTS[:, 1] * pm_cos)
    dy_microas = np.sum(_PM_COEFFICIENTS[:, 2] * pm_sin + _PM_COEFFICIENTS[:, 3] * pm_cos)

    ut_angles = np.fmod(_UT_ARGUMENTS @ arguments, _TWO_PI)
    dut1_microsec = np.sum(
        _UT_COEFFICIENTS[:, 0] * np.sin(ut_angles)
        + _UT_COEFFICIENTS[:, 1] * np.cos(ut_angles)
    )
    return HighFrequencyEopCorrection(dx_microas * 1.0e-6, dy_microas * 1.0e-6, dut1_microsec * 1.0e-6)


def high_frequency_eop_correction(mjd: float) -> HighFrequencyEopCorrection:
    ocean = ocean_tide_correction(mjd)
    libration = libration_correction(mjd)
    return HighFrequencyEopCorrection(
        ocean.xp_arcsec + libration.xp_arcsec,
        ocean.yp_arcsec + libration.yp_arcsec,
        ocean.ut1_sec + libration.ut1_sec,
    )


__all__ = [
    "HighFrequencyEopCorrection",
    "high_frequency_eop_correction",
    "libration_correction",
    "ocean_tide_correction",
]
