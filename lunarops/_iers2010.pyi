"""Type declarations for the f2py-generated IERS 2010 extension."""

from numpy import float64
from numpy.typing import ArrayLike, NDArray

def fcul_a(latitude: float, height_m: float, t_k: float, elev_deg: float) -> float: ...
def fculzd_hpa(
    latitude: float,
    ellip_ht: float,
    pressure: float,
    wvp: float,
    lambda_um: float,
) -> tuple[float, float, float]: ...
def ortho_eop(time: float) -> NDArray[float64]: ...
def pmsdnut2(rmjd: float) -> NDArray[float64]: ...
def utlibr(rmjd: float) -> tuple[float, float]: ...
def fundarg(t: float) -> tuple[float, float, float, float, float]: ...
def dehanttideinel(
    xsta: ArrayLike,
    yr: int,
    month: int,
    day: int,
    fhr: float,
    xsun: ArrayLike,
    xmon: ArrayLike,
) -> NDArray[float64]: ...
def hardisp(
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int,
    second: int,
    n: int,
    sample: float,
    blq_amp: ArrayLike,
    blq_phase: ArrayLike,
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]: ...
