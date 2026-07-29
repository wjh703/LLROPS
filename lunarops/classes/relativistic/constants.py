"""Relativistic and gravitational constants used by frame/time models."""
from __future__ import annotations

# IAU/JPL relativistic scale constants used by coordinate transforms.
# L_B and L_G are IAU defining constants. L_C is kept for compatibility,
# while the actual GCRS<->BCRS scale below uses the explicit L_B-L_G form.
L_B = 1.550519768e-8
L_G = 6.969290134e-10
L_C = 1.48082686741e-8
L_B_MINUS_L_G = L_B - L_G

# DE440-compatible lunar surface scale constant LL from Turyshev et al.
# Table 2 gives LL = 0.003139054 x 1e-8. The LCRS<->GCRS chain uses
# L_B - L_L for DE440 consistency.
L_L_DE440 = 0.003139054e-8
L_B_MINUS_L_L_DE440 = L_B - L_L_DE440


def l_b_minus_l_l_for_ephemeris(ephemeris_file) -> float:
    """Return the lunar relativistic scale term for the selected ephemeris.

    The DE440 lunar solution uses the DE440-compatible L_B-L_L scale. INPOP
    and EPM solutions already use a lunar scale convention for which this
    additional term should be zero. Unknown kernels default to zero rather
    than accidentally applying a DE440-specific scale.
    """
    text = str(ephemeris_file or "").lower()
    if "de440" in text:
        return float(L_B_MINUS_L_L_DE440)
    if "inpop" in text or "epm" in text:
        return 0.0
    return 0.0


GM_SUN = 1.32712440041e20
GM_EARTH = 3.986004355e14
GM_MOON = 4.9028002e12
GM_MERCURY = 2.2032e13
GM_VENUS = 3.24859e14
GM_MARS = 4.282837e13
GM_JUPITER = 1.26686534e17
GM_SATURN = 3.7931187e16
GM_URANUS = 5.793939e15
GM_NEPTUNE = 6.836529e15

# Keys are body names accepted by the LLR processing code (case-insensitive).
# For LLR the planets are best looked up via their barycenters because the
# barycenter is what is reliably available in standard DE ephemerides.
GM_BY_BODY = {
    "SUN": GM_SUN,
    "EARTH": GM_EARTH,
    "MOON": GM_MOON,
    "MERCURY BARYCENTER": GM_MERCURY,
    "VENUS BARYCENTER": GM_VENUS,
    "MARS BARYCENTER": GM_MARS,
    "JUPITER BARYCENTER": GM_JUPITER,
    "SATURN BARYCENTER": GM_SATURN,
    "URANUS BARYCENTER": GM_URANUS,
    "NEPTUNE BARYCENTER": GM_NEPTUNE,
}

# External-potential body lists used in BCRS<->GCRS / BCRS<->LCRS
# coordinate-time conversions (IERS Eq. 11.18 / 11.19; paper Eq. 21 / 23).
EARTH_EXTERNAL_POTENTIAL_BODIES = (
    "SUN", "MOON", "MERCURY BARYCENTER", "VENUS BARYCENTER",
    "MARS BARYCENTER", "JUPITER BARYCENTER",
    "SATURN BARYCENTER", "URANUS BARYCENTER", "NEPTUNE BARYCENTER",
)
MOON_EXTERNAL_POTENTIAL_BODIES = (
    "SUN", "EARTH", "MERCURY BARYCENTER", "VENUS BARYCENTER",
    "MARS BARYCENTER", "JUPITER BARYCENTER",
    "SATURN BARYCENTER", "URANUS BARYCENTER", "NEPTUNE BARYCENTER",
)

__all__ = [
    "EARTH_EXTERNAL_POTENTIAL_BODIES",
    "GM_BY_BODY",
    "GM_EARTH",
    "GM_JUPITER",
    "GM_MARS",
    "GM_MERCURY",
    "GM_MOON",
    "GM_NEPTUNE",
    "GM_SATURN",
    "GM_SUN",
    "GM_URANUS",
    "GM_VENUS",
    "L_B",
    "L_B_MINUS_L_G",
    "L_B_MINUS_L_L_DE440",
    "L_C",
    "L_G",
    "L_L_DE440",
    "MOON_EXTERNAL_POTENTIAL_BODIES",
    "l_b_minus_l_l_for_ephemeris",
]
