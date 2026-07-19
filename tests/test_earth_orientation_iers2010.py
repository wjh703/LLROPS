import erfa
import numpy as np

from llrops.base.epoch import Epoch, TimeScale, utc2tt
from llrops.classes.frames import C04EarthOrientation, EarthOrientationSample
from llrops.classes.frames.earth_orientation import read_iers_c04
from llrops.classes.frames.iers2010_eop import (
    HighFrequencyEopCorrection,
    libration_correction,
    ocean_tide_correction,
)
from llrops.classes.frames.terrestrial import TerrestrialFrameTransform


_ARCSEC_TO_RAD = np.deg2rad(1.0 / 3600.0)


def test_ocean_tide_correction_matches_iers_ortho_eop_reference():
    correction = ocean_tide_correction(47_100.0)

    np.testing.assert_allclose(
        correction.xp_arcsec * 1.0e6,
        -162.8386373279636530,
        rtol=0.0,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        correction.yp_arcsec * 1.0e6,
        117.7907525842668974,
        rtol=0.0,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        correction.ut1_sec * 1.0e6,
        -23.39092370609808214,
        rtol=0.0,
        atol=2.0e-13,
    )


def test_libration_corrections_match_iers_reference_values():
    polar_motion = libration_correction(54_335.0)
    np.testing.assert_allclose(
        [polar_motion.xp_arcsec * 1.0e6, polar_motion.yp_arcsec * 1.0e6],
        [24.83144238273364834, -14.09240692041837661],
        rtol=0.0,
        atol=2.0e-5,
    )

    np.testing.assert_allclose(
        libration_correction(44_239.1).ut1_sec * 1.0e6,
        2.441143834386761746,
        rtol=0.0,
        atol=2.0e-8,
    )
    np.testing.assert_allclose(
        libration_correction(55_227.4).ut1_sec * 1.0e6,
        -2.655705844335680244,
        rtol=0.0,
        atol=2.0e-8,
    )


def test_c04_parser_retains_dx_dy_for_supported_layouts(tmp_path):
    path = tmp_path / "eopc04.txt"
    path.write_text(
        "2020 1 1 58849 0.076 0.282 -0.177 0.001 0.0002 -0.0003\n"
        "2020 1 2 0 58850 0.077 0.283 -0.178 0.0004 -0.0005 0.0\n",
        encoding="utf-8",
    )

    first, second = read_iers_c04(path)
    assert (first.dx_arcsec, first.dy_arcsec) == (0.0002, -0.0003)
    assert (second.dx_arcsec, second.dy_arcsec) == (0.0004, -0.0005)


def test_ut1_interpolation_removes_leap_second_discontinuity():
    eop = C04EarthOrientation(
        (
            EarthOrientationSample(57_753.0, 0.0, 0.0, -0.4),
            EarthOrientationSample(57_754.0, 0.0, 0.0, 0.6),
        )
    )
    midday = Epoch.from_calendar(2016, 12, 31, 12, scale=TimeScale.UTC)

    np.testing.assert_allclose(eop.ut1_minus_utc_sec(midday), -0.4, rtol=0.0, atol=2.0e-11)


def test_terrestrial_matrix_applies_celestial_pole_offsets(monkeypatch):
    import llrops.classes.frames.terrestrial as terrestrial_module

    epoch = Epoch.from_calendar(2020, 1, 1, 6, scale=TimeScale.UTC)
    eop = C04EarthOrientation(
        (
            EarthOrientationSample(
                epoch.mjd,
                0.076,
                0.282,
                -0.177,
                0.0002,
                -0.0003,
            ),
        )
    )
    monkeypatch.setattr(
        terrestrial_module,
        "high_frequency_eop_correction",
        lambda mjd: HighFrequencyEopCorrection(0.0, 0.0, 0.0),
    )

    actual = TerrestrialFrameTransform(eop).celestial_to_terrestrial_matrix(epoch)

    tt = utc2tt(epoch)
    ut11, ut12 = erfa.utcut1(epoch.jd1, epoch.jd2, -0.177)
    x, y, s = erfa.xys06a(tt.jd1, tt.jd2)
    rc2i = erfa.c2ixys(x + 0.0002 * _ARCSEC_TO_RAD, y - 0.0003 * _ARCSEC_TO_RAD, s)
    rpom = erfa.pom00(
        0.076 * _ARCSEC_TO_RAD,
        0.282 * _ARCSEC_TO_RAD,
        erfa.sp00(tt.jd1, tt.jd2),
    )
    expected = erfa.c2tcio(rc2i, erfa.era00(ut11, ut12), rpom)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(actual @ actual.T, np.eye(3), rtol=0.0, atol=2.0e-15)
