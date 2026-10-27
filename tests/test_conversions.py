import unittest

import astropy.units as u
import numpy as np
from astropy.utils.masked import Masked

from specconvert import CGS_FLUX_UNIT, MAGGIE_UNIT, SI_FLUX_UNIT, convert_flux, convert_flux_value, convert_wave


class ConversionMatrixTests(unittest.TestCase):
    def setUp(self):
        self.wave = np.array([0.4, 1.2, 5.0]) * u.um
        self.flux_jy = np.array([1e-6, 1e-4, 1.0])
        self.sigma_jy = self.flux_jy * np.array([0.1, 0.2, 0.03])
        self.units = [u.Jy, u.mJy, u.uJy, u.nJy, u.erg / (u.s * u.cm**2 * u.Hz), CGS_FLUX_UNIT, SI_FLUX_UNIT, MAGGIE_UNIT, u.ABmag]

    def expected(self, unit):
        f = self.flux_jy
        lam = self.wave.to_value(u.AA)
        if unit == u.Jy:
            return f
        if unit == u.mJy:
            return f * 1e3
        if unit == u.uJy:
            return f * 1e6
        if unit == u.nJy:
            return f * 1e9
        if unit == u.erg / (u.s * u.cm**2 * u.Hz):
            return f * 1e-23
        flam = f * 1e-23 * 2.99792458e18 / lam**2
        if unit == CGS_FLUX_UNIT:
            return flam
        if unit == SI_FLUX_UNIT:
            return flam * 1e7
        if unit == MAGGIE_UNIT:
            return f / 3631
        return -2.5 * np.log10(f / 10 ** (23 - 0.4 * 48.6))

    def input_for(self, unit):
        if unit == u.Jy:
            return self.flux_jy, self.sigma_jy
        if unit == u.mJy:
            return self.flux_jy * 1e3, self.sigma_jy * 1e3
        if unit == u.uJy:
            return self.flux_jy * 1e6, self.sigma_jy * 1e6
        if unit == u.nJy:
            return self.flux_jy * 1e9, self.sigma_jy * 1e9
        if unit == u.erg / (u.s * u.cm**2 * u.Hz):
            return self.flux_jy * 1e-23, self.sigma_jy * 1e-23
        flam = self.flux_jy * 1e-23 * 2.99792458e18 / self.wave.to_value(u.AA)**2
        if unit == CGS_FLUX_UNIT:
            return flam, flam * np.array([0.1, 0.2, 0.03])
        if unit == SI_FLUX_UNIT:
            return flam * 1e7, flam * 1e7 * np.array([0.1, 0.2, 0.03])
        if unit == MAGGIE_UNIT:
            return self.flux_jy / 3631, self.sigma_jy / 3631
        mag = self.expected(u.ABmag)
        return mag, (2.5 / np.log(10)) * np.array([0.1, 0.2, 0.03]) * u.mag

    def test_all_ordered_pairs_against_analytic_oracle(self):
        for source in self.units:
            for target in self.units:
                with self.subTest(source=source, target=target):
                    values, errors = self.input_for(source)
                    fnu_units = (u.Jy, u.mJy, u.uJy, u.nJy, self.units[4], MAGGIE_UNIT, u.ABmag)
                    flam_units = (CGS_FLUX_UNIT, SI_FLUX_UNIT)
                    crosses_density_types = (source in fnu_units and target in flam_units) or (source in flam_units and target in fnu_units)
                    wave = self.wave if crosses_density_types else None
                    actual, sigma = convert_flux(values, source, target, err=errors, wave=wave)
                    expected = self.expected(target)
                    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=1e-300)
                    actual_q, sigma_q = convert_flux(values * source, source, target, err=errors, wave=wave, return_quantity=True)
                    self.assertEqual(actual_q.unit, target)
                    self.assertEqual(sigma_q.unit, u.mag if target == u.ABmag else target)
                    np.testing.assert_allclose(actual_q.value, expected, rtol=3e-12, atol=1e-300)
                    # Each source uncertainty is a fixed fractional increment in linear flux.
                    if target == u.ABmag:
                        expected_sigma = 2.5 / np.log(10) * np.array([0.1, 0.2, 0.03])
                    else:
                        expected_sigma = np.array([0.1, 0.2, 0.03]) * np.abs(expected)
                    np.testing.assert_allclose(sigma, expected_sigma, rtol=4e-12, atol=1e-300)
                    np.testing.assert_allclose(sigma_q.value, expected_sigma, rtol=4e-12, atol=1e-300)

    def test_quantity_return_units_and_string_units(self):
        value, sigma = convert_flux(self.flux_jy * u.Jy, "Jy", "mJy", err=self.sigma_jy * u.mJy, return_quantity=True)
        self.assertEqual(value.unit, u.mJy)
        self.assertEqual(sigma.unit, u.mJy)
        np.testing.assert_allclose(value.value, self.flux_jy * 1e3)
        wave = convert_wave([1, 2], "um", "nm", return_quantity=True)
        self.assertEqual(wave.unit, u.nm)
        np.testing.assert_allclose(wave.value, [1000, 2000])

    def test_wavelength_grid_and_input_contract(self):
        scales = {u.AA: 1e10, u.nm: 1e9, u.um: 1e6, u.m: 1.0}
        metres = np.array([4e-7, 1.2e-6, 5e-6])
        for source, source_scale in scales.items():
            values = metres * source_scale
            original = values.copy()
            for target, target_scale in scales.items():
                with self.subTest(source=source, target=target):
                    actual = convert_wave(values, source, target)
                    actual_q = convert_wave(values * source, source, target, return_quantity=True)
                    np.testing.assert_allclose(actual, metres * target_scale, rtol=1e-12)
                    np.testing.assert_allclose(actual_q.value, metres * target_scale, rtol=1e-12)
                    self.assertEqual(actual_q.unit, target)
            np.testing.assert_array_equal(values, original)
        self.assertAlmostEqual(convert_wave(1000 * u.nm, u.um, u.AA), 10000)
        for source, target in ((u.s, u.m), (u.m, u.s), (u.s, u.min)):
            with self.subTest(source=source, target=target), self.assertRaises(u.UnitConversionError):
                convert_wave(1.0, source, target)
        with self.assertRaises(ValueError):
            convert_wave(1.0 * u.s, u.um, u.AA)
        for values in (np.ma.array([1.0], mask=[True]), Masked([1.0] * u.um, mask=[True])):
            with self.assertRaises(TypeError):
                convert_wave(values, u.um, u.AA)

    def test_mixed_quantity_and_error_units(self):
        f, s = convert_flux([1, 2] * u.mJy, "Jy", "Jy", err=[100, 200] * u.uJy, return_quantity=True)
        np.testing.assert_allclose(f.value, [1e-3, 2e-3])
        np.testing.assert_allclose(s.value, [1e-4, 2e-4])

    def test_spectral_density_wave_units_and_broadcasting(self):
        wave_aa = np.array([4000, 12000, 50000]) * u.AA
        wave_nm = wave_aa.to(u.nm)
        wave_um = wave_aa.to(u.um)
        wave_m = wave_aa.to(u.m)
        flux = np.ones((3, 2, 2)) * 1e-20
        result_aa, _ = convert_flux(flux, CGS_FLUX_UNIT, u.Jy, wave=wave_aa[:, None, None], return_quantity=True)
        result_nm, _ = convert_flux(flux, CGS_FLUX_UNIT, u.Jy, wave=wave_nm[:, None, None], return_quantity=True)
        result_um, _ = convert_flux(flux, CGS_FLUX_UNIT, u.Jy, wave=wave_um[:, None, None], return_quantity=True)
        result_m, _ = convert_flux(flux, CGS_FLUX_UNIT, u.Jy, wave=wave_m[:, None, None], return_quantity=True)
        self.assertEqual(result_aa.shape, (3, 2, 2))
        for result in (result_nm, result_um, result_m):
            np.testing.assert_allclose(result_aa.value, result.value)
        scalar_wave, _ = convert_flux(1e-20, CGS_FLUX_UNIT, u.Jy, wave=5000 * u.AA, return_quantity=True)
        self.assertEqual(scalar_wave.shape, ())
        # Equal-axis last dimension remains ordinary NumPy broadcasting semantics.
        last_axis_wave, _ = convert_flux(np.ones((3, 3)), CGS_FLUX_UNIT, u.Jy, wave=wave_aa, return_quantity=True)
        self.assertEqual(last_axis_wave.shape, (3, 3))
        np.testing.assert_allclose(last_axis_wave.value[0], last_axis_wave.value[1])

    def test_wave_requirements_and_validation(self):
        with self.assertRaises(ValueError):
            convert_flux([1.0], CGS_FLUX_UNIT, u.Jy)
        with self.assertRaises(TypeError):
            convert_flux([1.0], CGS_FLUX_UNIT, u.Jy, wave=[5000])
        with self.assertRaises(ValueError):
            convert_flux([1.0], CGS_FLUX_UNIT, u.Jy, wave=[5000] * u.s)
        with self.assertRaises(ValueError):
            convert_flux([1.0], CGS_FLUX_UNIT, u.Jy, wave=[0] * u.AA)
        convert_flux([1.0], u.mJy, u.Jy)
        with self.assertRaises(ValueError):
            convert_flux(np.ones((3, 4)), CGS_FLUX_UNIT, u.Jy, wave=self.wave)
        with self.assertRaises(ValueError):
            convert_flux([1.0], CGS_FLUX_UNIT, u.Jy, wave=[-5000] * u.AA)
        for invalid_wave in (np.nan, np.inf):
            with self.subTest(wave=invalid_wave), self.assertRaises(ValueError):
                convert_flux(1.0, CGS_FLUX_UNIT, u.Jy, wave=invalid_wave * u.AA)
        with self.assertRaises(u.UnitConversionError):
            convert_flux([1.0] * u.m, u.Jy, u.mJy)
        with self.assertRaises(u.UnitConversionError):
            convert_flux([1.0] * u.Jy, u.mJy, u.mJy, err=[0.1] * u.s)

    def test_wrapper_no_error_and_return_modes(self):
        self.assertIsNone(convert_flux(2.0, u.Jy, u.mJy)[1])
        self.assertEqual(convert_flux_value(2.0, u.Jy, u.mJy), 2000.0)
        q = convert_flux_value(2.0, u.Jy, u.mJy, return_quantity=True)
        self.assertEqual(q.unit, u.mJy)

    def test_magnitude_error_unit_and_edge_values(self):
        with self.assertRaises(u.UnitConversionError):
            convert_flux(1.0, u.Jy, u.ABmag, err=0.1 * u.ABmag)
        numeric_error_mag, error_mag = convert_flux([10.0, 12.0], u.ABmag, u.ABmag, err=[0.1, 0.2])
        np.testing.assert_allclose(numeric_error_mag, [10.0, 12.0])
        np.testing.assert_allclose(error_mag, [0.1, 0.2])
        _, error_quantity_mag = convert_flux(10.0, u.ABmag, u.ABmag, err=0.1 * u.mag, return_quantity=True)
        self.assertEqual(error_quantity_mag.unit, u.mag)
        flux_new, err_new = convert_flux(20.0, u.ABmag, u.Jy, err=0.1, return_quantity=True)
        expected_jy = 10 ** (23 - 0.4 * 48.6) * 10 ** (-0.4 * 20)
        np.testing.assert_allclose(flux_new.value, expected_jy, rtol=1e-12)
        np.testing.assert_allclose(err_new.value, expected_jy * np.log(10) / 2.5 * 0.1, rtol=1e-12)
        self.assertEqual(err_new.unit, u.Jy)
        with self.assertRaises(TypeError):
            convert_flux(10.0, u.ABmag, u.ABmag, err=Masked([0.1], mask=[True]) * u.mag)
        with np.errstate(divide="ignore", invalid="ignore"):
            mag, _ = convert_flux([0, -1, np.inf, np.nan], u.Jy, u.ABmag)
        self.assertTrue(np.isposinf(mag[0]))
        self.assertTrue(np.isnan(mag[1]))
        self.assertTrue(np.isneginf(mag[2]))
        self.assertTrue(np.isnan(mag[3]))
        np.testing.assert_allclose(convert_flux([-1, 0], u.ABmag, u.Jy)[0][1], 3630.780547701003)
        linear, _ = convert_flux([-1.0, 0.0, np.inf, np.nan], u.Jy, u.mJy)
        np.testing.assert_allclose(linear, [-1000.0, 0.0, np.inf, np.nan], equal_nan=True)

    def test_unsupported_logarithmic_and_masked_inputs(self):
        for source in (u.mag, u.STmag):
            with self.subTest(source=source), self.assertRaises(u.UnitConversionError):
                convert_flux(1.0, source, u.Jy)
        with self.assertRaises(u.UnitConversionError):
            convert_flux(1 * u.mag, u.Jy, u.Jy)
        with self.assertRaises(TypeError):
            convert_flux(np.ma.array([1.0], mask=[True]), u.Jy, u.mJy)
        with self.assertRaises(TypeError):
            convert_flux([1.0], u.Jy, u.mJy, err=np.ma.array([0.1], mask=[True]))
        with self.assertRaises(TypeError):
            convert_flux(Masked([1.0], mask=[True]) * u.Jy, u.Jy, u.mJy)
        with self.assertRaises(TypeError):
            convert_flux([1.0], u.Jy, u.mJy, err=Masked([0.1], mask=[True]) * u.Jy)
        with self.assertRaises(TypeError):
            convert_flux([1.0], CGS_FLUX_UNIT, u.Jy, wave=Masked([5000], mask=[True]) * u.AA)

    def test_inputs_are_unchanged(self):
        values = np.array([1.0, 2.0])
        errors = np.array([0.1, 0.2])
        wave = np.array([5000, 6000]) * u.AA
        copies = values.copy(), errors.copy(), wave.copy()
        convert_flux(values, u.Jy, u.mJy, err=errors, wave=wave)
        np.testing.assert_array_equal(values, copies[0])
        np.testing.assert_array_equal(errors, copies[1])
        np.testing.assert_array_equal(wave, copies[2])


if __name__ == "__main__":
    unittest.main()
