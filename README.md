# specconvert

A small package for converting wavelengths, spectral flux densities and their uncertainties. Built on NumPy and Astropy, with the same interface for photometry, spectra and data cubes.

## Installation

`specconvert` requires Python 3.11 or newer. NumPy and Astropy are installed automatically if needed.

To install the latest version from GitHub:

```bash
python -m pip install "git+https://github.com/jonahpowley/specconvert.git"
```

For an editable installation, clone the repository and install it locally:

```bash
git clone https://github.com/jonahpowley/specconvert.git
cd specconvert
python -m pip install -e .
```

Run these commands in the Python environment used by your scripts or notebooks. Changes to the source code are picked up by an editable installation without reinstalling the package.

## Use

```python
import astropy.units as u
import numpy as np

from specconvert import CGS_FLUX_UNIT, SI_FLUX_UNIT, MAGGIE_UNIT, convert_flux, convert_flux_value, convert_wave
```

### Simple conversions

For the usual conversions handled by Astropy's `.to()`, supply the values and their input and output units:

```python
wave_A = convert_wave([0.5, 1.0], from_unit=u.um, to_unit=u.AA)
flux_njy = convert_flux_value([1.2, 0.8], from_unit="uJy", to_unit="nJy")
flux_cgs = convert_flux_value(1.0, from_unit=SI_FLUX_UNIT, to_unit=CGS_FLUX_UNIT)
```

Scalars, arrays and Astropy Quantities are accepted. Results are numeric by default; use `return_quantity=True` to keep the output units. Quantity inputs must have units compatible with `from_unit`. Unit strings accepted by Astropy can also be used.

### Fluxes and uncertainties

Use `convert_flux` to convert a flux and its uncertainty together:

```python
flux_new, err_new = convert_flux([1.2, 0.8], from_unit=u.uJy, to_unit=u.nJy, err=[0.1, 0.08])
```

This always returns `(flux_new, err_new)`, with `err_new=None` if no uncertainty was supplied. `convert_flux_value` returns only the converted flux.

### Spectral flux densities

The same function converts between flux density per unit wavelength, such as CGS `erg / (s cm² Å)`, and flux density per unit frequency, such as Jy. Supply a wavelength Quantity for each spectral sample or photometric band:

```python
wave = np.array([1.0, 2.0]) * u.um
flux = np.array([1e-19, 2e-19])
err = np.array([1e-20, 2e-20])

flux_jy, err_jy = convert_flux(flux, from_unit=CGS_FLUX_UNIT, to_unit=u.Jy, err=err, wave=wave)
```

Wavelengths may be in Å, nm, µm or other length units, and must be finite and positive. Wavelength is only needed when changing between the two spectral-density representations. Conversions such as µJy to nJy, or SI to CGS per unit wavelength, do not need it.

Scaled units can be supplied directly, for example to express a spectrum in units of `1e-19 erg / (s cm² Å)`:

```python
flux_scaled, err_scaled = convert_flux(flux_jy, from_unit=u.Jy, to_unit=1e-19 * CGS_FLUX_UNIT, err=err_jy, wave=wave)
```

`CGS_FLUX_UNIT` is `erg / (s cm² Å)` and `SI_FLUX_UNIT` is `W / m³`; both describe flux density per unit wavelength.

### AB magnitudes and maggies

Use `u.ABmag` for magnitudes. Fluxes, magnitudes and maggies can be converted through the same interface:

```python
mag, mag_err = convert_flux([100.0, 200.0], from_unit=u.nJy, to_unit=u.ABmag, err=[10.0, 20.0])
maggies = convert_flux_value(mag, from_unit=u.ABmag, to_unit=MAGGIE_UNIT)
flux_cgs = convert_flux_value(maggies, from_unit=MAGGIE_UNIT, to_unit=CGS_FLUX_UNIT, wave=wave)
```

Magnitude uncertainties are in mag. If supplied as Quantities, they must use `u.mag`; the flux itself uses `u.ABmag`. Magnitude errors use first-order propagation, with wavelength treated as exact.

AB magnitudes use Astropy's standard zero point of 3630.7805477 Jy. `MAGGIE_UNIT` is defined as 3631 Jy, so one maggie corresponds to approximately `-6.5623e-5 ABmag`.

### Data cubes

For a cube shaped `(n_wave, ny, nx)`, reshape the wavelength array to `(n_wave, 1, 1)`. Each wavelength then applies to its whole spatial slice:

```python
cube = np.ones((len(wave), 3, 3)) * 1e-19
cube_jy, _ = convert_flux(cube, from_unit=CGS_FLUX_UNIT, to_unit=u.Jy, wave=wave[:, None, None])
```

The functions use NumPy broadcasting without automatic reshaping. A scalar wavelength can also be used for an image or a whole cube.

Linear fluxes may be signed. Converting zero flux to AB magnitude returns positive infinity; negative flux returns NaN. Masked arrays and magnitude systems other than AB are not supported.

## Tests

Run from the repository directory:

```bash
python -m unittest discover -s tests -v
```

The tests check all 81 ordered pairs among nine flux units against analytical values and uncertainties, along with wavelength conversions, Quantity inputs, broadcasting and invalid inputs.

## License

Released under the [MIT License](LICENSE). For questions or bug reports, use [GitHub Issues](https://github.com/jonahpowley/specconvert/issues).
