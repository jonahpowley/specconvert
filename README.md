# specconvert

Small wavelength and spectral flux density conversions for Python 3.11+, built on NumPy and Astropy.

## Install

```bash
python -m pip install -e .
```

## Use

```python
import astropy.units as u
from specconvert import CGS_FLUX_UNIT, convert_flux, convert_flux_value, convert_wave

wave = convert_wave([0.5, 1.0], u.um, u.AA, return_quantity=True)
flux_jy, error_jy = convert_flux(
    [1.2, 0.8], u.mJy, u.Jy, err=[0.1, 0.08], return_quantity=True
)
flux_ab, error_ab = convert_flux(
    flux_jy, u.Jy, u.ABmag, err=error_jy, return_quantity=True
)
```

`convert_flux` always returns `(flux, uncertainty)`; the uncertainty is `None` if `err` is omitted. `convert_flux_value` returns only flux. Numeric fluxes and uncertainties use `from_unit`; Quantity inputs are converted to `from_unit` and must be compatible with it. Unit strings accepted by `astropy.units.Unit` are also accepted.

AB magnitudes use Astropy's standard AB zero point (3630.7805477 Jy). `MAGGIE_UNIT` is defined as 3631 Jy, so one maggie corresponds to about `-6.5623e-5 ABmag`. AB magnitude uncertainty inputs are differential increments: numeric errors are in mag, and Quantity errors must use `u.mag`. Uncertainty propagation uses a first-order Jacobian at fixed wavelength.

A wavelength is needed only when converting between frequency and wavelength spectral-density units, such as Jy and `CGS_FLUX_UNIT`. Supply a finite positive wavelength Quantity with length units. Unused wavelength arguments are ignored. The functions use ordinary NumPy broadcasting; a cube with wavelength along its first axis should pass `wave[:, None, None]` explicitly:

```python
flux_jy, _ = convert_flux(cube, CGS_FLUX_UNIT, u.Jy, wave=wave[:, None, None])
```

No automatic cube reshaping is performed. Linear fluxes may be signed. Converting zero flux to AB magnitude returns positive infinity; negative flux returns NaN through NumPy's logarithm. Masked arrays and unsupported logarithmic units such as `u.mag` and `u.STmag` are rejected.

The public package exports `convert_wave`, `convert_flux`, `convert_flux_value`, `MAGGIE_UNIT`, `CGS_FLUX_UNIT`, and `SI_FLUX_UNIT`.

## Tests

After installation, run `python -m unittest discover -s tests -v` from the repository root. The suite checks all 81 ordered pairs among nine representative flux units against analytical values and uncertainties, with separate checks for wavelength conversion, Quantity handling, broadcasting, and invalid inputs.
