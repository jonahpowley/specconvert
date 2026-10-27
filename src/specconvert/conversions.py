"""Convert wavelengths and spectral flux densities with Astropy units."""

import numpy as np
import astropy.units as u

MAGGIE_UNIT = u.def_unit("maggie", 3631 * u.Jy)
CGS_FLUX_UNIT = u.erg / (u.s * u.cm**2 * u.AA)
SI_FLUX_UNIT = u.W / u.m**3


def _check_unmasked(value, name):
    """Reject masked inputs before unit coercion can discard their masks."""
    if np.ma.isMaskedArray(value) or getattr(value, "mask", None) is not None:
        raise TypeError(f"{name} must be unmasked")


def _as_quantity(value, unit, name):
    """Attach units to numeric inputs while preserving existing quantities."""
    _check_unmasked(value, name)
    return value if isinstance(value, u.Quantity) else np.asanyarray(value) * unit


def _wave_quantity(wave):
    """Require a finite positive wavelength for spectral-density conversions."""
    if wave is None:
        raise ValueError("Wavelength is required for spectral-density conversion")
    _check_unmasked(wave, "wave")
    if not isinstance(wave, u.Quantity):
        raise TypeError("wave must be an Astropy Quantity with wavelength units")
    if not wave.unit.is_equivalent(u.m):
        raise u.UnitConversionError("wave must have wavelength units")
    if np.any(~np.isfinite(wave.value) | (wave.value <= 0)):
        raise ValueError("wave values must be finite and positive")
    return wave


def _to_jy(quantity, wave):
    """Convert to Jy, using wavelength only when direct conversion fails."""
    try:
        return quantity.to(u.Jy)
    except u.UnitConversionError:
        return quantity.to(u.Jy, equivalencies=u.spectral_density(_wave_quantity(wave)))


def _from_jy(quantity, target, wave):
    """Convert from Jy, using wavelength only when direct conversion fails."""
    try:
        return quantity.to(target)
    except u.UnitConversionError:
        return quantity.to(target, equivalencies=u.spectral_density(_wave_quantity(wave)))


def convert_wave(wave_in, from_unit, to_unit, return_quantity=False):
    """Convert wavelength values between units.

    Parameters
    ----------
    wave_in : array-like or Quantity
        Wavelength values. Numeric values use ``from_unit``; quantities must
        have units compatible with it.
    from_unit, to_unit : unit-like
        Input and output wavelength units.
    return_quantity : bool, optional
        Return an Astropy Quantity instead of numeric values.

    Returns
    -------
    scalar, ndarray, or Quantity
        Converted wavelength values.
    """
    source, target = u.Unit(from_unit), u.Unit(to_unit)
    if not source.is_equivalent(u.m) or not target.is_equivalent(u.m):
        raise u.UnitConversionError("Wavelength units must have length dimensions")
    wave = _as_quantity(wave_in, source, "wave_in")
    try:
        wave_new = wave.to(target)
    except u.UnitConversionError as exc:
        raise ValueError(f"Cannot convert wavelength from {source} to {target}: {exc}") from exc
    return wave_new if return_quantity else wave_new.value


def convert_flux(flux, from_unit, to_unit, err=None, wave=None, return_quantity=False):
    """Convert spectral flux density and an optional first-order uncertainty.

    Numeric inputs use the declared units. Quantity inputs are converted to
    ``from_unit`` first and must be compatible with it. Wavelength is needed
    only when converting between frequency and wavelength spectral-density
    units. AB magnitude uncertainties are differential increments in mag.

    Parameters
    ----------
    flux : array-like or Quantity
        Input spectral flux density. Signed linear values are preserved.
    from_unit, to_unit : unit-like
        Input and output flux units. AB magnitudes are supported; other
        logarithmic units are rejected.
    err : array-like or Quantity, optional
        One-sigma uncertainty in the input unit. For AB magnitudes, numeric
        uncertainties are in mag and Quantity uncertainties must use ``u.mag``.
    wave : Quantity, optional
        Finite positive wavelength with length units, used only for spectral-density
        conversions between frequency and wavelength representations.
        Uses NumPy broadcasting; reshape wavelength-first cube coordinates to
        ``(n_wave, 1, 1)`` explicitly. Unused wavelength arguments are ignored.
    return_quantity : bool, optional
        Return Astropy Quantities instead of numeric values.

    Returns
    -------
    flux_new : scalar, ndarray, or Quantity
        Converted flux in ``to_unit``.
    err_new : scalar, ndarray, Quantity, or None
        Converted uncertainty, or ``None`` when ``err`` is omitted. Uses
        ``to_unit``, except AB magnitude uncertainties use ``u.mag``.
    """
    declared_in, target = u.Unit(from_unit), u.Unit(to_unit)
    if declared_in.is_equivalent(u.mag) or (isinstance(declared_in, u.LogUnit) and declared_in != u.ABmag):
        raise u.UnitConversionError(f"Unsupported logarithmic flux unit: {declared_in}")
    if target.is_equivalent(u.mag) or (isinstance(target, u.LogUnit) and target != u.ABmag):
        raise u.UnitConversionError(f"Unsupported logarithmic flux unit: {target}")
    flux = _as_quantity(flux, declared_in, "flux")
    if flux.unit.is_equivalent(u.mag) or (isinstance(flux.unit, u.LogUnit) and flux.unit != u.ABmag):
        raise u.UnitConversionError(f"Unsupported logarithmic flux unit: {flux.unit}")
    try:
        flux = flux.to(declared_in)
    except u.UnitConversionError as exc:
        raise u.UnitConversionError(f"flux unit {flux.unit} is incompatible with {declared_in}") from exc
    source = declared_in
    if err is not None:
        if source == u.ABmag:
            if isinstance(err, u.Quantity):
                _check_unmasked(err, "err")
                if err.unit != u.mag:
                    raise u.UnitConversionError("An AB magnitude uncertainty Quantity must use u.mag")
                error_mag = err.to_value(u.mag)
            else:
                _check_unmasked(err, "err")
                error_mag = np.asanyarray(err)
            error = None
        else:
            error = _as_quantity(err, declared_in, "err")
            if error.unit.is_equivalent(u.mag) or isinstance(error.unit, u.LogUnit):
                raise u.UnitConversionError("Magnitude units are valid only for AB magnitude uncertainties")
            try:
                error = error.to(declared_in)
            except u.UnitConversionError as exc:
                raise u.UnitConversionError(f"err unit {error.unit} is incompatible with {declared_in}") from exc
            error_mag = None
    else:
        error = error_mag = None

    if source == u.ABmag and target == u.ABmag:
        flux_new = flux.to(u.ABmag)
        err_new = error_mag * u.mag if error_mag is not None else None
        return (flux_new, err_new) if return_quantity else (flux_new.value, None if err_new is None else err_new.value)
    if source == u.ABmag:
        flux_jy = flux.to(u.Jy)
        sigma_jy = (np.log(10.0) / 2.5) * np.abs(flux_jy.value) * error_mag * u.Jy if error_mag is not None else None
    elif target == u.ABmag:
        flux_jy = _to_jy(flux, wave)
        flux_new = flux_jy.to(u.ABmag)
        err_new = (2.5 / np.log(10.0)) * np.abs(_to_jy(error, wave).value / flux_jy.value) * u.mag if error is not None else None
        return (flux_new, err_new) if return_quantity else (flux_new.value, None if err_new is None else err_new.value)
    else:
        # Preserve ordinary compatible conversions without requiring a wavelength.
        try:
            flux_new = flux.to(target)
            err_new = error.to(target) if error is not None else None
        except u.UnitConversionError:
            flux_jy = _to_jy(flux, wave)
            flux_new = _from_jy(flux_jy, target, wave)
            err_new = _from_jy(_to_jy(error, wave), target, wave) if error is not None else None
        return (flux_new, err_new) if return_quantity else (flux_new.value, None if err_new is None else err_new.value)

    # Convert the AB flux and its first-order uncertainty to a linear target.
    flux_new = _from_jy(flux_jy, target, wave)
    err_new = _from_jy(sigma_jy, target, wave) if error_mag is not None else None
    return (flux_new, err_new) if return_quantity else (flux_new.value, None if err_new is None else err_new.value)


def convert_flux_value(flux, from_unit, to_unit, wave=None, return_quantity=False):
    """Convert flux values without an uncertainty.

    Parameters
    ----------
    flux : array-like or Quantity
        Input spectral flux density.
    from_unit, to_unit : unit-like
        Input and output flux units.
    wave : Quantity, optional
        Finite positive wavelength, required only for spectral-density conversions.
    return_quantity : bool, optional
        Return an Astropy Quantity instead of numeric values.

    Returns
    -------
    ndarray, scalar, or Quantity
        Converted flux values.
    """
    flux_new, _ = convert_flux(flux, from_unit, to_unit, wave=wave, return_quantity=return_quantity)
    return flux_new
