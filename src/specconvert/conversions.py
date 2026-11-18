"""Convert wavelengths and spectral flux densities with Astropy units."""

import warnings

import numpy as np
import astropy.units as u

FNU_CGS  = u.erg / (u.s * u.cm**2 * u.Hz)
FLAM_CGS = u.erg / (u.s * u.cm**2 * u.AA)
FLAM_SI  = u.W / u.m**3
MAGGIE_UNIT = u.def_unit("maggie", 3631 * u.Jy)
u.add_enabled_units([MAGGIE_UNIT])

# Deprecated names, resolved lazily so that access emits a warning.
_DEPRECATED = {"CGS_FLUX_UNIT": "FLAM_CGS", "SI_FLUX_UNIT": "FLAM_SI"}


def __getattr__(name):
    if name in _DEPRECATED:
        warnings.warn(
            f"{name} is deprecated; use {_DEPRECATED[name]}", DeprecationWarning, stacklevel=2
        )
        return globals()[_DEPRECATED[name]]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _check_unmasked(value, name):
    """Reject masked inputs before unit coercion can discard their masks."""
    if np.ma.isMaskedArray(value) or getattr(value, "mask", None) is not None:
        raise TypeError(f"{name} must be unmasked")


def _as_quantity(value, unit, name):
    """Attach units to numeric inputs while preserving existing quantities."""
    _check_unmasked(value, name)
    return value if isinstance(value, u.Quantity) else np.asanyarray(value) * unit


def _check_flux_unit(unit):
    """Reject logarithmic units other than AB magnitudes."""
    if unit.is_equivalent(u.mag) or (isinstance(unit, u.LogUnit) and unit != u.ABmag):
        raise u.UnitConversionError(f"Unsupported logarithmic flux unit: {unit}")


def _wave_quantity(wave, like):
    """Require a positive, non-infinite wavelength (NaN propagates) for spectral-density conversions."""
    if wave is None:
        raise ValueError("Wavelength is required for spectral-density conversion")
    _check_unmasked(wave, "wave")
    if not isinstance(wave, u.Quantity):
        raise TypeError("wave must be an Astropy Quantity with wavelength units")
    if not wave.unit.is_equivalent(u.m):
        raise u.UnitConversionError("wave must have wavelength units")
    if np.any(np.isinf(wave.value) | (wave.value <= 0)):
        raise ValueError("wave values must be positive and finite or NaN")
    if wave.size > 1 and like.ndim and wave.ndim != like.ndim:
        raise ValueError(
            f"wave has {wave.ndim} dim(s) but flux has {like.ndim}; reshape wave "
            f"explicitly, e.g. wave[:, None, None] for a (n_wave, ny, nx) cube"
        )
    return wave


def _pack(flux_new, err_new, return_quantity):
    """Return Quantities or plain values."""
    if return_quantity:
        return flux_new, err_new
    return flux_new.value, None if err_new is None else err_new.value


def _is_spectral_density(unit):
    """Whether a unit is a spectral flux density per frequency or per wavelength."""
    return unit.is_equivalent(FNU_CGS) or unit.is_equivalent(FLAM_CGS)


def _convert_density(quantity, target, wave):
    """Convert units directly; use wave only for Fν <-> Fλ."""
    try:
        return quantity.to(target)
    except u.UnitConversionError:
        if not (_is_spectral_density(quantity.unit) and _is_spectral_density(target)):
            raise
        return quantity.to(
            target, equivalencies=u.spectral_density(_wave_quantity(wave, quantity))
        )


def convert_wave(wave_in, from_unit, to_unit, *, return_quantity=False):
    """Convert wavelengths between length units.

    Parameters
    ----------
    wave_in : array-like or Quantity
        Wavelengths. Plain numbers are taken to be in ``from_unit``.
    from_unit, to_unit : unit-like
        Input and output length units.
    return_quantity : bool, optional
        Return a Quantity instead of plain values.

    Returns
    -------
    scalar, ndarray or Quantity
        Wavelengths in ``to_unit``.
    """
    source, target = u.Unit(from_unit), u.Unit(to_unit)
    if not source.is_equivalent(u.m) or not target.is_equivalent(u.m):
        raise u.UnitConversionError("Wavelength units must have length dimensions")
    wave = _as_quantity(wave_in, source, "wave_in")
    wave_new = wave.to(target)
    return wave_new if return_quantity else wave_new.value


def convert_flux(flux, from_unit, to_unit, *, err=None, wave=None, return_quantity=False):
    """Convert spectral flux density and an optional uncertainty.

    Parameters
    ----------
    flux : array-like or Quantity
        Flux density. Plain numbers are taken to be in ``from_unit``.
    from_unit, to_unit : unit-like
        Input and output units: a flux density unit or ``u.ABmag``.
    err : array-like or Quantity, optional
        One-sigma uncertainty in ``from_unit`` (in mag for ``u.ABmag``),
        propagated to first order.
    wave : Quantity, optional
        Wavelength at which the flux is evaluated, needed only for
        F_nu <-> F_lambda conversions. Must be positive and not infinite;
        NaN propagates. If it has more than one element it must have the same
        ``ndim`` as ``flux``, e.g. ``wave[:, None, None]`` for an
        ``(n_wave, ny, nx)`` cube.
    return_quantity : bool, optional
        Return Quantities instead of plain values.

    Returns
    -------
    flux_new : scalar, ndarray or Quantity
        Flux in ``to_unit``. NaN where ``flux`` is non-positive and
        ``to_unit`` is ``u.ABmag``.
    err_new : scalar, ndarray, Quantity or None
        Uncertainty in ``to_unit`` (in mag for ``u.ABmag``), or ``None`` if
        ``err`` is ``None``. NaN where ``flux`` is non-positive and
        ``to_unit`` is ``u.ABmag``.
    """
    source, target = u.Unit(from_unit), u.Unit(to_unit)
    _check_flux_unit(source)
    _check_flux_unit(target)
    flux = _as_quantity(flux, source, "flux")
    _check_flux_unit(flux.unit)
    try:
        flux = flux.to(source)
    except u.UnitConversionError as exc:
        raise u.UnitConversionError(f"flux unit {flux.unit} is incompatible with {source}") from exc

    if err is not None:
        if source == u.ABmag:
            _check_unmasked(err, "err")
            if isinstance(err, u.Quantity):
                if not err.unit.is_equivalent(u.mag):
                    raise u.UnitConversionError("An AB magnitude uncertainty Quantity must have magnitude units")
                error_mag = err.to_value(u.mag)
            else:
                error_mag = np.asanyarray(err)
            error = None
        else:
            error = _as_quantity(err, source, "err")
            if error.unit.is_equivalent(u.mag) or isinstance(error.unit, u.LogUnit):
                raise u.UnitConversionError("Magnitude units are valid only for AB magnitude uncertainties")
            try:
                error = error.to(source)
            except u.UnitConversionError as exc:
                raise u.UnitConversionError(f"err unit {error.unit} is incompatible with {source}") from exc
            error_mag = None
    else:
        error = error_mag = None

    if source == u.ABmag and target == u.ABmag:
        flux_new = flux.to(u.ABmag)
        err_new = None if error_mag is None else error_mag * u.mag

    elif source == u.ABmag:
        flux_jy = flux.to(u.Jy)
        flux_new = _convert_density(flux_jy, target, wave)
        err_new = None
        if error_mag is not None:
            sigma_jy = (np.log(10.0) / 2.5) * np.abs(flux_jy.value) * error_mag * u.Jy
            err_new = _convert_density(sigma_jy, target, wave)

    elif target == u.ABmag:
        flux_jy = _convert_density(flux, u.Jy, wave)
        # Non-positive flux has no AB magnitude: return NaN.
        flux_jy = np.where(flux_jy.value > 0, flux_jy.value, np.nan) * u.Jy
        flux_new = flux_jy.to(u.ABmag)
        err_new = None
        if error is not None:
            err_new = (2.5 / np.log(10.0)) * np.abs(_convert_density(error, u.Jy, wave).value / flux_jy.value) * u.mag

    else:
        flux_new = _convert_density(flux, target, wave)
        err_new = None if error is None else _convert_density(error, target, wave)

    return _pack(flux_new, err_new, return_quantity)


def convert_flux_value(flux, from_unit, to_unit, *, wave=None, return_quantity=False):
    """Convert spectral flux density without an uncertainty.

    Parameters
    ----------
    flux : array-like or Quantity
        Flux density. Plain numbers are taken to be in ``from_unit``.
    from_unit, to_unit : unit-like
        Input and output units: a flux density unit or ``u.ABmag``.
    wave : Quantity, optional
        Wavelength at which the flux is evaluated; see ``convert_flux``.
    return_quantity : bool, optional
        Return a Quantity instead of plain values.

    Returns
    -------
    scalar, ndarray or Quantity
        Flux in ``to_unit``.
    """
    flux_new, _ = convert_flux(flux, from_unit, to_unit, wave=wave, return_quantity=return_quantity)
    return flux_new