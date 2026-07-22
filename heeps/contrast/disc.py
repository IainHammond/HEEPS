"""
HEEPS disc injection and post‑processing utilities.

This module provides functions to inject a synthetic disc model into HEEPS
PSF data and to perform post‑processing with VIP.
The entry point is :func:`create_disc_sequence`, which creates a mock
observation sequence with a user-provided disc model. The :func:`postproc_disc` function is a
wrapper of the relevant VIP function for PCA post-processing.
"""

import numpy as np
from typing import Union
import os

from vip_hci.fits import open_fits, open_header
from vip_hci.preproc import frame_crop, cube_crop_frames, frame_shift
from vip_hci.fm import cube_inject_fakedisk

from heeps.util.psf_template import psf_template
from heeps.util.paralang import paralang

__all__ = ['create_disc_sequence', 'prepare_rdi_sequence']
__author__ = "Iain Hammond"

def create_disc_sequence(
        disc_model : np.ndarray,
        db_path : str = "vortex_psf_grid/",
        on_axis_psf : Union[np.ndarray, None] = None,
        off_axis_psf : Union[np.ndarray, None] = None,
        transmission : Union[np.ndarray, None] = None,
        seeing_q : Union[int] = 2,
        source_xy : Union[tuple, list, None] = None,
        extinction : Union[int, float] = 0,
        starphot : Union[float, int] = 1e11,
        imlib : str = "opencv",
        **conf
):
    """
    Create a mock observation sequence by injecting a synthetic disc model into a HEEPS PSF. Existing on- and off-axis
    PSF cubes and coronagraph transmission can be provided. If not provided, the function will attempt to load them
    from the PSF grid directory based on the conf parameters.

    Parameters
    ----------
    disc_model : np.ndarray
        2‑D array representing the disc model to be injected. NaNs are replaced with zeros. Odd-sized models with the
        stellar flux on the centre pixel are preferred.
    db_path : str
        Path to the PSF grid directory (the folder that contains run sub-directories). Required if on_axis_psf or
        off_axis_psf are not provided.
    on_axis_psf : Union[np.ndarray, None], optional
        Pre‑loaded on‑axis PSF cube. If ``None``, the PSF is loaded from the output directory
        using the conf parameters.
    off_axis_psf : Union[np.ndarray, None], optional
        Pre‑loaded off‑axis PSF frame. If ``None``, the PSF is loaded from the output directory.
    transmission : Union[np.ndarray, None], optional
        Coronagraph transmission. If ``None``, the appropriate transmission is loaded based
        on the instrument mode in conf. Uses VIP conventions.
    seeing_q : int, default=2
        Seeing quartile (1-3) to select the PSF. Uses a default seeing of Q2 if not specified.
    source_xy : Union[tuple, list, None], optional
        (x, y) pixel coordinates of the source for which extinction is applied.
    extinction : Union[int, float], default=0
        Extinction in magnitudes to apply to the source at ``source_xy``.
    starphot : float, default=1e11
        Photometric scaling factor for the star.
    imlib : str, default="opencv"
        Image library to use for image processing. Use "opencv" to go fast, or "vip-fft" for better flux conservation
        at the cost of speed. See VIP documentation for details.
    conf : dict, optional
        Configuration dictionary with parameters from HEEPS (e.g., background addition etc.).

    Returns
    -------
    psf_ON : numpy.ndarray
        Sequence with the fake disc injected into the on‑axis PSF cube.
    pa : numpy.ndarray
        Array of parallactic angles used for the injection.
    """
    # if mag is not in the grid, round to the closest available magnitude
    conf["mag"] = _check_mag(conf["mag"])

    # if one or both of the PSFs are not provided, prepare the path for loading them from the PSF grid directory
    if on_axis_psf is None or off_axis_psf is None:
        # make sure db_path ends with a slash
        if not db_path.endswith("/"):
            db_path += "/"
            
        # new grid folder structure and naming convention for the April 2026 grid files
        grid_dir = os.path.join(
            db_path,
            f"{conf['band']}_{conf['mode']}_s=Q{seeing_q}_mag={conf['mag']}_{conf['duration']}s_{int(conf['dit'] * 1000)}ms"
        )
        loadname = os.path.join(grid_dir, '%s_PSF_bckg1_%s_%s.fits' % ('%s', conf['band'], conf['mode']))

    # set nans to zero
    disc_model = np.nan_to_num(disc_model, nan=0)

    # remove any extra dimensions (e.g., MCFOST cubes)
    # first, squeeze singleton dimensions
    disc_model = np.squeeze(disc_model)
    # then collapse any remaining leading dimensions to a 2-D image
    while disc_model.ndim > 2:
        disc_model = disc_model[0]

    # record the stellar flux in the model for flux normalisation, and set its pixel to 0 (it will be in the on-axis PSF)
    # some codes have the star on the centre 4 pixels instead of the centre pixel only so we need to sum the values
    mask = disc_model == disc_model.max()
    star_val = disc_model[mask].sum()
    disc_model[mask] = 0

    # ensure the model has an odd size
    if disc_model.shape[-1] % 2 == 0:
        print("Converting input disc model to odd-dimensions.", flush=True)
        disc_model = frame_shift(disc_model, shift_y=0.5, shift_x=0.5, imlib=imlib)
        disc_model = disc_model[1:, 1:]

    if off_axis_psf is None:
        psf_OFF = open_fits(loadname % 'offaxis', verbose=False)
        print(f"Using off-axis PSF from {loadname%'offaxis'}")
    else:
        psf_OFF = off_axis_psf
    assert psf_OFF.ndim == 2, "off-axis PSF frame must be 2-dimensional"

    # crop everything to a common size
    min_crop = min(psf_OFF.shape[-1], disc_model.shape[-1], conf["ndet"])
    conf["ndet"] = min_crop
    if psf_OFF.shape[-1] > min_crop:
        psf_OFF = frame_crop(psf_OFF, min_crop, verbose=False)
    if disc_model.shape[-1] > min_crop:
        disc_model = frame_crop(disc_model, min_crop, verbose=False)

    # apply extinction if requested
    if extinction > 0 and source_xy is not None:
        source_x, source_y = source_xy
        extinction_factor = 10 ** (-0.4 * extinction)
        disc_model[source_y, source_x] *= extinction_factor
        print("Applied extinction of %.2f mag to source at (x, y) = (%d, %d)" % (extinction, source_x, source_y), flush=True)

    # load coronagraph transmission depending on the mode and band
    if transmission is None:
        if conf["mode"] == "ELT" or conf["mode"] is None:  # no coronagraph
            transmission = None
        elif "VC" in conf["mode"]:
            if conf["band"] in ("L", "M"):
                conf['f_oat'] = conf['dir_input'] + 'optics/vc/oat_L_%s.fits' % conf['mode']
            elif conf["band"] == "N2":
                conf['f_oat'] = conf['dir_input'] + 'optics/vc/oat_N2_CVC.fits'
            transmission = open_fits(conf['f_oat'], verbose=False)
            print(f"Using transmission from {conf['f_oat']}")
        else: # TODO
            raise NotImplementedError(f"Mode {conf['mode']} not implemented for disc injection.")

    # determine how many frames are in the on-axis sequence
    if on_axis_psf is None:
        nframes = open_header(loadname % 'onaxis')['NAXIS3']  # don't open the whole cube yet to save memory
    else:
        nframes = on_axis_psf.shape[0]

    # generate parallactic angles and inject the fake disc using VIP
    pa = paralang(npts=nframes, dec=conf["dec"], lat=conf["lat"], duration=int(nframes * conf["dit"]))

    cube = cube_inject_fakedisk(
        disc_model,
        pa,
        transmission=transmission,
        psf=psf_OFF,
        normalize_psf=False,
        nproc=conf["cpu_count"],
        mask_val=0,
        edge_blend="interp",
        interp_zeros=True,
        ker=1,
        imlib=imlib,
    )
    # write_fits(conf["dir_current"] + "/prepared_model.fits", cube[0])
    # normalise by the stellar flux that we removed earlier
    cube /= star_val

    # open and crop sequence if needed
    if on_axis_psf is None:
        psf_ON = open_fits(loadname % 'onaxis', verbose=False)
        print(f"Using on-axis PSF from {loadname % 'onaxis'}")
    else:
        psf_ON = on_axis_psf
    assert psf_ON.ndim == 3, "on-axis PSF cube must be 3-dimensional"

    if psf_ON.shape[-1] > min_crop:
        psf_ON = cube_crop_frames(psf_ON, min_crop, verbose=False)

    # add the disc model
    psf_ON += cube
    del cube

    if starphot is not None:
        _, _, ap_flux = psf_template(psf_OFF)
        psf_ON *= starphot / ap_flux
    del psf_OFF

    return psf_ON, pa


def prepare_rdi_sequence(
        db_path : str = "vortex_psf_grid/",
        on_axis_psf : Union[np.ndarray, None] = None,
        off_axis_psf : Union[np.ndarray, None] = None,
        rdi_mag : Union[int, float, None] = None,
        rdi_duration : Union[int, float, None] = None,
        seeing_q : Union[int] = 2,
        starphot : Union[float, int] = 1e11,
        **conf
):
        """
        Prepare a sequence from the PSF grid to be used as a reference for RDI (Reference Differential Imaging).
        The function extracts a random segment of the on-axis PSF cube depending on the specified duration.

        Parameters
        ----------
        db_path : str
            Path to the PSF grid directory (the folder that contains run sub-directories). Required if on_axis_psf or
            off_axis_psf are not provided.
        on_axis_psf : Union[np.ndarray, None], optional
            Pre‑loaded on‑axis PSF cube. If ``None``, the PSF is loaded based off conf and rdi_mag.
            using the conf parameters.
        off_axis_psf : Union[np.ndarray, None], optional
            Pre‑loaded off‑axis PSF frame. If ``None``, the PSF is loaded based off conf and rdi_mag.
        rdi_mag : Union[int, float, None], optional
            Magnitude to use for the RDI reference. If ``None``, magnitude in conf is used.
        rdi_duration : Union[int, float, None], optional
            Desired duration (in seconds) of the RDI reference sequence. If ``None``, a default of
            20% of the on‑axis sequence length is used.
        seeing_q : int, default=2
            Seeing quartile (1-3) to select the PSF. Uses a default seeing of Q2 if not specified.
        starphot : float, default=1e11
            Photometric scaling factor for the star.

        Returns
        -------
        psf_RDI : numpy.ndarray
            Reference cube for RDI.
        """
        if rdi_mag is None:
            rdi_mag = conf["mag"]
            print("Warning: rdi_mag was not set. Using mag in the conf dictionary.", flush=True)

        rdi_mag = _check_mag(rdi_mag)

        grid_dir = os.path.join(
            db_path,
            f"{conf['band']}_{conf['mode']}_s=Q{seeing_q}_mag={rdi_mag}_{conf['duration']}s_{int(conf['dit'] * 1000)}ms"
        )

        # convention for the April 2026 grid files, assuming dir_current points to the folder containing the PSF files
        loadname = os.path.join(grid_dir, '%s_PSF_bckg1_%s_%s.fits' % ('%s', conf['band'], conf['mode']))

        if off_axis_psf is None:
            psf_OFF = open_fits(loadname % 'offaxis', verbose=False)
            print(f"Using off-axis PSF from {loadname%'offaxis'}")
        else:
            psf_OFF = off_axis_psf
        assert psf_OFF.ndim == 2, "off-axis PSF frame must be 2-dimensional"

        if on_axis_psf is None:
            psf_RDI = open_fits(loadname % 'onaxis', verbose=False)
            print(f"Using on-axis PSF from {loadname % 'onaxis'}")
        else:
            psf_RDI = on_axis_psf
        assert psf_RDI.ndim == 3, "on-axis PSF cube must be 3-dimensional"

        if psf_OFF.shape[-1] > conf["ndet"]:
            psf_OFF = frame_crop(psf_OFF, conf["ndet"], verbose=False)

        if psf_RDI.shape[-1] > conf["ndet"]:
            psf_RDI = cube_crop_frames(psf_RDI, conf["ndet"], verbose=False)

        if rdi_duration is None:
            rdi_duration = int(psf_RDI.shape[0] * 0.2) * conf["dit"]
            print("Warning: rdi_duration was not set. Using 20% of full sequence.", flush=True)

        # convert into number of frames to extract from the on-axis PSF cube
        n_ref_frames = rdi_duration / conf["dit"]

        # check if the requested reference duration is longer than the available on-axis PSF cube
        if n_ref_frames > psf_RDI.shape[0]:
            n_ref_frames = psf_RDI.shape[0]
            print(f"WARNING! Requested RDI duration ({rdi_duration}s) exceeded available on-axis PSF cube length "
                  f"({psf_RDI.shape[0] * conf['dit']}s). Setting them to be the same.", flush=True)

        # random segment of the on-axis PSF cube to use as the RDI reference
        start_idx = np.random.randint(low=0, high=psf_RDI.shape[0] - int(n_ref_frames) + 1)
        psf_RDI = psf_RDI[start_idx:start_idx + int(n_ref_frames)]

        if starphot is not None:
            _, _, ap_flux = psf_template(psf_OFF)
            psf_RDI *= starphot / ap_flux
        del psf_OFF

        return psf_RDI


def _check_mag(mag : Union[float, int]) -> float:
    """
    Check if the provided magnitude is in the Liege grid. If not, round to the closest available magnitude.

    Parameters
    ----------
    mag : float, int
        The magnitude to check.

    Returns
    -------
    mag : float
        The closest available magnitude in the Liege grid.
    """
    mag_og = float(mag)

    allowed = np.arange(-1.5, 9.0 + 0.5, step=0.5)

    if mag not in allowed:
        mag = round(mag / 0.5) * 0.5

    # clamp to valid range
    if mag < allowed[0]:
        mag = allowed[0]
        print(f"Attention: mag={mag_og} is below minimum. Using mag={mag}")
    elif mag > allowed[-1]:
        mag  = allowed[-1]
        print(f"Attention: mag={mag_og} is above maximum. Using mag={mag}")
    elif mag != mag_og:
        print(f"Attention: mag={mag_og} rounded to nearest grid value mag={mag}")

    return float(mag)
