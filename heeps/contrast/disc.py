"""
HEEPS disc injection and post‑processing utilities.

This module provides functions to inject a synthetic disc model into HEEPS
PSF data and to perform post‑processing with VIP.
The entry point is :func:`create_disc_sequence`, which creates a mock
observation sequence with a user-provided disc mode. The :func:`post_proc_disc` function is a
wrapper of the relevant VIP function for PCA post-processing.
"""

import numpy as np
from typing import Union
import os

from vip_hci.fits import open_fits, open_header
from vip_hci.preproc import frame_crop, cube_subsample, cube_crop_frames
from vip_hci.fm import cube_inject_fakedisk
from vip_hci.psfsub import pca

from heeps.contrast.background import background
from heeps.util.psf_template import psf_template
from heeps.util.paralang import paralang

__all__ = ['create_disc_sequence', 'postproc_disc']
__author__ = "Iain Hammond"

def create_disc_sequence(
        disc_model : np.ndarray,
        on_axis_psf : Union[np.ndarray, None] = None,
        off_axis_psf : Union[np.ndarray, None] = None,
        transmission : Union[np.ndarray, None] = None,
        rdi : bool = True,
        rdi_mag : Union[int, float, None] = None,
        rdi_duration : Union[int, float, None] = None,
        source_xy : Union[tuple, list, None] = None,
        extinction : Union[int, float] = 0,
        tag : Union[str, None] = None,
        starphot : float = 1e11,
        imlib : str = "opencv",
        **conf
):
    """
    Create a mock observation sequence by injecting a synthetic disc model into a HEEPS
    PSF and optionally generate a reference differential imaging (RDI)
    cube. Existing on- and off-axis PSF cubes can be provided.

    Parameters
    ----------
    disc_model : np.ndarray
        2‑D array representing the disc model to be injected. NaNs are replaced with zeros.
    on_axis_psf : Union[np.ndarray, None], optional
        Pre‑loaded on‑axis PSF cube. If ``None``, the PSF is loaded from the output directory
        using the conf parameters.
    off_axis_psf : Union[np.ndarray, None], optional
        Pre‑loaded off‑axis PSF frame. If ``None``, the PSF is loaded from the output directory.
    transmission : Union[np.ndarray, None], optional
        Coronagraph transmission. If ``None``, the appropriate transmission is loaded based
        on the instrument mode in conf. Uses VIP conventions.
    rdi : bool, default=True
        Whether to generate an RDI reference cube from a segment of the on‑axis sequence.
    rdi_mag : Union[int, float, None], optional
        Magnitude to use for the RDI reference. If ``None``, the magnitude of the science target
        is used.
    rdi_duration : Union[int, float, None], optional
        Desired duration (in seconds) of the RDI reference sequence. If ``None``, a default of
        20 % of the on‑axis sequence length is used.
    source_xy : Union[tuple, list, None], optional
        (x, y) pixel coordinates of the source for which extinction is applied.
    extinction : Union[int, float], default=0
        Extinction in magnitudes to apply to the source at ``source_xy``.
    tag : Union[str, None], optional
        Tag to prepend to output filenames from earlier HEEPS runs.
    starphot : float, default=1e11
        Photometric scaling factor for the star.
    imlib : str, default="opencv"
        Image library to use for image processing. Use "opencv" to go fast, or "vio-fft" for better flux conservation
        at the cost of speed. See VIP documentation for details.
    **conf : dict
        Additional conf parameters from HEEPS (e.g., background addition etc.)

    Returns
    -------
    psf_ON : numpy.ndarray
        Sequence with the fake disc injected into the on‑axis PSF cube.
    pa : numpy.ndarray
        Array of parallactic angles used for the injection.
    psf_RDI : numpy.ndarray, optional
        Reference cube for RDI (if ``rdi`` is ``True``). Returned as the third element when
        ``rdi`` is enabled; otherwise only ``psf_ON`` and ``pa`` are returned.
    """
    tag = "" if tag is None else "%s_"%tag
    loadname = os.path.join(conf['dir_output'], '%s%s_PSF_%s_%s.fits'%(tag,'%s', conf['band'], conf['mode']))
    if off_axis_psf is None:
        psf_OFF = open_fits(loadname % 'offaxis', verbose=False)
        print(f"Using off-axis PSF from {loadname%'offaxis'}")
    assert psf_OFF.ndim == 2, "off-axis PSF frame must be 2-dimensional"

    # set nans to zero
    disc_model = np.nan_to_num(disc_model, nan=0)

    # Remove any extra dimensions (e.g., MCFOST cubes)
    # first, squeeze singleton dimensions
    disc_model = np.squeeze(disc_model)
    # then collapse any remaining leading dimensions to a 2-D image
    while disc_model.ndim > 2:
        disc_model = disc_model[0]

    # ensure the model has an odd size
    if disc_model.shape[-1] % 2 == 0:
        disc_model = disc_model[1:, 1:]  # we assume the star flux is on one pixel

    # crop everything to a common size
    # the HEEPS PSFs are usually 293px
    min_crop = min(psf_OFF.shape[-1], disc_model.shape[-1], conf["ndet"])
    if psf_OFF.shape[-1] > min_crop:
        psf_OFF = frame_crop(psf_OFF, min_crop, verbose=False)
    if disc_model.shape[-1] > min_crop:
        disc_model = frame_crop(disc_model, min_crop, verbose=False)

    # record the stellar flux in the model and set the pixel to 0
    star_val = np.max(disc_model)
    star_y, star_x = np.unravel_index(np.argmax(disc_model), disc_model.shape)
    disc_model[star_y, star_x] = 0

    # apply extinction if requested
    if extinction > 0 and source_xy is not None:
        source_x, source_y = source_xy
        extinction_factor = 10 ** (-0.4 * extinction)
        disc_model[source_y, source_x] *= extinction_factor

    # load coronagraph transmission
    if transmission is None:
        if conf["mode"] == "ELT" or conf["mode"] is None:  # no coronagraph
            transmission = None
        elif conf["mode"] == "CVC":
                transmission = open_fits(conf["f_vc_trans"], verbose=False)
                print(f"Using transmission from {conf['f_vc_trans']}")
        else: # TODO
            raise NotImplementedError(f"Mode {conf['mode']} not implemented for disc injection.")

    # determine how many frames are in the on-axis sequence (don't open the whole cube yet to save memory)
    if on_axis_psf is None:
        nframes = open_header(loadname % 'onaxis')['NAXIS3']
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

    # normalise by the stellar flux that we removed earlier
    cube /= star_val

    # open and crop sequence if needed
    if on_axis_psf is None:
        psf_ON = open_fits(loadname % 'onaxis', verbose=False)
        print(f"Using on-axis PSF from {loadname % 'onaxis'}")

    assert psf_ON.ndim == 3, "on-axis PSF cube must be 3-dimensional"
    if psf_ON.shape[-1] > min_crop:
        psf_ON = cube_crop_frames(psf_ON, min_crop, verbose=False)

    # add the disc model
    psf_ON += cube

    # background addition
    if conf["add_bckg"]:
        psf_ON, psf_OFF = background(psf_ON, psf_OFF, verbose=True, **conf)

    _, _, ap_flux = psf_template(psf_OFF)
    psf_ON *= starphot / ap_flux

    # RDI handling (at the moment we only support taking a chunk out of the science sequence
    if rdi:
        if rdi_duration is None:
            print("Warning: rdi_duration was not set. Using 20% of on-axis sequence.", flush=True)
            rdi_duration = int(0.2 * nframes) * conf["dit"]
        n_ref_frames = rdi_duration / conf["dit"]
        start_idx = np.random.randint(0, psf_ON.shape[0] - int(n_ref_frames) + 1)  # random segment of the on-axis PSF cube to use as the RDI reference, to better match the background conditions
        psf_RDI = open_fits(loadname % "onaxis", verbose=False)[start_idx:start_idx + int(n_ref_frames)]
        if psf_RDI.shape[-1] > min_crop:
            psf_RDI = cube_crop_frames(psf_RDI, min_crop, verbose=False)

        # remove the RDI segment from psf_ON to correctly represent lost integration time on the science target
        psf_ON = np.concatenate([psf_ON[:start_idx], psf_ON[start_idx + int(n_ref_frames):]], axis=0)
        pa = np.concatenate([pa[:start_idx], pa[start_idx + int(n_ref_frames):]], axis=0)

        psf_RDI_OFF = open_fits(loadname % "offaxis", verbose=False)
        if psf_RDI_OFF.shape[-1] > min_crop:
            psf_RDI_OFF = frame_crop(psf_RDI_OFF, min_crop, verbose=False)

        if rdi_mag is None:
            print("Warning: rdi_mag was not set. Using mag of science target.", flush=True)
        else:
            conf["mag"] = rdi_mag

        if conf["add_bckg"]:
            psf_RDI, psf_RDI_OFF = background(psf_RDI, psf_RDI_OFF, verbose=True, **conf)

        _, _, ap_flux = psf_template(psf_RDI_OFF)
        psf_RDI *= starphot / ap_flux
        del psf_RDI_OFF

        return psf_ON, pa, psf_RDI

    else:
        return psf_ON, pa


def postproc_disc(
        cube : np.ndarray,
        angle_list : np.ndarray,
        cube_ref : Union[np.ndarray, None] = None,
        subsample : int = 1,
        ncomp: Union[tuple, list, int] = 1,
        mask_center_px : int = 0,
        imlib : str = "vip-fft",
        source_xy : Union[tuple, None] = None,
        delta_rot : Union[float, int, None] = 1,
        fwhm : Union[float, int] = 5,
        mask_rdi: np.ndarray = None,
        ref_strategy: str = 'RDI',
        nproc : int = 1
):
    """
    Perform PCA-based post-processing on a disc‑injected data cube.

    Parameters
    ----------
    cube : np.ndarray
        3‑D array (n_frames, ny, nx) containing the science frames.
    angle_list : np.ndarray
        1‑D array of parallactic angles associated with each frame in ``cube``.
    cube_ref : np.ndarray or None, optional
        Reference cube for RDI (reference differential imaging). If ``None``, only ADI is performed.
    subsample : int, default=1
        Factor by which to down‑sample the cube (and reference) for speed. ``1`` means no subsampling.
    ncomp : int, tuple, or list, default=1
        Number of principal components to use. If an int, a single component is used. If a tuple ``(min, max)`` or a
        list, the function will compute results for each component in the range.
    mask_center_px : int, default=0
        Radius (in pixels) of a circular mask applied to the centre of each frame.
    imlib : str, default="vip-fft"
        Image library used by VIP for FFT‑based operations.
    source_xy : tuple or None, optional
        (x, y) pixel coordinates of a known source; used for frame rejection in the PCA library.
    delta_rot : float or int or None, default=1
        Minimum rotation (in FWHM) between frames.
    fwhm : float or int, default=5
        Full‑width at half‑maximum of the PSF, used for delta_rot.
    mask_rdi : np.ndarray, optional
        See description in VIP.
    ref_strategy : str, default='RDI'
        Strategy for reference handling (RDI or ARDI)
    nproc : int, default=1
        Number of processors to use for parallel computation.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_ncomp, ny, nx)`` where ``n_ncomp`` is the number of principal components evaluated.
        Each slice ``res[i]`` contains the PCA‑processed image for the corresponding number of components.

    Notes
    -----
    The function currently supports the subset of ``vip_hci.psfsub.pca`` parameters used in the HEEPS disc injection
     workflow. Additional ``pca`` arguments can be added in the future.
    """
    # subsample the cube and the reference if requested, for efficiency purposes
    if subsample > 1:
        cube, angle_list = cube_subsample(array=cube, n=subsample, parallactic=angle_list)
        if cube_ref is not None:
            cube_ref = cube_subsample(array=cube_ref, n=subsample)

    # loop over principal components
    if isinstance(ncomp, int):
        ncomp = [ncomp]
    elif isinstance(ncomp, tuple):
        ncomp = np.arange(ncomp[0], ncomp[-1]+1)

    res = np.zeros([len(ncomp), cube.shape[-2], cube.shape[-1]])

    print("Running PCA post-processing", flush=True)
    for i, npc in enumerate(ncomp):
        res[i] = pca(cube, angle_list=angle_list, cube_ref=cube_ref, ncomp=npc,
                  mask_center_px=mask_center_px, imlib=imlib, nproc=nproc,
                  source_xy=source_xy, delta_rot=delta_rot, fwhm=fwhm, mask_rdi=mask_rdi,
                  ref_strategy=ref_strategy)
    return res
