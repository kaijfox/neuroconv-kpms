from ....tools import get_package

from pynwb.behavior import SpatialSeries, Position, CompassDirection
from pynwb.base import TimeSeries
from neuroconv.tools import nwb_helpers
import numpy as np


def get_video_timestamps(
    session_name,
    video_dir,
):
    """
    Get timestamps from a the video file matching a session name

    Parameters
    ----------
    session_name : str
        Identifier for the session as used in DLC/SLEAP/kpms

    video_dir : str
        Directory in which to search for video files

    Returns
    -------
    timestamps : np.array (n_frames,)
        Array of timestamps, in seconds.
    """

    # pulled from : github.com/DeepLabCut/DLC2NWB

    iio = get_package('imageio.v3')
    util = get_package('keypoint_moseq.util')

    
    video_path = util.find_matching_videos([session_name],video_dir)[0]

    meta = iio.immeta(video_path, plugin = 'pyav')
    if 'duration' not in meta or 'fps' not in meta:
        raise IOError(
            "Missing duration / fps to read timestamps "
            f"from video file: {video_path}")
    n_frames = int(meta['duration'] / (1 / meta['fps']))
    timestamps = np.arange(n_frames) * (1/meta['fps'])
    return timestamps



def dense_syllables_to_events(syllables):
    """
    Convert syllable labels at each timestamps to an array of changepoints.
    
    Parameters
    ----------
    syllables : array (n_frames,)
        Syllable label for each frame.
        
    Returns
    -------
    ixs : array (n_changept,)
        Frame indices of the onset of each new syllable.
        
    labels : array (n_changept,)
        Label of the syllable switched to at each changepoint.
    """
    changepoints = np.where(syllables[1:] != syllables[:-1])[0]
    return changepoints, syllables[changepoints]



def write_subject_to_nwb(
    nwbfile,
    pose_arr,
    centroid_arr,
    heading_arr,
    latents_arr,
    syllables_arr,
    timestamps,
    kpms_model,
    kpms_version,
    bodyparts,
    skeleton = None,
):
    """
    Insert arrays from results file into an NWB file.
    
    Parameters
    ----------
    pose_arr : array (frames, keypts, spatial_dim)
    """

    ndx_pose = get_package('ndx_pose')
    ndx_events = get_package('ndx_events')

    # ----- set up data containers: spatial

    position_spatial_series = SpatialSeries(
        name="centroid_series",
        description="Position (x, y) of the subject's centroid.",
        data=centroid_arr,
        timestamps=timestamps,
        reference_frame=("origin corresponds to the top left corner of "
                         "the video."),
        unit="pixels",
    )
    position = Position(spatial_series = position_spatial_series, name='centroid')
    timestamps_ds = position_spatial_series.timestamps

    direction_spatial_series = SpatialSeries(
        name="heading_series",
        description=(
            "Inferred heading direction of the mouse."
        ),
        data=heading_arr,
        timestamps=timestamps_ds,
        reference_frame="0 points along positive x axis, towards the right side of the "
                        "video frame.",
        unit="radians",
    )
    direction = CompassDirection(spatial_series=direction_spatial_series, name="heading")


    # ----- set up data containers: pose

    pose_estimation_series = []
    for i_keypt, keypt_name in enumerate(bodyparts):

        pes = ndx_pose.PoseEstimationSeries(
            name=f"{keypt_name}",
            description=(f"Unaligned keypoint-moseq estimated keypoint "
                         f"{keypt_name}."),
            data=pose_arr[:, i_keypt],
            unit="pixels",
            reference_frame=("origin corresponds to the bottom left corner of "
                             "the video."),
            timestamps=timestamps_ds
        )
        pose_estimation_series.append(pes)

    pe = ndx_pose.PoseEstimation(
        pose_estimation_series=pose_estimation_series,
        description="Inferred denoised keypoint coordinates from keypoint-moseq.",
        scorer=kpms_model,
        source_software="keypoint-moseq",
        source_software_version=kpms_version,
        nodes=[pes.name for pes in pose_estimation_series],
        edges=skeleton if skeleton else None,
    )

    # ----- set up data containers: latents / syllables

    latents = TimeSeries(
        name='pose_latents',
        description="Latent pose states estimated by keypoint-moseq.",
        data=latents_arr,
        timestamps = timestamps,
        unit='n/a'
    )

    syll_ix, syll_lab = dense_syllables_to_events(syllables_arr)
    syllables = ndx_events.LabeledEvents(
        name='syllable',
        description="Syllable onset times.",
        timestamps=timestamps[syll_ix],
        data=syll_lab,
        labels=[f"Syllable {i}" for i in range(np.max(syll_lab+1))]
    )

    # ----- add data containers to nwb file

    behavior_pm = nwb_helpers.get_module(
        nwbfile,
        name="behavior",
        description="Processed behavioral data")
    behavior_pm.add(position)
    behavior_pm.add(direction)
    behavior_pm.add(pe)
    behavior_pm.add(latents)
    behavior_pm.add(syllables)

    return nwbfile, position_spatial_series.timestamps

