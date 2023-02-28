"""Author: Luiz Tauffer."""
import distutils.version
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import neo.io.baseio
import numpy as np
import pynwb
from hdmf.backends.hdf5 import H5DataIO

from ..nwb_helpers import add_device_from_metadata
from ...utils import OptionalFilePathType

response_classes = dict(
    voltage_clamp=pynwb.icephys.VoltageClampSeries,
    current_clamp=pynwb.icephys.CurrentClampSeries,
    izero=pynwb.icephys.IZeroClampSeries,
)

stim_classes = dict(
    voltage_clamp=pynwb.icephys.VoltageClampStimulusSeries,
    current_clamp=pynwb.icephys.CurrentClampStimulusSeries,
)


# TODO - get electrodes metadata
def get_electrodes_metadata(neo_reader, electrodes_ids: list, block: int = 0) -> list:
    """
    Get electrodes metadata from Neo reader.

    The typical information we look for is the information accepted by pynwb.icephys.IntracellularElectrode:
      - name – the name of this electrode
      - device – the device that was used to record from this electrode
      - description – Recording description, description of electrode (e.g., whole-cell, sharp, etc)
      - comment: Free-form text (can be from Methods)
      - slice – Information about slice used for recording.
      - seal – Information about seal used for recording.
      - location – Area, layer, comments on estimation, stereotaxis coordinates (if in vivo, etc).
      - resistance – Electrode resistance COMMENT: unit: Ohm.
      - filtering – Electrode specific filtering.
      - initial_access_resistance – Initial access resistance.

    Parameters
    ----------
    neo_reader ([type]): Neo reader
    electrodes_ids (list): List of electrodes ids.
    block (int, optional): Block id. Defaults to 0.

    Returns
    -------
    list: List of dictionaries containing electrodes metadata.
    """
    return []


def get_number_of_electrodes(neo_reader) -> int:
    """Get number of electrodes from Neo reader."""
    # TODO - take in account the case with multiple streams.
    return len(neo_reader.header["signal_channels"])


def get_number_of_segments(neo_reader, block: int = 0) -> int:
    """Get number of segments from Neo reader."""
    return neo_reader.header["nb_segment"][block]


def get_command_traces(neo_reader, segment: int = 0, cmd_channel: int = 0) -> Tuple[list, str, str]:
    """
    Get command traces (e.g. voltage clamp command traces).

    Parameters
    ----------
    neo_reader : neo.io.baseio
    segment : int, optional
        Defaults to 0.
    cmd_channel : int, optional
        ABF command channel (0 to 7). Defaults to 0.
    """
    try:
        traces, titles, units = neo_reader.read_raw_protocol()
        return traces[segment][cmd_channel], titles[segment][cmd_channel], units[segment][cmd_channel]
    except Exception as e:
        msg = ".\n\n WARNING - get_command_traces() only works for AxonIO interface."
        e.args = (str(e) + msg,)
        return e


def get_conversion_from_unit(unit: str) -> float:
    """
    Get conversion (to Volt or Ampere) from unit in string format.

    Parameters
    ----------
    unit (str): Unit as string. E.g. pA, mV, uV, etc...

    Returns
    -------
    float: conversion to Ampere or Volt
    """
    if unit in ["pA", "pV"]:
        conversion = 1e-12
    elif unit in ["nA", "nV"]:
        conversion = 1e-9
    elif unit in ["uA", "uV"]:
        conversion = 1e-6
    elif unit in ["mA", "mV"]:
        conversion = 1e-3
    elif unit in ["A", "V"]:
        conversion = 1.0
    else:
        conversion = 1.0
        warnings.warn("No valid units found for traces in the current file. Gain is set to 1, but this might be wrong.")
    return float(conversion)


def get_nwb_metadata(neo_reader, metadata: dict = None) -> dict:
    """
    Return default metadata for all recording fields.

    Parameters
    ----------
    neo_reader: Neo reader object
    metadata: dict, optional
        Metadata info for constructing the nwb file.
    """
    metadata = dict(
        NWBFile=dict(
            session_description="Auto-generated by NwbRecordingExtractor without description.",
            identifier=str(uuid.uuid4()),
        ),
        Icephys=dict(Device=[dict(name="Device", description="no description")]),
    )
    return metadata


def add_icephys_electrode(neo_reader, nwbfile, metadata: dict = None):
    """
    Add icephys electrodes to nwbfile object.

    Will always ensure nwbfile has at least one icephys electrode.
    Will auto-generate a linked device if the specified name does not exist in the nwbfile.

    Parameters
    ----------
    neo_reader : neo.io.baseio
    nwbfile : NWBFile
        NWBFile object to add the icephys electrode to.
    metadata : dict, optional
        Metadata info for constructing the nwb file.
        Should be of the format
            metadata['Icephys']['Electrodes'] = [
                {
                    'name': my_name,
                    'description': my_description,
                    'device_name': my_device_name
                },
                ...
            ]
    """
    assert isinstance(nwbfile, pynwb.NWBFile), "'nwbfile' should be of type pynwb.NWBFile"

    if len(nwbfile.devices) == 0:
        warnings.warn("When adding Icephys Electrode, no Devices were found on nwbfile. Creating a Device now...")
        add_device_from_metadata(nwbfile=nwbfile, modality="Icephys", metadata=metadata)

    if metadata is None:
        metadata = dict()

    if "Icephys" not in metadata:
        metadata["Icephys"] = dict()

    defaults = [
        dict(
            name=f"icephys_electrode_{elec_id}",
            description="no description",
            device_name=[i.name for i in nwbfile.devices.values()][0],
        )
        for elec_id in range(get_number_of_electrodes(neo_reader))
    ]

    if "Electrodes" not in metadata["Icephys"] or len(metadata["Icephys"]["Electrodes"]) == 0:
        metadata["Icephys"]["Electrodes"] = defaults

    assert all(
        [isinstance(x, dict) for x in metadata["Icephys"]["Electrodes"]]
    ), "Expected metadata['Icephys']['Electrodes'] to be a list of dictionaries!"

    # Create Icephys electrode from metadata
    for elec in metadata["Icephys"]["Electrodes"]:
        if elec.get("name", defaults[0]["name"]) not in nwbfile.icephys_electrodes:
            device_name = elec.get("device_name", defaults[0]["device_name"])
            if device_name not in nwbfile.devices:
                new_device_metadata = dict(Ecephys=dict(Device=[dict(name=device_name)]))
                add_device_from_metadata(nwbfile, modality="Icephys", metadata=new_device_metadata)
                warnings.warn(
                    f"Device '{device_name}' not detected in "
                    "attempted link to icephys electrode! Automatically generating."
                )
            electrode_kwargs = dict(
                name=elec.get("name", defaults[0]["name"]),
                description=elec.get("description", defaults[0]["description"]),
                device=nwbfile.devices[device_name],
            )
            nwbfile.create_icephys_electrode(**electrode_kwargs)


def add_icephys_recordings(
    neo_reader,
    nwbfile: pynwb.NWBFile,
    metadata: dict = None,
    icephys_experiment_type: str = "voltage_clamp",
    stimulus_type: str = "not described",
    skip_electrodes: Tuple[int] = (),
    compression: str = "gzip",
):
    """
    Add icephys recordings (stimulus/response pairs) to nwbfile object.

    Parameters
    ----------
    neo_reader : neo.io.baseio
    nwbfile : NWBFile
    metadata : dict, optional
    icephys_experiment_type : {'voltage_clamp', 'current_clamp', 'izero'}
        Type of icephys recording.
    stimulus_type : str, default: 'not described'
    skip_electrodes : tuple, default: ()
        Electrode IDs to skip.
    compression : str | bool
    """
    n_segments = get_number_of_segments(neo_reader, block=0)

    # Check for protocol data (only ABF2), necessary for stimuli data
    if neo_reader._axon_info["fFileVersionNumber"] < 2:
        n_commands = 0
        warnings.warn(
            f"Protocol section is only present in ABF2 files. {neo_reader.filename} has version "
            f"{neo_reader._axon_info['fFileVersionNumber']}. Saving experiment as 'i_zero'..."
        )
    else:
        protocol = neo_reader.read_raw_protocol()
        n_commands = len(protocol[0])

    if n_commands == 0:
        icephys_experiment_type = "izero"
        warnings.warn(
            f"No command data found by neo reader in file {neo_reader.filename}. Saving experiment as 'i_zero'..."
        )
    else:
        assert (
            n_commands == n_segments
        ), f"File contains inconsistent number of segments ({n_segments}) and commands ({n_commands})"

    assert icephys_experiment_type in ["voltage_clamp", "current_clamp", "izero"], (
        f"'icephys_experiment_type' should be 'voltage_clamp', 'current_clamp' or 'izero', but received value "
        f"{icephys_experiment_type}"
    )

    # Check and auto-create electrodes, in case they don't exist yet in nwbfile
    if len(nwbfile.icephys_electrodes) == 0:
        warnings.warn(
            "When adding Icephys Recording, no Icephys Electrodes were found on nwbfile. Creating Electrodes now..."
        )
        add_icephys_electrode(
            neo_reader=neo_reader,
            nwbfile=nwbfile,
            metadata=metadata,
        )

    if getattr(nwbfile, "intracellular_recordings", None):
        ri = max(nwbfile.intracellular_recordings["responses"].index)
    else:
        ri = -1

    if getattr(nwbfile, "icephys_simultaneous_recordings", None):
        simultaneous_recordings_offset = len(nwbfile.icephys_simultaneous_recordings)
    else:
        simultaneous_recordings_offset = 0

    if getattr(nwbfile, "icephys_sequential_recordings", None):
        sessions_offset = len(nwbfile.icephys_sequential_recordings)
    else:
        sessions_offset = 0

    relative_session_start_time = metadata["Icephys"]["Sessions"][sessions_offset]["relative_session_start_time"]
    session_stimulus_type = metadata["Icephys"]["Sessions"][sessions_offset]["stimulus_type"]

    # Sequential icephys recordings
    simultaneous_recordings = list()
    for si in range(n_segments):
        # Parallel icephys recordings
        recordings = list()
        for ei, electrode in enumerate(
            list(nwbfile.icephys_electrodes.values())[: len(neo_reader.header["signal_channels"]["units"])]
        ):
            if ei in skip_electrodes:
                continue
            # Starting time is the signal starting time within .abf file + time
            # relative to first session (first .abf file)
            ri += 1
            starting_time = neo_reader.get_signal_t_start(block_index=0, seg_index=si)
            starting_time = starting_time + relative_session_start_time

            sampling_rate = neo_reader.get_signal_sampling_rate()
            response_unit = neo_reader.header["signal_channels"]["units"][ei]
            response_conversion = get_conversion_from_unit(unit=response_unit)
            response_gain = neo_reader.header["signal_channels"]["gain"][ei]
            response_name = f"{icephys_experiment_type}-response-{si + 1 + simultaneous_recordings_offset:02}-ch-{ei}"

            response = response_classes[icephys_experiment_type](
                name=response_name,
                description=f"Response to: {session_stimulus_type}",
                electrode=electrode,
                data=H5DataIO(
                    data=neo_reader.get_analogsignal_chunk(block_index=0, seg_index=si, channel_indexes=ei),
                    compression=compression,
                ),
                starting_time=starting_time,
                rate=sampling_rate,
                conversion=response_conversion * response_gain,
                gain=np.nan,
            )
            if icephys_experiment_type != "izero":
                stim_unit = protocol[2][ei]
                stim_conversion = get_conversion_from_unit(unit=stim_unit)
                stimulus = stim_classes[icephys_experiment_type](
                    name=f"stimulus-{si + 1 + simultaneous_recordings_offset:02}-ch-{ei}",
                    description=f"Stim type: {session_stimulus_type}",
                    electrode=electrode,
                    data=protocol[0][si][ei],
                    rate=sampling_rate,
                    starting_time=starting_time,
                    conversion=stim_conversion,
                    gain=np.nan,
                )
                icephys_recording = nwbfile.add_intracellular_recording(
                    electrode=electrode, response=response, stimulus=stimulus
                )
            else:
                icephys_recording = nwbfile.add_intracellular_recording(electrode=electrode, response=response)

            recordings.append(icephys_recording)

        sim_rec = nwbfile.add_icephys_simultaneous_recording(recordings=recordings)
        simultaneous_recordings.append(sim_rec)

    nwbfile.add_icephys_sequential_recording(
        simultaneous_recordings=simultaneous_recordings, stimulus_type=stimulus_type
    )

    # TODO
    # # Add a list of sequential recordings table indices as a repetition
    # run_index = nwbfile.add_icephys_repetition(
    #     sequential_recordings=[
    #         seq_rec,
    #     ]
    # )

    # # Add a list of repetition table indices as a experimental condition
    # nwbfile.add_icephys_experimental_condition(
    #     repetitions=[
    #         run_index,
    #     ]
    # )


def add_all_to_nwbfile(
    neo_reader,
    nwbfile: pynwb.NWBFile = None,
    metadata: dict = None,
    compression: Optional[str] = "gzip",
    icephys_experiment_type: str = "voltage_clamp",
    stimulus_type: Optional[str] = None,
    skip_electrodes: Tuple[int] = (),
):
    """
    Auxiliary static method for nwbextractor.

    Adds all recording related information from recording object and metadata to the nwbfile object.

    Parameters
    ----------
    neo_reader: Neo reader object
    nwbfile: NWBFile
        nwb file to which the recording information is to be added
    metadata: dict
        metadata info for constructing the nwb file (optional).
        Check the auxiliary function docstrings for more information
        about metadata format.
    compression: str (optional, defaults to "gzip")
        Type of compression to use. Valid types are "gzip" and "lzf".
        Set to None to disable all compression.
    icephys_experiment_type: str (optional)
        Type of Icephys experiment. Allowed types are: 'voltage_clamp', 'current_clamp' and 'izero'.
        If no value is passed, 'voltage_clamp' is used as default.
    stimulus_type: str, optional
    skip_electrodes: tuple, optional
        Electrode IDs to skip. Defaults to ().
    """
    if nwbfile is not None:
        assert isinstance(nwbfile, pynwb.NWBFile), "'nwbfile' should be of type pynwb.NWBFile"

    add_device_from_metadata(nwbfile=nwbfile, modality="Icephys", metadata=metadata)

    add_icephys_electrode(
        neo_reader=neo_reader,
        nwbfile=nwbfile,
        metadata=metadata,
    )

    add_icephys_recordings(
        neo_reader=neo_reader,
        nwbfile=nwbfile,
        metadata=metadata,
        icephys_experiment_type=icephys_experiment_type,
        stimulus_type=stimulus_type,
        skip_electrodes=skip_electrodes,
        compression=compression,
    )


def write_neo_to_nwb(
    neo_reader: neo.io.baseio.BaseIO,
    save_path: OptionalFilePathType = None,  # pragma: no cover
    overwrite: bool = False,
    nwbfile=None,
    metadata: dict = None,
    compression: Optional[str] = "gzip",
    icephys_experiment_type: Optional[str] = None,
    stimulus_type: Optional[str] = None,
    skip_electrodes: Optional[tuple] = (),
):
    """
    Primary method for writing a Neo reader object to an NWBFile.

    Parameters
    ----------
    neo_reader: Neo reader
    save_path: PathType
        Required if an nwbfile is not passed. Must be the path to the nwbfile
        being appended, otherwise one is created and written.
    overwrite: bool
        If using save_path, whether to overwrite the NWBFile if it already exists.
    nwbfile: NWBFile
        Required if a save_path is not specified. If passed, this function
        will fill the relevant fields within the nwbfile.
    metadata: dict
        metadata info for constructing the nwb file (optional). Should be of the format
            metadata['Ecephys'] = {}
        with keys of the forms
            metadata['Ecephys']['Device'] = [
                {
                    'name': my_name,
                    'description': my_description
                },
                ...
            ]
            metadata['Ecephys']['ElectrodeGroup'] = [
                {
                    'name': my_name,
                    'description': my_description,
                    'location': electrode_location,
                    'device': my_device_name
                },
                ...
            ]
            metadata['Ecephys']['Electrodes'] = [
                {
                    'name': my_name,
                    'description': my_description
                },
                ...
            ]
            metadata['Ecephys']['ElectricalSeries'] = {
                'name': my_name,
                'description': my_description
            }

        Note that data intended to be added to the electrodes table of the NWBFile should be set as channel
        properties in the RecordingExtractor object.
    compression: str (optional, defaults to "gzip")
        Type of compression to use. Valid types are "gzip" and "lzf".
        Set to None to disable all compression.
    icephys_experiment_type: str (optional)
        Type of Icephys experiment. Allowed types are: 'voltage_clamp', 'current_clamp' and 'izero'.
        If no value is passed, 'voltage_clamp' is used as default.
    stimulus_type: str, optional
    skip_electrodes: tuple, optional
        Electrode IDs to skip. Defaults to ().
    """
    if nwbfile is not None:
        assert isinstance(nwbfile, pynwb.NWBFile), "'nwbfile' should be of type pynwb.NWBFile"

    assert (
        distutils.version.LooseVersion(pynwb.__version__) >= "1.3.3"
    ), "'write_neo_to_nwb' not supported for version < 1.3.3. Run pip install --upgrade pynwb"

    assert save_path is None or nwbfile is None, "Either pass a save_path location, or nwbfile object, but not both!"

    if metadata is None:
        metadata = get_nwb_metadata(neo_reader=neo_reader)

    kwargs = dict(
        neo_reader=neo_reader,
        metadata=metadata,
        compression=compression,
        icephys_experiment_type=icephys_experiment_type,
        stimulus_type=stimulus_type,
        skip_electrodes=skip_electrodes,
    )
    if nwbfile is None:
        if Path(save_path).is_file() and not overwrite:
            read_mode = "r+"
        else:
            read_mode = "w"

        with pynwb.NWBHDF5IO(str(save_path), mode=read_mode) as io:
            if read_mode == "r+":
                nwbfile = io.read()
            else:
                nwbfile_kwargs = dict(
                    session_description="Auto-generated by NwbRecordingExtractor without description.",
                    identifier=str(uuid.uuid4()),
                )
                if metadata is not None and "NWBFile" in metadata:
                    nwbfile_kwargs.update(metadata["NWBFile"])
                nwbfile = pynwb.NWBFile(**nwbfile_kwargs)

            add_all_to_nwbfile(nwbfile=nwbfile, **kwargs)
            io.write(nwbfile)
    else:
        add_all_to_nwbfile(nwbfile=nwbfile, **kwargs)
