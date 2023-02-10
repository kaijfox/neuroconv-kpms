"""The simplest, easiest to use class for converting all SpikeGLX data in a folder."""
from pathlib import Path
from typing import List, Optional

from pynwb import NWBFile
from pydantic import DirectoryPath

from .spikeglxdatainterface import SpikeGLXRecordingInterface, SpikeGLXLFPInterface
from .spikeglxnidqinterface import SpikeGLXNIDQInterface
from ....nwbconverter import ConverterPipe
from ....tools.nwb_helpers import make_or_load_nwbfile
from ....utils import get_schema_from_method_signature


class SpikeGLXConverter(ConverterPipe):
    """Primary conversion class for handling multiple SpikeGLX data streams."""

    @classmethod
    def get_source_schema(cls):
        source_schema = get_schema_from_method_signature(class_method=cls.__init__)
        source_schema["properties"]["folder_path"]["description"] = "Path to the folder containing SpikeGLX streams."
        return source_schema

    @classmethod
    def get_streams(cls, folder_path: DirectoryPath) -> List[str]:
        from spikeinterface.extractors import SpikeGLXRecordingExtractor

        available_streams = SpikeGLXRecordingExtractor.get_streams(folder_path=folder_path)[0]

    def __init__(
        self,
        folder_path: DirectoryPath,
        streams: Optional[List[str]] = None,
        verbose: bool = False,
    ):
        """
        Read all data from multiple streams stored in the SpikeGLX format.

        This can include...
            (a) single-probe but multi-band such as AP+LF streams
            (b) multi-probe with multi-band
            (c) with or without the associated NIDQ channels

        Parameters
        ----------
        folder_path : DirectoryPath
            Path to folder containing the NIDQ stream and subfolders containing each IMEC stream.
        streams : list of strings, optional
            A specific list of streams you wish to load.
            To see which streams are available, run `SpikeGLXConverter.get_streams(folder_path="path/to/spikeglx")`.
            By default, all available streams are loaded.
        verbose : bool, default: False
            Whether to output verbose text.
        """
        folder_path = Path(folder_path)

        available_streams = self.get_streams(folder_path=folder_path)
        streams = streams or available_streams

        data_interfaces = dict()
        for stream in streams:
            if "ap" in stream:
                probe_name = stream[:5]
                file_path = (
                    folder_path / f"{folder_path.stem}_{probe_name}" / f"{folder_path.stem}_t0.{probe_name}.ap.bin"
                )
                interface = SpikeGLXRecordingInterface(file_path=file_path)
            if "lf" in stream:
                probe_name = stream[:5]
                file_path = (
                    folder_path / f"{folder_path.stem}_{probe_name}" / f"{folder_path.stem}_t0.{probe_name}.ap.bin"
                )
                interface = SpikeGLXLFPInterface(file_path=file_path)
            if "nidq" in stream:
                file_path = folder_path / f"{folder_path.stem}_t0.nidq.bin"
                interface = SpikeGLXNIDQInterface(file_path=file_path)
            data_interfaces.update({stream: interface})

        super().__init__(data_interfaces=data_interfaces, verbose=verbose)

    def run_conversion(
        self,
        nwbfile_path: Optional[str] = None,
        nwbfile: Optional[NWBFile] = None,
        metadata: Optional[dict] = None,
        overwrite: bool = False,
    ):
        if metadata is None:
            metadata = self.get_metadata()
        self.validate_metadata(metadata=metadata)

        with make_or_load_nwbfile(
            nwbfile_path=nwbfile_path,
            nwbfile=nwbfile,
            metadata=metadata,
            overwrite=overwrite,
            verbose=self.verbose,
        ) as nwbfile_out:
            for interface_name, data_interface in self.data_interface_objects.items():
                data_interface.run_conversion(nwbfile=nwbfile_out, metadata=metadata)
