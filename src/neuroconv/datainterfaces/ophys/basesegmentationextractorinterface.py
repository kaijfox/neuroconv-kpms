"""Authors: Heberto Mayorquin, Cody Baker and Ben Dichter."""
from typing import Optional

from pynwb import NWBFile
from pynwb.device import Device
from pynwb.ophys import Fluorescence, ImageSegmentation, ImagingPlane, TwoPhotonSeries
from roiextractors import SegmentationExtractor

from ...basedatainterface import BaseDataInterface
from ...tools import get_package
from ...tools.roiextractors import write_segmentation, get_nwb_segmentation_metadata
from ...utils import get_schema_from_hdmf_class, fill_defaults, OptionalFilePathType, get_base_schema


class _LazySegmentationExtractor(type(BaseDataInterface), type):
    def __getattribute__(self, name):
        if name == "SegX" and super().__getattribute__("SegX") is None:
            roiextractors = get_package(package_name=self.SegXModule or "roiextractors")
            return getattr(roiextractors, self.SegXName or self.__name__.replace("Interface", "Extractor"))
        return super().__getattribute__(name)


class BaseSegmentationExtractorInterface(BaseDataInterface, metaclass=_LazySegmentationExtractor):
    """Parent class for all ImagingExtractorInterfaces."""

    SegXModule: Optional[str] = None  # Defaults to "roiextractors". Manually override in subclass if needed.
    SegXName: Optional[str] = None  # Defaults to __name__.replace("Interface", "Extractor").
    SegX: SegmentationExtractor = None  # Loads dynamically on first access attempt

    def __new__(cls, *args, **kwargs):
        cls.SegX = getattr(cls, "SegX")
        return object.__new__(cls)

    def __init__(self, **source_data):
        super().__init__(**source_data)
        self.segmentation_extractor = self.SegX(**source_data)

    def get_metadata_schema(self):
        metadata_schema = super().get_metadata_schema()
        metadata_schema["required"] = ["Ophys"]
        metadata_schema["properties"]["Ophys"] = get_base_schema()
        metadata_schema["properties"]["Ophys"]["properties"] = dict(
            Device=dict(type="array", minItems=1, items=get_schema_from_hdmf_class(Device)),
        )
        metadata_schema["properties"]["Ophys"]["properties"].update(
            Fluorescence=get_schema_from_hdmf_class(Fluorescence),
            ImageSegmentation=get_schema_from_hdmf_class(ImageSegmentation),
            ImagingPlane=get_schema_from_hdmf_class(ImagingPlane),
            TwoPhotonSeries=get_schema_from_hdmf_class(TwoPhotonSeries),
        )
        metadata_schema["properties"]["Ophys"]["required"] = ["Device", "ImageSegmentation"]

        # Temporary fixes until centralized definition of metadata schemas
        metadata_schema["properties"]["Ophys"]["properties"]["ImagingPlane"].update(type="array")
        metadata_schema["properties"]["Ophys"]["properties"]["TwoPhotonSeries"].update(type="array")

        metadata_schema["properties"]["Ophys"]["properties"]["Fluorescence"]["properties"]["roi_response_series"][
            "items"
        ]["required"] = list()
        metadata_schema["properties"]["Ophys"]["properties"]["ImageSegmentation"]["additionalProperties"] = True
        metadata_schema["properties"]["Ophys"]["properties"]["Fluorescence"]["properties"]["roi_response_series"].pop(
            "maxItems"
        )
        metadata_schema["properties"]["Ophys"]["properties"]["DfOverF"] = metadata_schema["properties"]["Ophys"][
            "properties"
        ]["Fluorescence"]

        fill_defaults(metadata_schema, self.get_metadata())
        return metadata_schema

    def get_metadata(self):
        metadata = super().get_metadata()
        metadata.update(get_nwb_segmentation_metadata(self.segmentation_extractor))
        return metadata

    def run_conversion(
        self,
        nwbfile_path: OptionalFilePathType = None,
        nwbfile: Optional[NWBFile] = None,
        metadata: Optional[dict] = None,
        overwrite: bool = False,
        save_path: OptionalFilePathType = None,
    ):

        write_segmentation(
            self.segmentation_extractor,
            nwbfile_path=nwbfile_path,
            nwbfile=nwbfile,
            metadata=metadata,
            overwrite=overwrite,
            verbose=self.verbose,
            save_path=save_path,
        )
