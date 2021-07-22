from tempfile import mkdtemp
from shutil import rmtree
from pathlib import Path

from pynwb import NWBFile

try:
    from ndx_events import LabeledEvents
    HAVE_NDX_EVENTS = True
except ImportError:
    HAVE_NDX_EVENTS = False

from nwb_conversion_tools.basedatainterface import BaseDataInterface
from nwb_conversion_tools import NWBConverter, TutorialRecordingInterface


def test_converter():
    if HAVE_NDX_EVENTS:
        test_dir = Path(mkdtemp())
        nwbfile_path = str(test_dir / "extension_test.nwb")

        class NdxEventsInterface(BaseDataInterface):
            def run_conversion(self, nwbfile: NWBFile, metadata: dict):
                events = LabeledEvents(
                    name="LabeledEvents",
                    description="events from my experiment",
                    timestamps=[0., 0.5, 0.6, 2., 2.05, 3., 3.5, 3.6, 4.],
                    resolution=1e-5,
                    data=[0, 1, 2, 3, 5, 0, 1, 2, 4],
                    labels=["trial_start", "cue_onset", "cue_offset", "response_left", "response_right", "reward"]
                )
                nwbfile.add_acquisition(events)

        class ExtensionTestNWBConverter(NWBConverter):
            data_interface_classes = dict(NdxEvents=NdxEventsInterface)

        converter = ExtensionTestNWBConverter(source_data=dict(NdxEvents=dict()))
        converter.run_conversion(nwbfile_path=nwbfile_path, overwrite=True)

        rmtree(test_dir)
