Keypoint-Moseq data conversion
------------------------------

First, install NeuroConv with the additional dependencies necessary to convert keypoint-moseq data.

.. code-block:: bash

    pip install neuroconv[keypointmoseq]


In general, when working with keypoint moseq, you will have a project directory that contains some number
of subfolders corresponding to your trained models. To convert the results from a particular tained keypoint moseq model to NWB,
use :py:class:`~neuroconv.datainterfaces.behavior.keypointmoseq.kpmsdatainterface.KeypointMoseqSubjectInterface`.

If you need a dataset to experiment with, we provide an `example DeepLabCut datset <https://drive.google.com/drive/folders/1UNHQ_XCQEKLPPSjGspRopWBj6-YNDV6G>`_
for the keypoint moseq `tutorial <https://keypoint-moseq.readthedocs.io/en/latest/tutorial.html>`_, where we walk through training a model
and extracting behavioral syllables.

.. code-block:: python

    from neuroconv import NWBConverter
    from neuroconv.datainterfaces import KeypointMoseqSubjectInterface
    import keypoint_moseq as kpms
    import datetime

    # set up an NWBConverter to run our conversion
    # add other DataInterfaces here according to any other data streams,
    # such as DeepLabCut or SLEAP keypoints
    class ExampleKpmsNWBConverter(NWBConverter):
        data_interface_classes = dict(
            KeypointMoseq=KeypointMoseqSubjectInterface,
        )

    # identify keypoint-moseq project and model, and the nwb output location
    model_folder = 'kpms_model'
    project_dir = 'demo_project/'
    nwb_path_fmt = 'data/{session}.nwb'

The :py:class:`~neuroconv.datainterfaces.behavior.keypointmoseq.kpmsdatainterface.KeypointMoseqSubjectInterface` class exposes a few tools
beyond the standard `DataInterface` to integrate with the keypoint moseq project structure. Since the keypoint-moseq results files contain
syllables for all sessions in the dataset, we iterate over these sessions using :py:func:`~neuroconv.datainterfaces.behavior.keypointmoseq.kpmsdatainterface.KeypointMoseqSubjectInterface.list_sessions`,
and run a conversion for each one.

.. code-block:: python

    for session_name in KeypointMoseqSubjectInterface.list_sessions(project_dir, model_folder):
        print("> Converting session:", session_name)

        source_data = dict(
            KeypointMoseq=dict(
                project_dir = project_dir,
                model_folder = model_folder,
                session_name = session_name,
                **KeypointMoseqSubjectInterface.metadata_from_config(**config())
            )
        )
        converter = ExampleKpmsNWBConverter(source_data = source_data)

        metadata = converter.get_metadata()
        metadata["NWBFile"]['session_description'] = "Open-field behavior session from keypoint-moseq tutorial dataset."
        metadata["NWBFile"]['identifier'] = session_name
        metadata["NWBFile"]['session_start_time'] = datetime.datetime.now(datetime.timezone.utc)
        metadata["BehavioralSyllable"]['kpms_version'] = kpms.__version__

        converter.run_conversion(metadata = metadata, nwbfile_path = nwb_path.format(session = session_name))
