
from ....basedatainterface import BaseDataInterface
from ....utils import FolderPathType, ArrayType
from pynwb import NWBFile
import h5py

from .kpms_utils import write_subject_to_nwb, get_video_timestamps



class KeypointMoseqSubjectInterface(BaseDataInterface):

    def __init__(self,
        project_dir: FolderPathType,
        model_folder: FolderPathType,
        session_name: FolderPathType,
        video_dir: FolderPathType,
        use_bodyparts: ArrayType,
        skeleton: ArrayType,
        **kwargs
    ):
        super().__init__(
            session_name = session_name,
            project_dir = project_dir,
            model_folder = model_folder,
            video_dir = video_dir,
            use_bodyparts = use_bodyparts,
            skeleton = skeleton
        )


    def get_metadata_schema(self):
        metadata_schema = super().get_metadata_schema()
        metadata_schema['properties']["BehavioralSyllable"] = dict(
            type = "object",
            properties = dict(
                kpms_version =  dict(type = "string")
            )
        )
        return metadata_schema



    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: dict):

        # ----- set up arguments to nwb-write helper function
        
        results_h5_pth = (f"{self.source_data['project_dir']}/"
                          f"{self.source_data['model_folder']}/results.h5")
        with h5py.File(results_h5_pth, 'r') as h5:
            session_data = {
                k: h5[self.source_data['session_name']][k][:]
                for k in h5[self.source_data['session_name']].keys()}
            
        timestamps = get_video_timestamps(
            self.source_data['session_name'],
            self.source_data['video_dir'])
        
        skeleton = [
            [self.source_data['use_bodyparts'].index(bp) for bp in bone]   
            for bone in self.source_data['skeleton']
            if all(bp in self.source_data['use_bodyparts'] for bp in bone)
        ]


        # ----- call nwb-write helper function

        write_subject_to_nwb(
            nwbfile,
            pose_arr = session_data['estimated_coordinates'],
            centroid_arr = session_data['centroid'],
            heading_arr = session_data['heading'],
            latents_arr = session_data['latent_state'],
            syllables_arr = session_data['syllables'],
            timestamps = timestamps,
            kpms_model = self.source_data['model_folder'],
            kpms_version = metadata['BehavioralSyllable']["kpms_version"],
            bodyparts = self.source_data['use_bodyparts'],
            skeleton = skeleton
        )


    # ===== Utility functions for integrating with kpms project structure
    
    def metadata_from_config(**config):
        """
        Extract keys from kpms config that are needed as source_data
        for `KeypointMoseqSubjectInterface`.

        Parameters
        ----------
        config : dict as kwargs
            Dictionary loaded from keypoint moseq `config.yml`
        
        Returns
        -------
        metadata : dict
            Dictionary containing needed as metadata for source_data
            to pass to `KeypointMoseqSubjectInterface` or an
            `NWBConverter` that uses a keypoint moseq data interface.
        """
        return dict(
            video_dir = config['video_dir'],
            use_bodyparts = config['use_bodyparts'],
            skeleton = config['skeleton'],
        )
    

    def list_sessions(project_dir, model_folder):
        """
        Retreive list of sessions processed by a keypoint moseq model
        
        Parameters
        ----------
        project_dir : str
            Keypoint-moseq project directory.
            
        model_folder : str
            Model name; i.e. name of the folder in the keypoint-moseq
            project directory containing the model checkpoint/results
            file."""
        results_h5_pth = (f"{project_dir}/{model_folder}/results.h5")
        with h5py.File(results_h5_pth, 'r') as h5:
            return list(h5.keys())