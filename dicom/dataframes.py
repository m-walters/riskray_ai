from enum import Enum

from .. dicom.dataset_tools import generate_LR_file_items, process_dicom_items, load_pickle
from .. settings import STORE_PATH


class DicomFilePath(Enum):
    # TODO -- Query Bucket to grab all items with wildcard; remove max_index
    CROPPED_CONTROL = {
        "left_prefix": "dicom/croppedDICOMs_2/control/L/control_L",
        "right_prefix": "dicom/croppedDICOMs_2/control/R/control_R",
        "rejects": []
    }
    CROPPED_ATRISK = {
        "left_prefix": "dicom/croppedDICOMs_2/at_risk/L/at_risk_L",
        "right_prefix": "dicom/croppedDICOMs_2/at_risk/R/at_risk_R",
        "rejects": []
    }
    CROPPED_CONTROL_OLD = {
        "left_prefix": "dicom/croppedDICOMs/control/control_set_L",  # Imagine a wild-card asterisk after this
        "right_prefix": "dicom/croppedDICOMs/control/control_set_R",
        "rejects": [],
    }
    CROPPED_FRACTURE_OLD = {
        "left_prefix": "dicom/croppedDICOMs/fracture/fracture_set_L",
        "right_prefix": "dicom/croppedDICOMs/fracture/fracture_set_R",
        "rejects": [
            "dicom/croppedDICOMs/fracture/fracture_set_L63"
        ],
    }
    RAW_FRACTURE = {
        "prefix": "dicom/fracture 1/DICOM",
        "suffix": "",
        "rejects": []
    }
    # The next two have messy and inconsistent subdirectory structures
    RAW_CONTROL_1 = {
        "prefix": "dicom/Control 1/DICOM/PAT_0000",
        "rejects": []
    }
    RAW_CONTROL_2 = {
        "prefix": "dicom/Control 2/DICOM/PAT_0000",
        "rejects": []
    }

    @staticmethod
    def list_options():
        return [e.name for e in list(DicomFilePath)]


class TrainingDataFrame:
    """
    Use this to work with different runs etc. and manage their dataframes
    """
    # Identify this dataframe
    name = None

    # Hold onto the dataframe object
    dataframe = None

    def __init__(self, name: str, control: str, at_risk: str, **kwargs):
        """
        :param name:        Identify the dataframe
        :param control:     DicomFilePath Name for the control set
        :param at_risk:     DicomFilePath Name for the at-risk set
        """
        control_name = control
        at_risk_name = at_risk
        assert control_name in DicomFilePath.list_options(), \
            f"Control DicomFilePath {control_name} is not valid. Options are {DicomFilePath.list_options()}"
        assert at_risk_name in DicomFilePath.list_options(), \
            f"At-risk DicomFilePath {at_risk_name} is not valid. Options are {DicomFilePath.list_options()}"

        # Set these
        self.control_dicom: DicomFilePath = DicomFilePath[control_name]
        self.at_risk_dicom: DicomFilePath = DicomFilePath[at_risk_name]

        self.control_label = 0
        self.at_risk_label = 1

        self.name = name  # This should be unique for different datasets
        self.pickle_path = STORE_PATH + f"/dataframes/{self.name}.pkl"

    def create_base_dataframe(self, save=False):
        """
        Create the plain dataframe from the fetched dicoms.
        These pixel arrays will not be cropped or padded or anything

        :param save:            Save as .pkl boolean
        """
        # Get dicom items
        dicom_items = generate_LR_file_items(self.control_dicom, self.control_label)
        dicom_items.extend(
            generate_LR_file_items(self.at_risk_dicom, self.at_risk_label)
        )

        # Process them
        df = process_dicom_items(dicom_items, save=save, save_path=self.pickle_path)
        return df

    def get_pickle(self):
        """
        Get the pickle
        """
        print(f"Loading {self.pickle_path}")
        try:
            self.dataframe = load_pickle(self.pickle_path)
        except:
            print(f"Could not get pickle at {self.pickle_path}")
            return None
            
        return self.dataframe

    def save_pickle(self, df=None):
        """
        Save the pickle dataframe
        """
        if df:
            self.dataframe = df
        self.dataframe.to_pickle(self.pickle_path)
        print(f"Saved {self.pickle_path}")


class DefaultDataFrame(TrainingDataFrame):
    """
    The default set. Based off the 'control' and 'fracture' croppedDICOMs subdirectories
    """
    name = "default"

    def __init__(self, **kwargs):
        super(DefaultDataFrame, self).__init__(
            control=DicomFilePath.CROPPED_CONTROL.name,
            at_risk=DicomFilePath.CROPPED_ATRISK.name,
            name=self.name,
            **kwargs
        )
