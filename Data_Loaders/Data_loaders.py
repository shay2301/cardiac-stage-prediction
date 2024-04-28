import torch
from torch.utils.data import Dataset
import unittest
import os
import pandas as pd
from IPython.display import display
import collections
import Utilis.utilis as util
class CardicFrame():
    def __init__(self,row):
        self.esv_frame= row['ESV_frame']
        self.edv_frame= row['EDV_frame']
        self.edv_poly= row['EDV_polygon'].str.split("\n")
        self.esv_poly= row['ESV_polygon'].str.split("\n")
    
    def __str__(self):
        return  (f'ESV Frame - {self.esv_frame}/nESV poly - {self.esv_poly}/n EDV Frame - {self.edv_frame} /n EDV poly - {self.edv_poly}/n')    


class EchonetDataset(Dataset):
    """EchoNet-Dynamic Dataset.

    Args:
        root (string): Root directory of dataset (defaults to `echonet.config.DATA_DIR`)
        split (string): One of {``train'', ``val'', ``test'', ``all'', or ``external_test''}
        target_type (string or list, optional): Type of target to use,
            ``Filename'', ``EF'', ``EDV'', ``ESV'', ``LargeIndex'',
            ``SmallIndex'', ``LargeFrame'', ``SmallFrame'', ``LargeTrace'',
            or ``SmallTrace''
            Can also be a list to output a tuple with all specified target types.
            The targets represent:
                ``Filename'' (string): filename of video
                ``EF'' (float): ejection fraction
                ``EDV'' (float): end-diastolic volume
                ``ESV'' (float): end-systolic volume
                ``LargeIndex'' (int): index of large (diastolic) frame in video
                ``SmallIndex'' (int): index of small (systolic) frame in video
                ``LargeFrame'' (np.array shape=(3, height, width)): normalized large (diastolic) frame
                ``SmallFrame'' (np.array shape=(3, height, width)): normalized small (systolic) frame
                ``LargeTrace'' (np.array shape=(height, width)): left ventricle large (diastolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
                ``SmallTrace'' (np.array shape=(height, width)): left ventricle small (systolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
            Defaults to ``EF''.
        mean (int, float, or np.array shape=(3,), optional): means for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not shifted).
        std (int, float, or np.array shape=(3,), optional): standard deviation for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not scaled).
        length (int or None, optional): Number of frames to clip from video. If ``None'', longest possible clip is returned.
            Defaults to 16.
        period (int, optional): Sampling period for taking a clip from the video (i.e. every ``period''-th frame is taken)
            Defaults to 2.
        max_length (int or None, optional): Maximum number of frames to clip from video (main use is for shortening excessively
            long videos when ``length'' is set to None). If ``None'', shortening is not applied to any video.
            Defaults to 250.
        clips (int, optional): Number of clips to sample. Main use is for test-time augmentation with random clips.
            Defaults to 1.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        external_test_location (string): Path to videos to use for external testing.
    """
    def __init__(self, root=None,
                 split="train", target_type="EF",
                 mean=0., std=1.,
                 length=16, period=2,
                 max_length=250,
                 clips=1,
                 target_transform=None,
                 external_test_location=None,
                 is_debug_run = True
                 ):

        self.root  = root
        self.split = split.upper()
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = target_type
        self.mean = mean
        self.std = std
        self.length = length
        self.max_length = max_length
        self.clips = clips
        self.target_transform = target_transform
        self.external_test_location = external_test_location

        self.fnames, self.outcome = [], []

        if self.split == "EXTERNAL_TEST":
            self.fnames = sorted(os.listdir(self.external_test_location))
        else:
            # Load video-level label

           with open(os.path.join(self.root, "training_database.csv")) as f:
                data = pd.read_csv(f)
                display(data)
        if self.split != "ALL":
            data = data[data["Split"] == self.split]

        self.header = data.columns.tolist()
        self.fnames = data["FileName"].tolist()
        #self.fnames = [ fn + ".avi" for fn in self.fnames if os.path.splitext(fn)[1] == ""]  # Assume avi if no suffix
        #     self.outcome = data.values.tolist()

        #     # Check that files are present
        if (is_debug_run):
            missing = set(self.fnames) - set(os.listdir(os.path.join(self.root,"TestData" ,"Videos")))
            self.fnames = set(os.listdir(os.path.join(self.root,"TestData" ,"Videos")))
            print(f'Testing only {self.fnames}')
        else:                       
            missing = set(self.fnames) - set(os.listdir(os.path.join(self.root,"Videos")))
            if len(missing) != 0:
                print("{} videos could not be found in {}:".format(len(missing), os.path.join(self.root, "Videos")))
                for f in sorted(missing):
                    print("\t", f)
                raise FileNotFoundError(os.path.join(self.root, "Videos", sorted(missing)[0]))

        
        #### Load frames data
        self.frames = collections.defaultdict(list)

        for file in self.fnames:
            row  = data[data['FileName'] == file]
            self.frames['file'] = CardicFrame(row)
            print(self.frames['file'])
    def load_data(self):
        # Implement data loading logic
        # For example, load images or CSV data
        return []

    # def __len__(self):
    #     return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
    
        video_path = os.path.join(self.root, self.fnames[idx])
        video = utilis.load_video(video_path)        
        
        if self.transform:
            sample = self.transform(video)
        print(video.shape())
        return video

class TestCustomDataset(unittest.TestCase):
    def test_data_loading(self):
        # Specify the path to your test data
        test_path = 'DataBase'
        dataset = EchonetDataset(root = test_path, is_debug_run = 1)
        # Check if data is not empty
        self.assertTrue(len(dataset) > 0, "Data has not been loaded correctly")

if __name__ == "__main__":
    unittest.main()