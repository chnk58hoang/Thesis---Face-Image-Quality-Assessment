from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class ExFIQA(Dataset):
    def __init__(self, df):
        super().__init__()
        self.dataframe = df
        self.image_transform = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Resize(112)])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        img_path = self.dataframe.iloc[index]['path']
        pose = self.dataframe.iloc[index]['pose']
        #br = self.dataframe.iloc[index]['br']
        #img_path = img_path.replace('/kaggle/input/multicmu/multi_PIE_crop_128',
        #                            '/home/artorias/Downloads/multi_PIE_crop_128')
        image = Image.open(img_path)
        image = self.image_transform(image)
        return image, int(pose)
