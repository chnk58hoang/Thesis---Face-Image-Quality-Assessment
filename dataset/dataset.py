from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class ExFIQA(Dataset):
    def __init__(self, df):
        super().__init__()
        self.dataframe = df
        self.image_transform = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Resize(224),
                                                   transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        img_path = self.dataframe.iloc[index]['path']
        qscore = self.dataframe.iloc[index]['qscore']
        image = Image.open(img_path)
        image = self.image_transform(image)
        return image, qscore
