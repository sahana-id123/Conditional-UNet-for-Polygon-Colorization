import os
import json
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

COLOR2IDX = {
    'red': 0, 'blue': 1, 'green': 2, 'yellow': 3,
    'orange': 4, 'purple': 5, 'black': 6, 'white': 7,
    'pink': 8, 'brown': 9
}

class PolygonDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.input_dir = os.path.join(root_dir, 'inputs')
        self.output_dir = os.path.join(root_dir, 'outputs')
        self.mapping = json.load(open(os.path.join(root_dir, 'data.json')))
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        item = self.mapping[idx]
        input_img = Image.open(os.path.join(self.input_dir, item['input'])).convert('RGB')
        output_img = Image.open(os.path.join(self.output_dir, item['output'])).convert('RGB')
        color = COLOR2IDX[item['color'].lower()]

        return self.transform(input_img), self.transform(output_img), color
