from imagenetv2_pytorch import ImageNetV2Dataset
from PIL import Image

from .imagenet import ImageNet
from src.templates.openai_imagenet_template import openai_imagenet_template

class ImageNetV2DatasetWithPaths(ImageNetV2Dataset):
    def __getitem__(self, i):
        img, label = Image.open(self.fnames[i]), int(self.fnames[i].parent.name)
        if self.transform is not None:
            img = self.transform(img)
        return {
            'images': img,
            'labels': label,
            'image_paths': str(self.fnames[i])
        }

class ImageNetV2(ImageNet):
    def get_test_dataset(self):
        return ImageNetV2DatasetWithPaths(transform=self.preprocess, location=self.location)
