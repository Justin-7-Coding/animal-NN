import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from PIL import Image


input_folder = 'C:/Users/kille/OneDrive/Desktop/raw_photos/'
output_folder = 'C:/Users/kille/OneDrive/Desktop/resized_photos/'

'''
size = (300, 300)
extensions = ('jpg', 'png', 'jpeg')

for subfolder in os.listdir(input_folder):
    subfolder_path = input_folder + subfolder
    for image in os.listdir(subfolder_path):
        print(image)

        #img_path = os.path.join(input_folder, file)
        img_path = subfolder_path + '/' + image
        raw_img = Image.open(img_path).convert('RGB')

        resized_img = raw_img.resize(size, Image.Resampling.LANCZOS)

        if not os.path.exists(output_folder + '/' + subfolder):
            os.makedirs(output_folder + '/' + subfolder)
        output_path = os.path.join(output_folder + '/' + subfolder, image)
        resized_img.save(output_path, quality=95)
'''


transform = transforms.ToTensor()

subfolder_path = output_folder

dataset = datasets.ImageFolder(root=subfolder_path, transform=transform)

loader = DataLoader(dataset, batch_size=2, shuffle = True)

images, labels = next(iter(loader))

print(images)
print(images.size())
print(labels)
