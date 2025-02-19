import os

import pandas as pd

import lotus
from lotus.dtype_extensions import ImageArray
from lotus.models import LM

lotus.settings.configure(lm=LM(model="gpt-4o-mini"))

# The images folder contain images representing digits taken from MNIST dataset
image_file_names = os.listdir("images")  # get all file in the folder

# file names are the same as the digit represented by image
labels = [os.path.splitext(image)[0] for image in image_file_names]
image_paths = [os.path.join("images", image) for image in image_file_names]

df = pd.DataFrame({"image": ImageArray(image_paths), "label": labels, "image_path": image_paths})

df = df.sem_filter("{image} represents number 1")
print(df)
