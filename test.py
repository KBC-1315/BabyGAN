#버전 확인 
# %를 사용하는 매직커멘드를 통해 버전 변경
# %tensorflow_version 1.x 

#keras load_model에서 오류 발생할 때
# !pip install h5py==2.10.0 #:: 다운그레이드 실시로 문제 해결
import h5py
print(h5py.__version__)
#@title ← Clone Git repository and install all requirements

import os
import cv2
import math
import pickle
import imageio
import warnings
import PIL.Image
import numpy as np
from PIL import Image
import tensorflow as tf
from random import randrange
import moviepy.editor as mpy
# from google.colab import drive 코랩 관련 명령어는 무시
# from google.colab import files
import matplotlib.pyplot as plt
from IPython.display import clear_output
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
warnings.filterwarnings("ignore")

def get_watermarked(pil_image: Image) -> Image:
  try:
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    (h, w) = image.shape[:2]
    image = np.dstack([image, np.ones((h, w), dtype="uint8") * 255])
    pct = 0.08
    full_watermark = cv2.imread('./content/BabyGAN/media/logo.png', cv2.IMREAD_UNCHANGED)
    (fwH, fwW) = full_watermark.shape[:2]
    wH = int(pct * h*2)
    wW = int((wH * fwW) / fwH*0.1)
    watermark = cv2.resize(full_watermark, (wH, wW), interpolation=cv2.INTER_AREA)
    overlay = np.zeros((h, w, 4), dtype="uint8")
    (wH, wW) = watermark.shape[:2]
    overlay[h - wH - 10 : h - 10, 10 : 10 + wW] = watermark
    output = image.copy()
    cv2.addWeighted(overlay, 0.5, output, 1.0, 0, output)
    rgb_image = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)
  except: return pil_image

def generate_final_images(latent_vector, direction, coeffs, i):
    new_latent_vector = latent_vector.copy()
    new_latent_vector[:8] = (latent_vector + coeffs*direction)[:8]
    new_latent_vector = new_latent_vector.reshape((1, 18, 512))
    generator.set_dlatents(new_latent_vector)
    img_array = generator.generate_images()[0]
    img = PIL.Image.fromarray(img_array, 'RGB')
    if size[0] >= 512: img = get_watermarked(img)
    img_path = "./content/BabyGAN/for_animation/" + str(i) + ".png"
    img.thumbnail(animation_size, PIL.Image.ANTIALIAS)
    img.save(img_path)
    face_img.append(imageio.imread(img_path))
    clear_output()
    return img

def generate_final_image(latent_vector, direction, coeffs):
    new_latent_vector = latent_vector.copy()
    new_latent_vector[:8] = (latent_vector + coeffs*direction)[:8]
    new_latent_vector = new_latent_vector.reshape((1, 18, 512))
    generator.set_dlatents(new_latent_vector)
    img_array = generator.generate_images()[0]
    img = PIL.Image.fromarray(img_array, 'RGB')
    if size[0] >= 512: img = get_watermarked(img)
    img.thumbnail(size, PIL.Image.ANTIALIAS)
    img.save("face.png")
    if download_image == True: files.download("face.png")
    return img

def plot_three_images(imgB, fs = 10):
  f, axarr = plt.subplots(1,3, figsize=(fs,fs))
  axarr[0].imshow(Image.open('./content/BabyGAN/aligned_images/father_01.png'))
  axarr[0].title.set_text("Father's photo")
  axarr[1].imshow(imgB)
  axarr[1].title.set_text("Child's photo")
  axarr[2].imshow(Image.open('./content/BabyGAN/aligned_images/mother_01.png'))
  axarr[2].title.set_text("Mother's photo")
  plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
  plt.show()

# !rm -rf sample_data
# !git clone https://github.com/tg-bomze/BabyGAN.git


import config
import dnnlib
import dnnlib.tflib as tflib
from encoder.generator_model import Generator


age_direction = np.load('./content/BabyGAN/ffhq_dataset/latent_directions/age.npy')
horizontal_direction = np.load('./content/BabyGAN/ffhq_dataset/latent_directions/angle_horizontal.npy')
vertical_direction = np.load('./content/BabyGAN/ffhq_dataset/latent_directions/angle_vertical.npy')
eyes_open_direction = np.load('./content/BabyGAN/ffhq_dataset/latent_directions/eyes_open.npy')
gender_direction = np.load('./content/BabyGAN/ffhq_dataset/latent_directions/gender.npy')
smile_direction = np.load('./content/BabyGAN/ffhq_dataset/latent_directions/smile.npy')

clear_output()
#@title ← Upload portrait of a Person1
import os
'''
!rm -rf ./content/BabyGAN/father_image/*.*
#@markdown *이미지 **url** 이나 로컬환경에 있는 사진을 업로드 하세요.*
url = './content/BabyGAN/sample.png' #@param {type:"string"}

try:
  fat = url.split('/')[-1]
except BaseException:
  print("Something wrong. Try uploading a photo from your computer")

FATHER_FILENAME = "father." + fat.split(".")[-1]
os.rename(fat, FATHER_FILENAME)
father_path = "./content/BabyGAN/father_image/" + FATHER_FILENAME
!mv -f father_filename father_path
'''

!python3 ./content/BabyGAN/align_images.py ./content/BabyGAN/father_image ./content/BabyGAN/aligned_images
# clear_output()
if os.path.isfile('./content/BabyGAN/aligned_images/father_01_01.png'):
  try :
    pil_father = Image.open('./content/BabyGAN/aligned_images/father_01_01.png')
    (fat_width, fat_height) = pil_father.size
    resize_fat = max(fat_width, fat_height)/256
    display(pil_father.resize((int(fat_width/resize_fat), int(fat_height/resize_fat))))
  except Exception as e:
    print(e)
#@title ← Upload portrait of a Person2

'''
!rm -rf /content/BabyGAN/mother_image/*.*
#@markdown *이미지 **url** 이나 로컬환경에 있는 사진을 업로드 하세요.*
url = '' #@param {type:"string"}
if url == '':
  uploaded = list(files.upload().keys())
  if len(uploaded) > 1: raise ValueError('You cannot upload more than one image at a time!')
  mot = uploaded[0]
else:
  try:
    !wget $url
    mot = url.split('/')[-1]
  except BaseException:
    print("Something wrong. Try uploading a photo from your computer")

MOTHER_FILENAME = "mother." + mot.split(".")[-1]
os.rename(mot, MOTHER_FILENAME)
mother_path = "/content/BabyGAN/mother_image/" + MOTHER_FILENAME
!mv -f mother_filename mother_path
'''

!python3 ./content/BabyGAN/align_images.py ./content/BabyGAN/mother_image ./content/BabyGAN/aligned_images
# clear_output()
if os.path.isfile('./content/BabyGAN/aligned_images/mother_01_01.png'):
  pil_mother = Image.open('./content/BabyGAN/aligned_images/mother_01_01.png')
  (mot_width, mot_height) = pil_mother.size
  resize_mot = max(mot_width, mot_height)/256
  display(pil_mother.resize((int(mot_width/resize_mot), int(mot_height/resize_mot))))
else: raise ValueError('No face was found or there is more than one in the photo.')

#@title ← Get latent representation
#@markdown *몇 분의 시간이 소요되니 기다려주세요.*
'''
#use_pretraineg_model = True #@param {type:"boolean"}
if use_pretraineg_model == False:
  !rm finetuned_resnet.h5
  !python3 train_resnet.py \
  --test_size 256 \
  --batch_size 1024 \
  --loop 1 \
  --max_patience 1'''

#!python encode_images.py --help
!python3 ./content/BabyGAN/encode_images.py \
  --early_stopping False \
  --lr=0.25 \
  --batch_size=2 \
  --iterations=100 \
  --output_video=False \
  ./content/BabyGAN/aligned_images \
  ./content/BabyGAN/generated_images \
  ./content/BabyGAN/latent_representations

tflib.init_tf()
URL_FFHQ = "./content/BabyGAN/karras2019stylegan-ffhq-1024x1024.pkl"
with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
    generator_network, discriminator_network, Gs_network = pickle.load(f)
generator = Generator(Gs_network, batch_size=1, randomize_noise=False)
model_scale = int(2*(math.log(1024,2)-1))

clear_output()
if len(os.listdir('./content/BabyGAN/generated_images')) == 2:
  first_face = np.load('./content/BabyGAN/latent_representations/father_01.npy')
  second_face = np.load('./content/BabyGAN/latent_representations/mother_01.npy')
  print("Generation of latent representation is complete! Now comes the fun part.")
else: raise ValueError('Something wrong. It may be impossible to read the face in the photos. Upload other photos and try again.')