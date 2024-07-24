import pandas as pd
import numpy as np
import os
import yaml

import torch
import torch.optim as optim
import torch.nn.functional as tFun
from torchinfo import summary
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor
import lightning.pytorch as lPyT

from uNet import *

from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torchmetrics.functional.classification import multilabel_precision, multilabel_recall

import random
import matplotlib.pyplot as plt

# Load configuration file
# 'C:\\Users\\NicoT\\Desktop\\config.yaml'
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Directory containing files (images and masks)
directory = config['directories']['main']
#directory = "C:\\Users\\NicoT\\Downloads\\EBHI-SEG\\EBHI-SEG"
#directory = "/hpc/home/nicolo.thei/Deep/EBHI-SEG"

# Create a DataFrame containing file paths for images and masks
dFrame = pd.DataFrame(columns=['image', 'label'])
cancerTypes = config['cancerTypes']
colors = config['colors']

# Loop through each cancer type directory and collect image and mask file paths
for cancer_type in cancerTypes:
    image_dir = os.path.join(directory, cancer_type, 'image')
    mask_dir = os.path.join(directory, cancer_type, 'label')

    # Check if the image and mask directories exist
    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        print(f"Directory NOT Found for {cancer_type}")
        continue

    image_files = sorted(os.listdir(image_dir))
    
    for file in image_files:
        image_file = os.path.join(image_dir, file)
        mask_file = os.path.join(mask_dir, file)  # Assuming mask file name is derived from image file name MODIFICATO

        # Check if mask file exists, if not both image and mask files are not inserted into the DataFrame
        if os.path.isfile(mask_file):
            dFrame = pd.concat([dFrame, pd.DataFrame({'image': [image_file], 'label': [mask_file]})], ignore_index=True)

# Shuffle the DataFrame to mix the data
dFrame = dFrame.sample(frac=1).reset_index(drop=True)

# Print the length of the DataFrame
print("Total Number of Samples:", len(dFrame))

# Function to label the images with their corresponding cancer type
def label_image(img_path):
    for cancer_type in cancerTypes:
        if cancer_type in img_path:
            return cancer_type
        
dFrame['class'] = dFrame['image'].copy().map(label_image)

# Split the dataset into training and testing sets
dFrame['split'] = 'train'  #Training set by default
train_df, test_df = train_test_split(dFrame, test_size=config['data']['testSize'], random_state=config['data']['randomState'])  #random_state for reproducibility (can be omitted)
dFrame.loc[test_df.index, 'split'] = 'testing'

#dFrame.sample(10)

#Custom Dataset Loader (I) -> loads, transforms and returns the image and mask
class customDataset(Dataset): 
    def __init__(self, imageList, labelList, classList, classDict, transforms=None):
        self.imageList = imageList
        self.labelList = labelList
        self.classList = classList
        self.classDict = classDict
        self.transforms = transforms
    def __len__(self):
        return len(self.imageList)
    def __getitem__(self, index):
        #Loads an image and its corresponding mask from the file paths at the specified index
        image = np.array(Image.open(self.imageList[index]))
        mask = np.array(Image.open(self.labelList[index]))
        #Even if it still is a binary mask it converts it to a binary mask and then multiplies it by the class index corresponding to the class of the sample
        mask = (mask > 0).astype(np.uint8) * self.classDict[self.classList[index]]
        #print(mask)
        if self.transforms: #If transformations are specified (see dataModule class), applies them to the image and mask
            transformed = self.transforms(image=image, mask=mask)
            imgAug = transformed['image'].contiguous()
            maskAug = transformed['mask'].contiguous()
            return imgAug, maskAug
        #Returns the image and mask as tensors if no transformations are specified
        image = torch.as_tensor(image).float().contiguous()
        mask = torch.as_tensor(mask).long().contiguous()
        return image, mask
    
#Custom Dataset Loader (II) -> manages data preparation and creates training and testing dataloaders
class dataModule(lPyT.LightningDataModule):
    def __init__(self, dFrame, classDict, batchSize):
        super().__init__()
        self.dFrame = dFrame #It contains image paths, labels, classes and split info (train or test)
        self.batchSize = batchSize
        self.classDict = classDict
    def _getTransform(self, transformBool=config['transformBool']):
        if transformBool: #Boolean indicating if transform applies for training or validation/prediction or not.
            return A.Compose([A.OneOf([A.RandomRotate90(), A.VerticalFlip(), A.HorizontalFlip()], p=config['augmentationProb']), ToTensorV2()]) 
        return A.Compose([ToTensorV2()])
    def setup(self, stage : str):
        trainTransform = self._getTransform()
        valTransform = self._getTransform(False)
        #Training dataset with training data and transformations (training trasformation can be set in config file)
        self.trainingSet = customDataset(dFrame[dFrame['split'] == 'train']['image'].tolist(), dFrame[dFrame['split'] == 'train']['label'].tolist(), dFrame[dFrame['split'] == 'train']['class'].tolist(), classDict=self.classDict, transforms=trainTransform)
        #Testing dataset with testing data and validation transformations (always False).
        self.testSet = customDataset(dFrame[dFrame['split'] == 'testing']['image'].tolist(), dFrame[dFrame['split'] == 'testing']['label'].tolist(), dFrame[dFrame['split'] == 'testing']['class'].tolist(), classDict=self.classDict, transforms=valTransform)
    def train_dataloader(self):
        return DataLoader(self.trainingSet, batch_size=self.batchSize, num_workers=config['workers'], drop_last=True, pin_memory=True, shuffle=True, persistent_workers=True)
    def predict_dataloader(self):
        return DataLoader(self.testSet, batch_size=self.batchSize, num_workers=config['workers'], drop_last=True, pin_memory=True, shuffle=False)

# Instantiate the UNet model
#model = UNet(n_channels=config['model']['nChannels'], n_classes=config['model']['nClasses'], residual=config['model']['residual'], attention=config['model']['attention'])
#summary(model, input_size=(1, 3, 224, 224))

# Define the PyTorch module for the UNet model
class UNetModule(lPyT.LightningModule):
    def __init__(self, nClasses=7, learnRate=config['model']['learnRate'], wDecay=config['model']['decay']):
        super().__init__()
        
        self.model = UNet(n_channels=config['model']['nChannels'], n_classes=nClasses, residual=config['model']['residual'], attention=config['model']['attention']) #Initialize the UNet model with the specified number of input channels and output classes

        self.nClasses = nClasses
        self.learnRate = learnRate
        self.wDecay = wDecay
        
        self.save_hyperparameters() #Save the number of classes, learning rate, weight decay and CrossEntropyLoss
    def forward(self, x):
        return self.model(x) #Define how the input is passed through the model
    def training_step(self, batch):
        img, mask = batch
        img = img.to(device=self.device, dtype=torch.float32, memory_format=torch.channels_last)
        mask = mask.to(device=self.device, dtype=torch.long)
        predMask = self.forward(img)
        
        #Use the combined loss function
        totalLoss, ceLoss, diceLossVal, jaccardLoss = combinedLoss(predMask, mask, self.nClasses)
        
        #Log the losses
        self.log('totalLoss', totalLoss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('crossEntroyLoss', ceLoss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('diceLoss', diceLossVal, on_step=False, on_epoch=True, prog_bar=True)
        self.log('jaccardLoss', jaccardLoss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('jaccard Coefficient', 1-jaccardLoss, on_step=False, on_epoch=True, prog_bar=True)

        return totalLoss

    def predict_step(self, batch):
        #Data preparation, it moves the images and masks to the device (GPU) and sets the correct data type
        img, mask = batch
        img = img.to(device = self.device, dtype=torch.float32)
        outputs = self.forward(img)
        predMask = outputs.argmax(dim=1)

        mask = tFun.one_hot(mask.long(), self.nClasses).permute(0, 3, 1, 2).float()
        predictionMask = tFun.one_hot(predMask.cpu(), self.nClasses).permute(0, 3, 1, 2).float()

        jaccard = jaccardIndex(predMask, mask, train=True) #Computes the Jaccard Index to evaluate prediction accuracy

        print(f" Jaccard Index: {jaccard:.2f}")

        return img, predMask.long().squeeze().cpu().numpy(), mask, predictionMask
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learnRate, weight_decay=self.wDecay) #Uses the Adam optimizer
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config['optimizer']['patience'], mode='min') #It reduces the learning rate when the Jaccard loss stops improving.
        return {
            'optimizer' : optimizer, 'lr_scheduler' : { 'scheduler' : scheduler, 'monitor' : 'jaccardLoss' }
        }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataMod = dataModule(dFrame=dFrame, classDict=config['idCancerTypes'], batchSize=config['training']['batchSize'])
model = UNetModule(learnRate=config['model']['learnRate']).to(device)
lrMonitor = LearningRateMonitor(logging_interval=config['logging']['interval'])
logger = TensorBoardLogger(config['directories']['tensorboardLogs'], name='UNetSegm')
trainer = lPyT.Trainer(accelerator='gpu', devices=[0], max_epochs=config['training']['maxEpochs'], logger=logger, callbacks=[lrMonitor], enable_checkpointing=True)

# Train the model
trainer.fit(model, datamodule=dataMod)

# Trainer can resume after being stopped by loading the last checkpoint
# trainer.fit(model, datamodule=dataMod, ckpt_path="/hpc/home/nicolo.thei/Deep/tb_logs/UNetSegm/version_8/checkpoints/epoch=19-step=54.ckpt")

# ONLY Prediction
# Uncomment if you want to start prediction on a successfully trained checkpoint
# in this case put in ckpt_path the directory to the trained checkpoint
'''
dataMod1 = dataModule(dFrame=dFrame, classDict=config['idCancerTypes'], batchSize=config['training']['batchSize'])
model1 = UNetModule(learnRate=config['model']['learnRate']).to(device)
lrMonitor1 = LearningRateMonitor(logging_interval=config['logging']['interval'])
logger1 = TensorBoardLogger(config['directories']['tensorboardLogs'], name='UNetSegm')
trainer1 = lPyT.Trainer(accelerator='gpu', devices=[0], max_epochs=config['training']['maxEpochs'], callbacks=[lrMonitor1])
'''
# ckpt_path="/hpc/home/nicolo.thei/Deep/tb_logs/UNetSegm/version_9/checkpoints/epoch=99-step=11100.ckpt"
results = trainer.predict(model, datamodule=dataMod, ckpt_path="best") #Predicts the masks for the testing set

numClasses = [0,1,2,3,4,5,6]

def drawPicture(preds, mask_values, colors=colors):
    listOverlay = []
    listGroundTruthMasks = []
    
    #Extracts the original mask, the predicted mask, the image and the mask from preds wich is a list of tuples containing images batch, masks batch, original masks batch and predicted masks batch
    originalMask = [batch[2] for batch in preds]
    originalMask = torch.stack([orMask for batch in originalMask for orMask in batch])
    predictionMask = [batch[3] for batch in preds]
    predictionMask = torch.stack([preMask for batch in predictionMask for preMask in batch])
    images = [batch[0] for batch in preds]
    images = torch.stack([img for batch in images for img in batch])
    masks = [batch[1] for batch in preds]
    masks = np.array([mask for batch in masks for mask in batch])

    #Computes the multilabel precision for each class
    precisions = multilabel_precision(predictionMask, originalMask, num_labels=len(mask_values), average='none')
    print(f'Precisions = ' + ', '.join([f'{label}: {p:.2f}' for label, p in zip(["Background", "Normal", "Polyp", "LOW-Grade IN", "Adenocarcinoma", "HIGH-Grade IN", "Serrated Adenoma"], precisions)]))

    for img, mask, original in zip(images, masks, originalMask):
        #Initialize a matrix of zeros for each image-mask couple
        if mask_values == [0, 1]:
            out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
        else:
            out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)
        
        #Converts each mask from multi-dimensiona to a single class index
        if mask.ndim == 3:
            mask = np.argmax(mask, axis=0)

        #Apply the mask_values to the mask
        for i, v in enumerate(mask_values):
            out[mask == i] = v

        #Converts the previous zero matrix to a colored mask following the color map (each color for each tumor type)
        classColorMap = np.array(colors, dtype=np.uint8)
        realImg = Image.fromarray(img.cpu().permute(1,2,0).numpy().astype(np.uint8))
        annIm = Image.fromarray(classColorMap[out])
        annImage = annIm.convert('RGB')
        listOverlay.append(Image.blend(realImg, annImage, alpha=.7))
        
        classIndices = torch.argmax(original, dim=0)
        greyScale = torch.linspace(0, 255, steps=7).long()
        greyImg = greyScale[classIndices]
        greyImgNp = greyImg.cpu().numpy().astype(np.uint8)
        finalMask = Image.fromarray(greyImgNp, 'L')
        listGroundTruthMasks.append(finalMask)
        
    return listOverlay, listGroundTruthMasks

overlay, groundTruthMasks  = drawPicture(results, numClasses, colors=colors)

# Display the overlay images
fig, axes = plt.subplots(5, 5, figsize=(80,80))
indices = np.arange(0, len(overlay), step=1)
choices = []
counter = 0
for i, ax in enumerate(axes.flatten()):
    choices.append(random.choices(indices)[0])
    ax.imshow(overlay[choices[counter]])
    counter += 1
    ax.axis('off')

plt.savefig('image'+str(config['model']['residual'])+str(config['model']['attention'])+'.png')

counter = 0
fig, axes = plt.subplots(5, 5, figsize=(80,80))
indices = np.arange(0, len(groundTruthMasks), step=1)
for i, ax in enumerate(axes.flatten()):
    ax.imshow(groundTruthMasks[choices[counter]])
    counter += 1
    ax.axis('off')

plt.savefig('masks'+str(config['model']['residual'])+str(config['model']['attention'])+'.png')