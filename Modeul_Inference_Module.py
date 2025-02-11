#بسم الله الرحمن الرحيم

from train import load_chk_point
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn, optim
from torchvision import datasets, transforms, models
import torch
import json


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

mean_vals = [0.485, 0.456, 0.406]
std_vals = [0.229, 0.224, 0.225]
trn_img_trnsfrms = transforms.Compose([ transforms.Resize((224,224)),
                                        transforms.RandomRotation(30),
                                        transforms.RandomCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(), 
                                        transforms.Normalize(mean= mean_vals, std=std_vals)])

tst_img_trnsfrms = transforms.Compose([transforms.Resize((255,255)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=mean_vals, std=std_vals)])

vld_img_trnsfrms = transforms.Compose([transforms.Resize((255,255)), 
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(), 
                                       transforms.Normalize(mean=mean_vals, std=std_vals)])

trn_img_dataset = datasets.ImageFolder(train_dir, transform=trn_img_trnsfrms)

tst_img_dataset = datasets.ImageFolder(test_dir, transform=tst_img_trnsfrms)

vld_img_dataset = datasets.ImageFolder(valid_dir, transform=vld_img_trnsfrms)


model = load_chk_point('chk_point.pth')[0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img_jpg = Image.open(image)
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]
    img_processes = transforms.Compose([transforms.Resize((255, 255)), transforms.ToTensor(),
                                        transforms.Normalize(mean=mean_vals, std=std_vals)])
    img_tnsr = img_processes(img_jpg)
    #img_tnsr = img_prcs_mdl(img_tnsr)
    # Apply the image into the model

    # convert the tensor form image into a numpy array
    return img_tnsr
    # TODO: Process a PIL image for use in a PyTorch mo
 
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
    
    
'''
    Functions that used for mapping labels with indices.
'''
def get_key(dic, value):
    for key, val in dic.items():
        if val == value:
            return key

def map_dict(top_clss, categ_json_fl, img_dataset):
    top_clss.squeeze_(0)
    #top_clss = top_clss.cpu().numpy()
    img_dataset_idx = img_dataset.class_to_idx
    mapped_dic = {}
    for i in top_clss:
        if i in img_dataset_idx.values():
            mapped_dic[get_key(img_dataset_idx,i)] = categ_json_fl[get_key(img_dataset_idx,i)]
    return mapped_dic







def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    img = Image.open(image_path)
    data_transforms = transforms.Compose([transforms.Resize((224,224)), 
                                           transforms.ToTensor(), 
                                           transforms.Normalize(mean=mean_vals, std=std_vals)])
    
    # get a tensor structure for the specified format of image.
    img_tnsr = data_transforms(img)
    
    # transform the tensor to be processed on the gpu alternatively.
    img_tnsr = img_tnsr.to(device)
    
    #to remove a dimension of a tensor for example transform a 2D tensor into 1D tensor. __USE <tensor.unsqueeze()>
    img_tnsr.unsqueeze_(0)
    
    # receiving a gpu sturct. expecting cpu structure of image so the structure should be transferred into the cpu instead.
    img_tnsr = img_tnsr.cpu().clone()
    probe_vals = torch.exp(model(img_tnsr))

    top_probe, top_clss = probe_vals.topk(topk, dim=1)
    top_clss = map_dict(top_clss, cat_to_name, trn_img_dataset)
    top_probe.detach_()
    top_probe.squeeze_()
    return list(top_clss.values()), top_probe.numpy().tolist()
    # TODO: Implement the code to predict the class from an image file
