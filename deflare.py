import os

import scipy
from matplotlib import pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import glob
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from PIL import Image
from basicsr.data.flare7k_dataset import Flare_Image_Loader,RandomGammaCorrection
from basicsr.archs.uformer_arch import Uformer
from basicsr.archs.unet_arch import U_Net
from basicsr.utils.flare_util import blend_light_source,get_args_from_json,save_args_to_json,mkdir,predict_flare_from_6_channel,predict_flare_from_3_channel
from torch.distributions import Normal
import torchvision.transforms as transforms

def mkdir(path):
	folder = os.path.exists(path)
	if not folder:
		os.makedirs(path)

def demo(images_path,output_path,model_type,output_ch,pretrain_dir):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    test_path=glob.glob(images_path)
    result_path=output_path
    torch.cuda.empty_cache()
    if model_type=='Uformer':
        model=Uformer(img_size=512,img_ch=3,output_ch=output_ch).cuda()
        model.load_state_dict(torch.load(pretrain_dir))
    elif model_type=='U_Net' or model_type=='U-Net':
        model=U_Net(img_ch=3,output_ch=output_ch).cuda()
        model.load_state_dict(torch.load(pretrain_dir))
    else:
        assert False, "This model is not supported!!"
    to_tensor=transforms.ToTensor()
    resize=transforms.Resize((512,512)) #The output should in the shape of 128X
    for i,image_path in tqdm(enumerate(test_path)):
        mkdir(result_path+"deflare/")
        mkdir(result_path+"flare/")
        mkdir(result_path+"input/")
        mkdir(result_path+"blend/")

        deflare_path = result_path+"deflare/"+str(i).zfill(5)+"_deflare.png"
        flare_path = result_path+"flare/"+str(i).zfill(5)+"_flare.png"
        merge_path = result_path+"input/"+str(i).zfill(5)+"_input.png"
        blend_path = result_path+"blend/"+str(i).zfill(5)+"_blend.png"
        mask_path = result_path + "mask/"+str(i).zfill(5)+"_mask.png"
        merge_img = Image.open(image_path).convert("RGB")
        merge_img = resize(to_tensor(merge_img))
        merge_img = merge_img.cuda().unsqueeze(0)

        model.eval()
        with torch.no_grad():
            output_img=model(merge_img)
            #if ch is 6, first three channels are deflare image, others are flare image
            #if ch is 3, unsaturated region of output is the deflare image.
            gamma=torch.Tensor([2.2])
            if output_ch==6:
                deflare_img,flare_img_predicted,merge_img_predicted=predict_flare_from_6_channel(output_img,gamma)
            elif output_ch==3:
                flare_mask=torch.zeros_like(merge_img)
                deflare_img,flare_img_predicted=predict_flare_from_3_channel(output_img,flare_mask,output_img,merge_img,merge_img,gamma)
            else:
                assert False, "This output_ch is not supported!!"

            blend_img,mask= blend_light_source(merge_img, deflare_img, 0.97)
            mask = mask.cpu().numpy().transpose(1,2,0)
            plt.imshow(mask)
            plt.savefig(mask_path)
            torchvision.utils.save_image(merge_img, merge_path)
            torchvision.utils.save_image(flare_img_predicted, flare_path)
            torchvision.utils.save_image(deflare_img, deflare_path)
            torchvision.utils.save_image(blend_img, blend_path)


model_type="Uformer"
images_path="test/test_images/*.*"
result_path="result/test_images/Uformer/"
pretrain_dir='experiments/pretrained_models/uformer/net_g_last.pth'
output_ch=6
mask_flag=False
demo(images_path,result_path,model_type,output_ch,pretrain_dir)