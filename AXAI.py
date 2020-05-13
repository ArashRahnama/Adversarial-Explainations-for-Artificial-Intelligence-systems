## Arash Rahnama
import torch
import copy
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import quickshift
from torchvision import datasets, transforms
###################################################################
class AXAI():
    def __init__(self, model, inputs, loss_criterion, std, mean):
        ## initialization function
        ## model: model to explain
        ## inputs: data to explain
        ## loss_criterion: loss function of the model used for adversarial attack
        ## mean, std: mean and std of dataset used for pre-processing
        self.model = model
        self.inputs = inputs
        self.loss_criterion = loss_criterion
        self.std = std
        self.mean = mean
        
    def tensor2cuda(self, tensor):
        ## AXAI utilizes the GPU if GPUs are detected on the user's machine.
        ## tensor: Pytorch tensor
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        return tensor

    def gen_adv(self):
        ## this function generates adversarial inputs using the PGD attack
        ## diff: this is the difference between clean and adversarial input
        self.model.eval()
        with torch.no_grad():
            adv_examples=[]
            data = self.tensor2cuda(self.inputs)
            output = self.model(data)
            pred = torch.max(output, dim=1)[1] 
            with torch.enable_grad():
                adv_data, diff_tmp= self.pgdm(net=self.model, x=data,\
                y=pred, loss_criterion=self.loss_criterion, \
                alpha=0.001, eps=.2, steps=20, radius=.2, norm=2)         
            diff = diff_tmp.squeeze().detach().cpu().numpy()
        return diff
    
    def Attack_and_Filter(self):
        ## this fucntion returns the Filtered_Attacks (refer to the paper)
        ## this function wraps gen_adv() and threshold()
        Attacks = self.gen_adv()
        Filtered_Attacks = self.threshold(Attacks)
        return Filtered_Attacks
    
    def explain(self, K=5, kernel_size=8, max_dist=10, ratio=.1):
        ## this function performs segmentation via QuickShift (Vedaldi et al., 2008)
        ## maps the filtered adversarial attacks back to the original input.
        ## K explanation length, i.e., the number of explainable segments one desires to show
        ## kernel_size, max_dist, and ratio are three input parameters for QuickShift. 
        ## it is suggested that the user experiment with the 4 arguments in the explain() method to get the best output 
        ## please refer to our paper for further explanations
        ## this routine returns the original image and its explanation

        # pulls out the filtered attacks
        Filtered_Attacks = self.Attack_and_Filter()
        
        # image segmentation process
        data_org = self.inputs.squeeze().detach().cpu().numpy()
        image = np.transpose(data_org, (1, 2, 0))
        segments_orig = quickshift(image, kernel_size=kernel_size,
                                   max_dist=max_dist, ratio=ratio)

        values, counts = np.unique(segments_orig, return_counts=True)
        attack_frequency=[]
        attack_intensity=[]

        for i in range(len(values)):
            segments_orig_loc=segments_orig==values[i]
            tmp = np.logical_and(segments_orig_loc,Filtered_Attacks)
            attack_frequency.append(np.sum(tmp))
            attack_intensity.append(np.sum(tmp)/counts[i])
            
        # mapping process
        top_attack = np.sort(attack_intensity)[::-1][:K]
        zero_filter = np.zeros(np.array(attack_intensity).shape, 
                               dtype=bool)
        for i in range(len(top_attack)):
            intensity_filter = attack_intensity == top_attack[i]
            zero_filter = zero_filter+intensity_filter

        strongly_attacked_list = values[zero_filter]
        un_slightly_attacked_list = np.delete(values, 
                                              strongly_attacked_list)
        strongly_attacked_image = copy.deepcopy(image)
        for x in un_slightly_attacked_list:
            strongly_attacked_image[segments_orig == x] = (255,255,255)
        
        # turn the original_img into the desired format
        original_img = np.transpose(self.inputs.squeeze().
                                    detach().cpu().numpy(), (1, 2, 0))

        return self.denormalize(strongly_attacked_image),\
               self.denormalize(original_img)

            
    def threshold(self,diff, percentage=15):
        ## According to our paper to produce explanations and filter out unuseful features,
        ## one needs to maskout the features based on a thershold value (refer to our paper)
        ## diff: adversarial attack difference matrix generated via diff
        ## percentage: percentile threshold
        ## returns the filtered attacks
        dif_total_1 = copy.deepcopy(diff[0])
        dif_total_2 = copy.deepcopy(diff[1])
        dif_total_3 = copy.deepcopy(diff[2])
        thres_1_1=np.percentile(dif_total_1, percentage)
        thres_1_2=np.percentile(dif_total_1, 100-percentage)
        mask_1_2 = (dif_total_1 >= thres_1_1) &\
                    (dif_total_1 < thres_1_2)
        dif_total_1[mask_1_2] = 0
        
        thres_2_1=np.percentile(dif_total_2, percentage)
        thres_2_2=np.percentile(dif_total_2, 100-percentage)
        mask_2_2 = (dif_total_2 >= thres_2_1) &\
                    (dif_total_2 < thres_2_2)
        dif_total_2[mask_2_2] = 0

        thres_3_1=np.percentile(dif_total_3, percentage)
        thres_3_2=np.percentile(dif_total_3, 100-percentage)
        mask_3_2 = (dif_total_3 >= thres_3_1) &\
                    (dif_total_3 < thres_3_2)
        dif_total_3[mask_3_2] = 0        
        dif_total = dif_total_1+dif_total_2+dif_total_3

        return dif_total

    def pgdm(self,net, x, y, loss_criterion, alpha, eps, steps,\
             radius, norm):
        ## the implemntation of the projected gradient descenct method (Madry et al., 2017)
        ## steps: number of steps 
        pgd = x.new_zeros(x.shape)
        adv_x = x + pgd
        for step in range(steps):
            pgd = pgd.detach()
            x = x.detach()
            adv_x = adv_x.clone().detach()
            adv_x.requires_grad = True 
            preds = net(adv_x)
            net.zero_grad()
            loss = loss_criterion(preds, y)
            loss.backward(create_graph=False, retain_graph=False)
            adv_x_grad = adv_x.grad
            scaled_adv_x_grad = adv_x_grad/adv_x_grad.\
                                view(adv_x.shape[0], -1)\
                                .norm(norm, dim=-1).view(-1, 1, 1, 1)

            pgd = pgd + (alpha*scaled_adv_x_grad)

            mask = pgd.view(pgd.shape[0], -1).norm(norm, dim=1) <= eps
            scaling_factor = pgd.view(pgd.shape[0], -1).\
                             norm(norm, dim=1)
            scaling_factor[mask] = eps
            pgd *= eps / scaling_factor.view(-1, 1, 1, 1)
            adv_x = x + pgd 
        return adv_x, pgd
    
    def denormalize(self,norm_img):
        ## this function de-normalizes a normalized image for visualization
        return norm_img*self.std+self.mean
    
    def plotter(self,Explanations,save_path=None,save=False):
        ## this function plots a side by side image (original image and the corresponding explanation)
        ## if save=True, then the plot is saved as a .png file
        original_img = np.transpose(self.inputs.squeeze().
                                    detach().cpu().numpy(), (1, 2, 0))
        original_img = self.denormalize(original_img)

        fig = plt.figure(figsize=(6,3))
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(original_img)
        ax.axis('off')
        ax = fig.add_subplot(1, 2, 2)
        ax.imshow(Explanations)
        ax.axis('off')
        if save:
            plt.savefig(os.path.join(save_path,'_explanation.png'),
                        dpi=300,bbox_inches='tight')
