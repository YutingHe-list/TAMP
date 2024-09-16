import numpy as np
import random
import SimpleITK as sitk
import time
from torch.utils.data import Dataset

class Database(Dataset):
    def __init__(self, opt):       

        self.input_folder = f"samples/fine-tuning/input/{opt.NICT_setting}_{opt.defect_degree}"
        self.label_folder = f"samples/fine-tuning/label"
        self.nii_start_index = opt.nii_start_index 
        self.queue_len = opt.queue_len
        self.labels_sets, self.inputs_sets = np.arange(1, 45), np.arange(1, 45)

    def __getitem__(self, index):        
        inputs = self.train_input[index]
        labels = self.train_label[index]
            
        return inputs, labels
    
    def __len__(self):
        return self.slice_num //100
    
    def load_nii_to_queue(self):
        
        self.file_queue = [] 
        self.slice_queue = []  
        self.slice_num = 0
        need_load_nii_num = self.queue_len
        while need_load_nii_num>0:

            input_set = self.inputs_sets[self.nii_start_index]
            label_set = self.labels_sets[self.nii_start_index]
            
            self.nii_start_index+=1

            new_original_path = f"{self.label_folder}/{label_set}.nii.gz"
            new_distorted_path = f"{self.input_folder}/{input_set}.nii.gz"

            start_time = time.time()
            labels =  sitk.GetArrayFromImage(sitk.ReadImage(new_original_path))
            inputs =  sitk.GetArrayFromImage(sitk.ReadImage(new_distorted_path))

            data_dict = [inputs, labels]
            print("Loading of Nii data number %d completed, time taken: %2fs." % (self.nii_start_index-1,time.time()-start_time))

            self.file_queue.append(data_dict)
            self.slice_queue.append(inputs.shape[0])
            self.slice_num += inputs.shape[0]
            need_load_nii_num -= 1

        print("\nThe loading of Nii training data is complete!")

    def init_train_sequence(self):
        self.train_input = [] 
        self.train_label = [] 

        for data_dict in self.file_queue:
            inputs, labels = data_dict[0], data_dict[1]

            for i in range(len(inputs)):
                input_slice = self.preprocess_data(inputs[i]) 
                label_slice = self.preprocess_data(labels[i])

                self.train_input.append(input_slice)
                self.train_label.append(label_slice)

        combined = list(zip(self.train_input, self.train_label))
        random.shuffle(combined)
        self.train_input, self.train_label = zip(*combined)

    def unstandard(self,standard_img):        
        mean=-556.882367
        variance=225653.408219
        nii_slice = standard_img * np.sqrt(variance) + mean
        return nii_slice        

    def standard(self,nii_slice):
        mean=-556.882367
        variance=225653.408219
        nii_slice = nii_slice.astype(np.float32)
        nii_slice = (nii_slice - mean) / np.sqrt(variance)
        return nii_slice

    def preprocess_data(self,nii_slice):
        nii_slice = self.standard(nii_slice)
        H, W = nii_slice.shape 

        if H < 512:
            pad_height = 512 - H
            top_padding = pad_height // 2
            bottom_padding = pad_height - top_padding
            nii_slice = np.pad(nii_slice, ((top_padding, bottom_padding), (0, 0)), mode='constant', constant_values=0)

        if W < 512:
            pad_width = 512 - W
            left_padding = pad_width // 2
            right_padding = pad_width - left_padding
            nii_slice = np.pad(nii_slice, ((0, 0), (left_padding, right_padding)), mode='constant', constant_values=0)

        if H > 512:
            nii_slice = nii_slice[:512, :]

        if W > 512:
            nii_slice = nii_slice[:, :512]

        return nii_slice
    
    def refresh_next_train(self):
        print("\nUpdating data...")
        
        self.slice_num = self.slice_num - self.slice_queue[0]
        self.file_queue.pop(0)
        self.slice_queue.pop(0)

        self.nii_start_index+=1

        input_set = self.inputs_sets[self.nii_start_index]
        label_set = self.labels_sets[self.nii_start_index]

        new_original_path = f"{self.label_folder}/{label_set}.nii.gz"
        new_distorted_path = f"{self.input_folder}/{input_set}.nii.gz"

        new_labels = sitk.GetArrayFromImage(sitk.ReadImage(new_original_path))
        new_inputs = sitk.GetArrayFromImage(sitk.ReadImage(new_distorted_path))
        self.file_queue.append([new_inputs, new_labels])
        self.slice_queue.append(new_inputs.shape[0])

        self.slice_num += self.slice_queue[-1]

        self.init_train_sequence()

        print("\nData update completed!")
