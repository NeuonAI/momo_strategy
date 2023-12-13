# -*- coding: utf-8 -*-
"""
Created on Mon May 29 21:33:31 2023

@author: user
"""
import os
import numpy as np
import cv2
from PIL import Image
import copy
import random
import math


class database_module(object):

    def __init__(
                self,
                image_source_dir,
                database_file,
                batch,
                input_size,
                numclasses,
                max_sample,
                shuffle = False
            ):
        
        self.image_source_dir = image_source_dir
        self.database_file = database_file
        self.batch = batch
        self.input_size = input_size
        self.numclasses = numclasses
        self.max_sample = max_sample
        self.shuffle = shuffle
        
        self.load_data_list()
        self.cursor = 0
        self.epoch = 0
        self.epoch_complete = 0
        self.epoch_complete_list = []     
        self.negative_list = []
        self.predefine_dataset()
        
        

        
        
    def load_data_list(self):

        with open(self.database_file,'r') as fid:
            lines = [x.strip() for x in fid.readlines()]
        
            
        data_paths = [os.path.join(self.image_source_dir,
                                        x.split(' ')[0]) for x in lines]   
        
        data_labels2 = [int(x.split(' ')[3]) for x in lines] # family
        data_labels3 = [int(x.split(' ')[4]) for x in lines] # genus
        data_labels1 = [int(x.split(' ')[5]) for x in lines] # species
        data_labels4 = [int(x.split(' ')[1]) for x in lines] # class
        data_labels5 = [int(x.split(' ')[2]) for x in lines] # order
        
        
        self.data_paths = []
        self.data_labels2 = []
        self.data_labels3 = []
        self.data_labels1 = []
        self.data_labels4 = []
        self.data_labels5 = []
        
        #   Get only valid images
        for spe, fam, gen, cl, order, fp in zip(data_labels1, data_labels2, data_labels3, 
                                                data_labels4, data_labels5, data_paths):
            
            im = self.read_image(fp)
            if im is not None:
                self.data_paths.append(fp)
                self.data_labels2.append(fam)
                self.data_labels3.append(gen)
                self.data_labels1.append(spe)
                self.data_labels4.append(cl)
                self.data_labels5.append(order)
            
        
        #   Create species database dictionary
        self.database_dict_species = {}
        for spe, fam, gen, cl, order, fp in zip(self.data_labels1, self.data_labels2, self.data_labels3, 
                                                self.data_labels4, self.data_labels5, self.data_paths):
            if spe not in self.database_dict_species:
                self.database_dict_species[spe] = {"family": "", "genus": "", "class": "", "order": "", "filepaths": []}

            self.database_dict_species[spe]["family"] = fam
            self.database_dict_species[spe]["genus"] = gen
            self.database_dict_species[spe]["class"] = cl
            self.database_dict_species[spe]["order"] = order
            self.database_dict_species[spe]["filepaths"].append(fp)
            
              
        #   Create species counter dictionary
        self.database_dict_species_counter = {}
        for spe in self.database_dict_species.keys():
            if spe not in self.database_dict_species_counter:
                self.database_dict_species_counter[spe] = ""
            spe_filepaths_len = len(self.database_dict_species[spe]["filepaths"])
            

            self.database_dict_species_counter[spe] = spe_filepaths_len                
            
            
        #   Create family (species list) database dictionary
        self.database_dict_family = {}
        for spe, fam in zip(self.data_labels1, self.data_labels2):
            if fam not in self.database_dict_family:
                self.database_dict_family[fam] = []
            if spe not in self.database_dict_family[fam]:
                self.database_dict_family[fam].append(spe)
        

        #   Family and species unique labels
        self.unique_family_lbl = list(set(self.data_labels2))
        self.unique_species_lbl = list(set(self.data_labels1))
        

        self.reset_dict_species()
        self.reset_dict_species_list()
        self.reset_dict_species_counter()
        
        
    
    def reset_dict_species(self):
        self.database_dict_species_buffer = copy.deepcopy(self.database_dict_species)
        
    def reset_dict_species_list(self):
        self.unique_species_lbl_buffer = copy.deepcopy(self.unique_species_lbl)
        
    def reset_dict_species_counter(self):
        self.database_dict_species_counter_buffer = copy.deepcopy(self.database_dict_species_counter)
        
   
    def read_image(self, filepath):
        try:
            im = cv2.imread(filepath)
            if im is None:
               im = cv2.cvtColor(np.asarray(Image.open(filepath).convert('RGB')),cv2.COLOR_RGB2BGR)
            im = cv2.resize(im,(self.input_size[0:2]))
            if np.ndim(im) == 2:
                im = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
            else:
                im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)            
        except:
            pass
            
        return im      
    
    def check_negative_list(self):
        if self.negative_list != []:
            status = True
        else:
            status = False
        return status
    
        


    def predefine_dataset(self):
        print("Defining dataset")
        
        status = self.check_negative_list()
        if status == False:
            completed_species = []
            self.predefined_filepaths = []
            self.predefined_labels1 = []
            self.predefined_labels2 = []
            self.predefined_labels3 = []
            self.predefined_labels4 = []
            self.predefined_labels5 = []
            self.reset_dict_species_list()
            self.reset_dict_species_counter()
            counter = 0
            prev_predefined_filepaths_len = 0
            negative_predefined_filepaths_len = 0

            
        else:
            prev_predefined_filepaths_len = len(self.predefined_filepaths)
            negative_list_len = len(self.negative_list)
            max_negative_len = int(0.7 * len(self.predefined_filepaths))
            if max_negative_len > int(0.7 * len(self.data_paths)):
                max_negative_len = int(0.7 * len(self.data_paths))
                
            print("Max negative samples:", max_negative_len)

            if negative_list_len > max_negative_len:
                random_negative_list = random.sample(self.negative_list, max_negative_len)
            else:
                random_negative_list = copy.deepcopy(self.negative_list)
            
            self.reset_dict_species()
            completed_species = []
            self.predefined_filepaths = []
            self.predefined_labels1 = []
            self.predefined_labels2 = []
            self.predefined_labels3 = []
            self.predefined_labels4 = []
            self.predefined_labels5 = []
            self.reset_dict_species_list()
            self.reset_dict_species_counter()
            counter = 0
            self.negative_list = []
            
            for path in random_negative_list:
                path_index = self.data_paths.index(path)
                sample_class = self.data_labels4[path_index] # class
                sample_order = self.data_labels5[path_index] #order
                sample_family = self.data_labels2[path_index] #family
                sample_genus = self.data_labels3[path_index] #genus
                sample_species = self.data_labels1[path_index] #species
                

                counter += 1
                self.predefined_filepaths.append(path)
                self.predefined_labels1.append(sample_species)
                self.predefined_labels2.append(sample_family)
                self.predefined_labels3.append(sample_genus)
                self.predefined_labels4.append(sample_class)
                self.predefined_labels5.append(sample_order)
                
                species_counter = self.database_dict_species_counter_buffer[sample_species]

                if path in self.database_dict_species_buffer[sample_species]["filepaths"]:
                    self.database_dict_species_buffer[sample_species]["filepaths"].remove(path)
                self.database_dict_species_counter_buffer[sample_species] = species_counter - 1

                if species_counter < 1 and sample_species in self.unique_species_lbl_buffer:
                    self.unique_species_lbl_buffer.remove(sample_species)

                species_counter = self.database_dict_species_counter_buffer[sample_species]
                if sample_species not in completed_species and species_counter < 1:
                    completed_species.append(sample_species)                        

            negative_predefined_filepaths_len = len(random_negative_list)

            
        
        while len(completed_species) < len(self.unique_species_lbl):
            print("Counter:", counter, "Predefined dataset len:", len(self.predefined_filepaths), "Completed species:", len(completed_species), "/", len(self.unique_species_lbl))
            
            random_species = random.choice(self.unique_species_lbl_buffer)
            
                  
            species_filepaths = copy.deepcopy(self.database_dict_species_buffer[random_species]["filepaths"])
            species_family = copy.deepcopy(self.database_dict_species_buffer[random_species]["family"])
            species_genus = copy.deepcopy(self.database_dict_species_buffer[random_species]["genus"])
            species_class = copy.deepcopy(self.database_dict_species_buffer[random_species]["class"])
            species_order = copy.deepcopy(self.database_dict_species_buffer[random_species]["order"])
            
            species_counter = self.database_dict_species_counter_buffer[random_species]
            
            if len(species_filepaths) >= 1 and species_counter >= 1:
                random_filepaths = random.sample(species_filepaths, 1)


                for random_filepath in random_filepaths:

                    counter += 1
                    self.predefined_filepaths.append(random_filepath)
                    self.predefined_labels1.append(random_species)
                    self.predefined_labels2.append(species_family)
                    self.predefined_labels3.append(species_genus)
                    self.predefined_labels4.append(species_class)
                    self.predefined_labels5.append(species_order)
                    self.database_dict_species_buffer[random_species]["filepaths"].remove(random_filepath)
                    self.database_dict_species_counter_buffer[random_species] = species_counter - len(random_filepaths)#1



                        
            else:
                
                if len(species_filepaths) == 0:
                    
                    if random_species not in completed_species:
                        self.database_dict_species_buffer[random_species]["filepaths"] = copy.deepcopy(self.database_dict_species[random_species]["filepaths"])
                    
                    
                    if random_species not in self.epoch_complete_list:
                        self.epoch_complete_list.append(random_species)

                
                if species_counter < 1 and random_species in self.unique_species_lbl_buffer:
                    self.unique_species_lbl_buffer.remove(random_species)
                        

            species_counter = self.database_dict_species_counter_buffer[random_species]
            if random_species not in completed_species and species_counter < 1:
                completed_species.append(random_species)                      
                    
            
            if len(self.unique_species_lbl_buffer) < 1:       
                self.reset_dict_species_list()
          
      
      

        current_predefined_filepaths_len = len(self.predefined_filepaths)
        positive_predefined_filepaths_len = current_predefined_filepaths_len - negative_predefined_filepaths_len
        
        positive_percentage = round(positive_predefined_filepaths_len / current_predefined_filepaths_len,2)
        print("Current fp:", current_predefined_filepaths_len, "Prev:", prev_predefined_filepaths_len)
        if positive_percentage < 0.3:
            positive_overall = 0.3 * current_predefined_filepaths_len
            added_len = positive_overall - positive_predefined_filepaths_len
        
        else:
            added_len = 0
        
        print("====================================================================")
        print("Added:", added_len)
        added_predefined_filepaths_len = current_predefined_filepaths_len + added_len
        print("Total:", added_predefined_filepaths_len)
        fullbatch_iter = math.ceil(added_predefined_filepaths_len / self.batch)
        total_filepaths_fullbatch = self.batch * fullbatch_iter
        print("Complete:", total_filepaths_fullbatch)
        print("====================================================================")
        

        
                
            
        while len(self.predefined_filepaths) < total_filepaths_fullbatch: 
            print("Patching counter:", counter, "Predefined dataset len:", len(self.predefined_filepaths), "/", total_filepaths_fullbatch)

            
            random_species = random.choice(self.unique_species_lbl_buffer)
            
                  
            species_filepaths = copy.deepcopy(self.database_dict_species_buffer[random_species]["filepaths"])
            species_family = copy.deepcopy(self.database_dict_species_buffer[random_species]["family"])
            species_genus = copy.deepcopy(self.database_dict_species_buffer[random_species]["genus"])
            species_class = copy.deepcopy(self.database_dict_species_buffer[random_species]["class"])
            species_order = copy.deepcopy(self.database_dict_species_buffer[random_species]["order"])
            
            species_counter = self.database_dict_species_counter_buffer[random_species]
            
            if len(species_filepaths) >= 1 and species_counter >= 1:
                random_filepaths = random.sample(species_filepaths, 1)

                for random_filepath in random_filepaths:

                    counter += 1
                    self.predefined_filepaths.append(random_filepath)
                    self.predefined_labels1.append(random_species)
                    self.predefined_labels2.append(species_family)
                    self.predefined_labels3.append(species_genus)
                    self.predefined_labels4.append(species_class)
                    self.predefined_labels5.append(species_order)
                    self.database_dict_species_buffer[random_species]["filepaths"].remove(random_filepath)
                    self.database_dict_species_counter_buffer[random_species] = species_counter - len(random_filepaths)#1


                        
            else:
                
                if len(species_filepaths) == 0:

                    self.database_dict_species_buffer[random_species]["filepaths"] = copy.deepcopy(self.database_dict_species[random_species]["filepaths"])
                

                    if random_species not in self.epoch_complete_list:
                        self.epoch_complete_list.append(random_species)

                
                if species_counter < 1 and random_species in self.unique_species_lbl_buffer:
                    self.unique_species_lbl_buffer.remove(random_species)
                       
                 
                        
            if len(self.unique_species_lbl_buffer) < 1:       
                self.reset_dict_species_list()
                self.reset_dict_species_counter()
            

        print("Final predefined:", len(self.predefined_filepaths))


        self.shuffle_predefined_dataset()
        
        
    
    def shuffle_predefined_dataset(self):
        print("Shuffling predefined dataset...")
        data_num = len(self.predefined_filepaths)
        data_idx = np.arange(data_num)
        np.random.shuffle(data_idx)
        
        self.shuffled_predefined_filepaths = [self.predefined_filepaths[x] for x in data_idx]
        self.shuffled_predefined_labels1 = [self.predefined_labels1[x] for x in data_idx]
        self.shuffled_predefined_labels2 = [self.predefined_labels2[x] for x in data_idx]
        self.shuffled_predefined_labels3 = [self.predefined_labels3[x] for x in data_idx]
        self.shuffled_predefined_labels4 = [self.predefined_labels4[x] for x in data_idx]
        self.shuffled_predefined_labels5 = [self.predefined_labels5[x] for x in data_idx]


        
    def update_negative_list(self,current_wrong_predictions):
        self.negative_list = copy.deepcopy(current_wrong_predictions)
    
    def read_batch(self):
        img = []
        lbl1 = []
        lbl2 = []
        lbl3 = []
        lbl4 = []
        lbl5 = []
        filepaths = []

        while len(img) < self.batch:
            try:
            
                im = cv2.imread(self.shuffled_predefined_filepaths[self.cursor])
                if im is None:
                   im = cv2.cvtColor(np.asarray(Image.open(self.shuffled_predefined_filepaths[self.cursor]).convert('RGB')),cv2.COLOR_RGB2BGR)
                im = cv2.resize(im,(self.input_size[0:2]))
                if np.ndim(im) == 2:
                    img.append(cv2.cvtColor(im,cv2.COLOR_GRAY2RGB))
                else:
                    img.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
                lbl1.append(self.shuffled_predefined_labels1[self.cursor])
                lbl2.append(self.shuffled_predefined_labels2[self.cursor])
                lbl3.append(self.shuffled_predefined_labels3[self.cursor])
                lbl4.append(self.shuffled_predefined_labels4[self.cursor])
                lbl5.append(self.shuffled_predefined_labels5[self.cursor])
                filepaths.append(self.shuffled_predefined_filepaths[self.cursor])

            except:
                pass
            
            self.cursor += 1
            if self.cursor >= len(self.shuffled_predefined_filepaths):
                self.predefine_dataset()
                self.cursor = 0
                self.epoch += 1

                
            if len(self.epoch_complete_list) == len(self.unique_species_lbl):
                self.epoch_complete += 1
                self.epoch_complete_list = []
        
        img = np.asarray(img,dtype=np.float32)/255.0
        lbl1 = np.asarray(lbl1,dtype=np.int32)
        lbl2 = np.asarray(lbl2,dtype=np.int32)
        lbl3 = np.asarray(lbl3,dtype=np.int32)
        lbl4 = np.asarray(lbl4,dtype=np.int32)
        lbl5 = np.asarray(lbl5,dtype=np.int32)
        
        return (img,lbl1,lbl2,lbl3,lbl4,lbl5,filepaths)

    
    
    
    
    
    
