from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys
import pandas as pd

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''
        
        
        self.txt_list = self.split+".txt"    #cosí ho la lista dei file che adopereró ["train.txt", "test.txt"]
        df = pd.read_csv("/content/Caltech101/"+self.txt_list, sep=' ', index_col=0) #leggo i file
        #con index_col=0 impongo pandas a mettere il label all'indice 0
        #sep ' ' é il separatore 
        self.img_names = df.index.values
        self.root = root #in root é presente la locazione del direttorio principale
        self.transform = transform #viene salvata la trasformazione da effettuare al dataset, al momento nessuna
        classes, class_to_idx = self._find_classes(self.root) #trovo le classi
        self.classes = classes #salvo le classi
        self.class_to_idx = class_to_idx #meccanismo per passare da classe ad indice
        self.samples = []
        for i in self.img_names:
            c_name = i.split("/")[0] #prendo la prima parte del nome
            if(c_name != "BACKGROUND_Google"): #se non equivale alla classe BACKGROUND_google (che é una classe da scartare)
                c_id = class_to_idx[c_name] #trovo il suo id
                self.samples.append(tuple((i, c_id))) # e lo metto nella lista di samples
    
        def _find_classes(self, dir: str):
            """
            Finds the class folders in a dataset.

            Args:
                dir (string): Root directory path.

            Returns:
                tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

            Ensures:
                No class is a subdirectory of another.
            """
            classes = [d.name for d in os.scandir(dir) if d.is_dir() and d.name != "BACKGROUND_Google"]
            classes.sort()
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
            return classes, class_to_idx
        

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        image, label = self.samples[index] # Provide a way to access image and label via index
                           # Image should be a PIL Image
                           # label can be int

        image = pil_loader(image) 
        
        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length =  len(self.samples) # Provide a way to get the length (number of elements) of the dataset
        return lenght
