import numpy as np
import pydicom as dicom
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image
from random import shuffle
import time



def rgb2gray(rgb):
   fil = [0.299, 0.587, 0.144]
   return np.dot(rgb, fil)

def convert_to_gray(dicom_metadata):
    new_img = []
    for i in dicom_metadata:
        new = i.pixel_array.astype(float)
        new = rgb2gray(new)
        new_img.append(new)
    return np.stack(np.array(new_img))

def load_scan(path):
    slices = [ dicom.dcmread(i) for i in path ]
    slices_loc = [s for s in slices if 'SliceLocation' in s] 
    if slices_loc != []:
        slices = slices_loc
        try:
    #         slices = [s for s in slices if 'SliceLocation' in s]

            slices.sort(key = lambda x: int(x.InstanceNumber))
            try:
                slice_thickness = np.abs(slices[0].ImagePositionPatient[2] -   
                                  slices[1].ImagePositionPatient[2])
            except:
                try:
                    slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
                except:
                    try:
                        slice_thickness = slices[0].SliceThickness #if there's only 1 slice annotated for each study
                    except:
                        pass

            for s in slices:
                s.SliceThickness = slice_thickness
        except:
            pass
        
    return slices


def get_pixels_hu(scans):

    #check if Monochrome or RGB
    color = scans[0].PhotometricInterpretation 
    if color == 'MONOCHROME2':
        image = np.stack([s.pixel_array for s in scans])
        image = image.astype(np.int16)    
    else: #RGB, convert to Grayscale
        image = convert_to_gray(scans)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    try:
        # Convert to Hounsfield units (HU)
        intercept = scans[0].RescaleIntercept #intercept = -1024
        slope = scans[0].RescaleSlope #slope = 1
    except:
        intercept = -1024
        slope = 1.0
        
    if slope != 1:
        image = slope * image.astype(np.float32)
        image = image.astype(np.float32)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)


