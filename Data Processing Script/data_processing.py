"""
Data Processing for Lung-PET-CT-Dx for Lung Cancer Staging 

"""
import os, time, copy, argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import *
from skimage.transform import resize
import warnings
# Settings the warnings to be ignored
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser('Data Processing')
    parser.add_argument('--clinicaldata-path', type=str, default='clinical data.csv',
                        help='path to the csv file that contains the clinical data')
    parser.add_argument('--annotation-path', type=str,
                        help='path to the folder which stores the annotation files (.xml) or a path to a single annotation file')
    parser.add_argument('--ctimages-path', type=str, 
                        help='path to the folder where to store the converted dicom files to .tiff files')
    return parser.parse_args()

def main():
    args = parse_args()
    clinical_data = pd.read_csv(args.clinicaldata_path) #clinical data.csv
    
    try:
        ct_df = pd.read_csv(args.annotation_path) #annotated_files.csv
    except Exception as e:
        print('Error in reading the clinical data csv file. {}'.format(e))
    ct_df.drop(columns=['No'],inplace=True)
    ct_df.sort_values(['Patient_ID','Path'],ascending=True,inplace=True)
    ct_df.reset_index(drop=True,inplace=True)

    #add column for study folder
    ct_df['Study'] = ct_df['Path'].map(lambda x: x.split('\\')[-2])

    patients = ct_df['Patient_ID'].unique().tolist()
    patients = clinical_data['NewPatientID'].to_list()
    
    #Check if all patient ID are valid 
    annotated_patients = ct_df['Patient_ID'].unique().tolist()
    not_annotated = set(patients) - set(annotated_patients)
    not_annotated_list =  list(not_annotated) #there's no patient ID A0266 in the dataset
    print('The following Patient IDs are invalid for use:', not_annotated_list)     


    cwd_path = args.ctimages_path #file path  to store images/current work dir
    try:
        os.chdir(cwd_path)
    except Exception:
        print('Error in changing the current working directory. Check the path')

    #Dataframe
    dir_df = pd.DataFrame(columns=['filenames','labels'])
    csv_path = os.path.join(cwd_path,'raw ct images.csv')

    start = time.time() #measure time elapsed
    print('Converting DICOM files to Image files...')
    for patient in tqdm(patients,desc='Converting DICOM files per PatientID'):
        studies = ct_df[ct_df['Patient_ID']==patient]['Study'].unique().tolist()
        for study in studies:
            image_paths = ct_df[(ct_df['Patient_ID']==patient) & (ct_df['Study']==study)]['Path'].tolist()
            patient_dicom = load_scan(image_paths)
            patient_pixels = get_pixels_hu(patient_dicom)
                    
            #Get only CT images
            if patient_dicom[0].Modality == 'CT':
                #get the labels 
                t = ct_df[(ct_df['Patient_ID']==patient) & (ct_df['Study']==study)]['T-Stage'].unique()[0]
                m = ct_df[(ct_df['Patient_ID']==patient) & (ct_df['Study']==study)]['M-Stage'].unique()[0]
                n = ct_df[(ct_df['Patient_ID']==patient) & (ct_df['Study']==study)]['N-Stage'].unique()[0]
                labels = [t,n,m]
                
                #filenames
                filenames = list((map(lambda x: '_'.join(x.split('\\')[-4:]),image_paths)))
                filenames = list(map(lambda x: x[:-4],filenames))
                
                #save each slice to only one folder
                idx = 0
                for patient_pixel in patient_pixels:
                    filename = filenames[idx] + '.tiff'
                    
                    #resize to 224 x 224
                    slice_resized = resize(patient_pixel,(224,224))
                    
                    #save masks to destinaton folder
                    matplotlib.image.imsave(filename, slice_resized, cmap='gray')
        
                    #save to dataframe
                    df_row = pd.DataFrame([{'filenames':filename,'labels':labels}])
                    dir_df = pd.concat([dir_df,df_row],ignore_index=True)  
                    
                    
                    idx+=1       

    dir_df.to_csv(csv_path)
    end = time.time()
    time_elapsed = end - start
    
    print('Converted images have been stored successfully.')
    print('time elapsed:',time_elapsed, ' seconds')

if __name__ == '__main__':    
    # print('Use command "-h" for help. Enter the paths for dicom files, annotation file and clinical data path')
    main()
