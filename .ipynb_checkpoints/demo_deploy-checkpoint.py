import numpy as np
import pandas as pd
import gradio as gr
import tensorflow
from tensorflow.keras.applications.mobilenet import preprocess_input
import sys
if not sys.warnoptions:
    import warnings
    
#Stages
t_class = ['1','1a','1b','1c','2','2a','2b','3','4','is'] 
n_class = ['0','1','2','3']
m_class = ['0','1','1a','1b','1c','2','3'] 

#Load all models
t_mobilenetv2 = tensorflow.keras.models.load_model(r'D:\Lung-PET-CT-Dx\manifest\Models\multiclass_t_stage_mobilenet_v3.h5')
n_mobilenetv2 = tensorflow.keras.models.load_model(r'D:\Lung-PET-CT-Dx\manifest\Models\multiclass_n_stage_mobilenet.h5')
m_mobilenetv2 = tensorflow.keras.models.load_model(r'D:\Lung-PET-CT-Dx\manifest\Models\multiclass_m_stage_mobilenet.h5')

def predict_t_stage_mobnet(img):
  img = tensorflow.image.resize(img, size=(224,224))
  # Make image tensor right size for model
  img = tensorflow.expand_dims(img, axis=0)
  img_preprocessed = preprocess_input(img)    
  # Predict the image
  pred = t_mobilenetv2.predict(img_preprocessed)[0]
  output = dict(zip(t_class, map(float,pred)))
  return output

def predict_m_stage_mobnet(img):
  img = tensorflow.image.resize(img, size=(224,224))
  # Make image tensor right size for model
  img = tensorflow.expand_dims(img, axis=0)
  img_preprocessed = preprocess_input(img)    
  # Predict the image
  pred = m_mobilenetv2.predict(img_preprocessed)[0]
  output = dict(zip(m_class, map(float,pred)))
  return output

def predict_n_stage_mobnet(img):
  img = tensorflow.image.resize(img, size=(224,224))
  # Make image tensor right size for model
  img = tensorflow.expand_dims(img, axis=0)
  img_preprocessed = preprocess_input(img)    
  # Predict the image
  pred = n_mobilenetv2.predict(img_preprocessed)[0]
  output = dict(zip(n_class, map(float,pred)))
  return output

def main():
    t_mobnet_int = gr.Interface(fn=predict_t_stage_mobnet,
             inputs='image',
             outputs=gr.outputs.Label(num_top_classes=3,label='T-Stage'),
             live=False)
    n_mobnet_int = gr.Interface(fn=predict_n_stage_mobnet,
             inputs='image',
             outputs=gr.outputs.Label(num_top_classes=3,label='N-Stage'),
             live=False)
    m_mobnet_int = gr.Interface(fn=predict_m_stage_mobnet,
             inputs='image',
             outputs=gr.outputs.Label(num_top_classes=3,label='M-Stage'),
             live=False)
    
    with gr.Blocks() as mobnetv2_demo:
        gr.Markdown("# TNM Staging Demo ")
        gr.Parallel(t_mobnet_int, n_mobnet_int, m_mobnet_int)
    
    mobnetv2_demo.launch(show_api=False,share=True)
    
if __name__== "__main__":
    main()