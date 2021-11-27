import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2 
from PIL import Image
import PIL
from torch.autograd import Variable
from network_def import SortingClassifier128
#import shap
#import pickle
#import argparse
  
  
def run_live():
    # define a video capture object 
    vid = cv2.VideoCapture(0) 
    loader = transforms.Compose([transforms.ToTensor()])

    # model = MiniVggNet()
    # model.load_state_dict(torch.load('net_epoch_3.pth'))
    model = SortingClassifier128()
    model.load_state_dict(torch.load('bb_epoch_6.pth')) #77 and 76 74 73
    model.eval()

    font = cv2.FONT_HERSHEY_SIMPLEX 

    while(True): 
        
        # Capture the video frame 
        # by frame 
        ret, frame = vid.read() 
        
        frame = cv2.resize(frame,(frame.shape[1]//5, frame.shape[0]//5))
        frame = frame[0:80, 0:80]
    
        img = np.asarray(frame) # convert to numpy
        img = (Image.fromarray(img)).convert('L') # convert to PIL gray
        img = (np.asarray(img))/255 # convert to numpy
        img = (torch.Tensor(img).float()) # convert to tensor
        img = img.unsqueeze(0) # add a dimension
        img = img.unsqueeze(0)
        #print(img.size())
        
        img_test = Variable(img, requires_grad=True)
        out1, out2 = model(img_test)
        #out = model(img_test)
        
        #print(output.data[0][0], output.data[0][1], output.data[0][2], output.data[0][3])
        #print(output[0][0], output[0][1])
        #frame = cv2.rectangle(frame, (int(out2.data[0][0]),int(out2.data[0][1])), (int(out2.data[0][0]+out2.data[0][2]),int(out2.data[0][1]+out2.data[0][3])), (255,0,0))
        #frame = cv2.circle(frame, (output.data[0][0],output.data[0][1]), radius=1, color=(0, 0, 255), thickness=-1)
        #frame = cv2.circle(frame, (output.data[0][2],output.data[0][3]), radius=1, color=(0, 0, 255), thickness=-1)
        # _, result = torch.max(output,1)
        #print(output[0][0], output[0][1])
        
        diff = out1[0][0] - out1[0][1]
        print(out2.data,end='\r')
        if diff > 0: #result.data[0] == 0 :
            cv2.putText(frame, 'Person',(0, 20),font, 0.5,(0, 255, 255),2,cv2.LINE_4) 
            frame = cv2.rectangle(frame, (int(out2.data[0][0]),int(out2.data[0][1])), (int(out2.data[0][0]+out2.data[0][2]),int(out2.data[0][1]+out2.data[0][3])), (255,0,0),thickness=2)
        else:
            cv2.putText(frame, 'None',(0, 20),font, 0.5,(0, 255, 255),2,cv2.LINE_4) 
            

        # diff = out[0][0] - out[0][1]
        # print(diff,end='\r')
        # if diff > 2: #result.data[0] == 0 :
        #     cv2.putText(frame, 'Person',(0, 20),font, 0.5,(0, 255, 255),2,cv2.LINE_4) 
        # else:
        #     cv2.putText(frame, 'None',(0, 20),font, 0.5,(0, 255, 255),2,cv2.LINE_4) 
            
            
    
        # Use putText() method for 
        # inserting text on video 
        
        # Display the resulting frame 
        cv2.imshow('frame', frame) 
        
        # the 'q' button is set as the 
        # quitting button you may use any 
        # desired button of your choice 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    
    # After the loop release the cap object 
    vid.release() 
    # Destroy all the windows 
    cv2.destroyAllWindows()
    
def test_shap(test_loader, model):
    #matplotlib.use('TkAgg')
    print("Generating plot...")
    #images, _ = iter(test_loader).next()
    batch = next(iter(test_loader))
    images, _ = batch
    print(images.size())
    background = images[:100]
    test_images = images[100:100 + 2]
    
    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(test_images)
    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)
    # Plot the feature attributions
    shap.image_plot(shap_numpy, test_numpy)
 
 
if __name__ == "__main__":
    run_live()
    # model = GeffNet()
    # model.load_state_dict(torch.load('net.pth'))
    # #model.eval()
    # train_dataset, test_dataset = geffnet_faces_get_datasets()
    # testloader = DataLoader(test_dataset, batch_size=256, num_workers=4, shuffle=True)
    # test_shap(testloader,model)