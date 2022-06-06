import json
import os
import cv2
import numpy as np
from shutil import copyfile
import math

class SyntheticDataset:
    def __init__(self, path_to_dataset, grid_shape=(5, 5)):
        self._path_to_dataset = path_to_dataset
        json_file = open(os.path.join(self._path_to_dataset, 'annotations.json'))
        data = json.load(json_file)
        self.annotations = data['annotations']
        self.num_samples = len(self.annotations)
        self.num_grid_x, self.num_grid_y = grid_shape

    def display_target(self, M, source_img, ret=False):
        img = source_img.copy()
        img_height,img_width = img.shape[:2]
        grid_dim_x = img_width/self.num_grid_x
        grid_dim_y = img_height/self.num_grid_y

        annotations = []
        for i in range(self.num_grid_y):
            for j in range(self.num_grid_x):

                if M[i,j,0] > 0.7:
                    cx, cy,  distance, yaw_relative = M[i,j,1:]
                    rw, rh = 0, 0
                    print(M[i,j,0],i,j,cx,cy,rh, distance, yaw_relative)

                    cx_on_img = int(j*grid_dim_x + cx*grid_dim_x)
                    cy_on_img = int(i*grid_dim_y + cy*grid_dim_y)
                    cv2.circle(img, (cx_on_img, cy_on_img), 3, (0,0,255), 3)                

                    h = int(rh*img_height)
                    w = int(rw*img_width)

                    cv2.rectangle(img, (int(cx_on_img-(w/2)), int(cy_on_img-(h/2))), 
                                        (int(cx_on_img+(w/2)), int(cy_on_img+(h/2))), (0,0,255), 3)
            

                    yaw_relative = yaw_relative/np.pi*180.

                    rel_text = "{:3.2f}".format(abs(distance))+" m."
                    yaw_text = "{:3.2f}".format(yaw_relative)+" deg." 

                    cv2.putText(img, rel_text, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 2, lineType=cv2.LINE_AA)                      
                    cv2.putText(img, yaw_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 2, lineType=cv2.LINE_AA)                      

                    annotations.append([cx_on_img, cy_on_img, abs(float(distance))])

        if ret:
            return img, annotations



    def display_target_large_annot(self, M, source_img, ret=False):
        img = source_img.copy()
        img_height,img_width = img.shape[:2]
        grid_dim_x = img_width/self.num_grid_x
        grid_dim_y = img_height/self.num_grid_y

        annotations = []
        for i in range(self.num_grid_y):
            for j in range(self.num_grid_x):

                if M[i,j,0] > 0.8:
                    cx, cy,  gx, gy, gz, yaw_relative = M[i,j,1:]
                    rw, rh = 0, 0
                    print(M[i,j,0],i,j,cx,cy,gx,gy,gz, yaw_relative)

                    cx_on_img = int(j*grid_dim_x + cx*grid_dim_x)
                    cy_on_img = int(i*grid_dim_y + cy*grid_dim_y)
                    cv2.circle(img, (cx_on_img, cy_on_img), 3, (0,0,255), 3)                

                    h = int(rh*img_height)
                    w = int(rw*img_width)

                    cv2.rectangle(img, (int(cx_on_img-(w/2)), int(cy_on_img-(h/2))), 
                                        (int(cx_on_img+(w/2)), int(cy_on_img+(h/2))), (0,0,255), 3)

                    yaw_relative = yaw_relative/np.pi*180.

                    distance = np.sqrt(gx**2 + gy**2 + gz**2)
                    rel_text = "{:3.2f}".format(abs(distance))+" m."
                    yaw_text = "{:3.2f}".format(yaw_relative)+" deg." 

                    cv2.putText(img, rel_text, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 2, lineType=cv2.LINE_AA)                      
                    cv2.putText(img, yaw_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 2, lineType=cv2.LINE_AA)                      

                    annotations.append([cx_on_img, cy_on_img, abs(float(distance))])

        if ret:
            return img, annotations




    def get_data_by_index(self, index, save_img=False, show_grid=False):

        annotation = self.annotations[index]

        img = cv2.imread(os.path.join(self._path_to_dataset, 'images', annotation['image']))
        ori = img.copy()
        assert(img is not None)

        img_height,img_width = img.shape[:2]
        grid_dim_x = img_width/self.num_grid_x
        grid_dim_y = img_height/self.num_grid_y
        
        # prob,cx,cy,rw,rh,distance, yaw_rel
        M = np.zeros( (self.num_grid_y, self.num_grid_x, 5), dtype=np.float32)
        for annot in annotation['annotations']:

            # Read bbox.
            x = annot['xmin']
            y = annot['ymin']
            w = annot['xmax']-annot['xmin']
            h = annot['ymax']-annot['ymin']

            if "center_x" in annot.keys():
                cx, cy = annot["center_x"], annot["center_y"]
            else:  
                exit()

            x_index = int(cx/grid_dim_x)
            y_index = int(cy/grid_dim_y)

            cx, cy = (cx % grid_dim_x)/grid_dim_x, (cy % grid_dim_y)/grid_dim_y

            # Read distance and orientation.
            distance = annot['distance']

            yaw_relative = annot['yaw_relative']


            M[y_index, x_index]=np.array([1., cx, cy, distance, yaw_relative])

        if save_img:
            for annot in annotation['annotations']:
                cv2.rectangle(img, (annot['xmin'], annot['ymin']), (annot['xmax'], annot['ymax']), (0,0,255), 3)

            if show_grid:
                for i in range(self.num_grid_y+1):
                    for j in range(self.num_grid_x+1):
                        cv2.line(img, (int(j*grid_dim_x), 0), (int(j*grid_dim_x), img_height), (0,255,0), 1)
                        cv2.line(img, (0, int(i*grid_dim_y)), (img_width, int(i*grid_dim_y)), (0,255,0), 1)

            cv2.imwrite('img.png', img)
            self.display_target(M, ori)

        return img, M

import json
import os
import cv2
import numpy as np
from shutil import copyfile


def display_target(M, source_img, ret=False):
    img = source_img.copy()
    img_height,img_width = img.shape[:2]
    grid_dim_x = img_width/5
    grid_dim_y = img_height/5

    bbox_results = []
    for i in range(5):
        for j in range(5):
            #cv2.line(img, (int(j*grid_dim_x), 0), (int(j*grid_dim_x), img_height), (0,255,0), 1)
            #cv2.line(img, (0, int(i*grid_dim_y)), (img_width, int(i*grid_dim_y)), (0,255,0), 1)

            if M[i,j,0] > 0.5:
                cx, cy, rw, rh, distance, yaw_relative = M[i,j,1:]

                #print(M[i,j,0],i,j,cx,cy,rh)

                cx_on_img = int(j*grid_dim_x + cx*grid_dim_x)
                cy_on_img = int(i*grid_dim_y + cy*grid_dim_y)
                cv2.circle(img, (cx_on_img, cy_on_img), 3, (0,0,255), 3)                

                h = int(rh*img_height)
                w = int(rw*img_width)

                cv2.rectangle(img, (int(cx_on_img-(w/2)), int(cy_on_img-(h/2))), 
                                    (int(cx_on_img+(w/2)), int(cy_on_img+(h/2))), (0,0,255), 3)

                # Unnormalize
                distance = distance*20.
                yaw_relative = (yaw_relative*np.pi)-np.pi                    

                yaw_relative = yaw_relative/np.pi*180.

                rel_text = "{:3.2f}".format(distance)+" meters"
                yaw_text = "{:3.2f}".format(yaw_relative)+" degree" 

                cv2.putText(img, rel_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, lineType=cv2.LINE_AA)                      
                cv2.putText(img, yaw_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, lineType=cv2.LINE_AA)                      

                bbox_results.append((cx_on_img,cy_on_img,abs(float(distance)),yaw_relative))
    #cv2.imwrite("display.png", img)
    if ret:
        return img, bbox_results


def display_target_woWH(M, source_img, output_shape, threshold, ret=False):
    img = source_img.copy()
    img_height,img_width = img.shape[:2]

    bbox_results = []
    nrow, ncol = output_shape[0], output_shape[1]
    grid_dim_x = img_width/ncol
    grid_dim_y = img_height/nrow


    for i in range(3):
        for j in range(nrow):
            #cv2.line(img, (int(j*grid_dim_x), 0), (int(j*grid_dim_x), img_height), (0,255,0), 1)
            #cv2.line(img, (0, int(i*grid_dim_y)), (img_width, int(i*grid_dim_y)), (0,255,0), 1)

            if M[i,j,0] > threshold:
                cx, cy, distance, yaw_relative = M[i,j,1:]

                #print(M[i,j,0],i,j,cx,cy,rh)

                cx_on_img = int(j*grid_dim_x + cx*grid_dim_x)
                cy_on_img = int(i*grid_dim_y + cy*grid_dim_y)
                cv2.circle(img, (cx_on_img, cy_on_img), 3, (0,0,1), 3)                

                #h = int(rh*img_height)
                #w = int(rw*img_width)

                #cv2.rectangle(img, (int(cx_on_img-(w/2)), int(cy_on_img-(h/2))), 
                #                    (int(cx_on_img+(w/2)), int(cy_on_img+(h/2))), (0,0,255), 3)

                # Unnormalize
                #distance = distance*20.
                #yaw_relative = yaw_relative*np.pi
                #yaw_relative = (yaw_relative+np.pi)/(np.pi)                    

                #yaw_relative = yaw_relative/np.pi*180.

                #rel_text = "{:3.2f}".format(distance)+" meters"
                #yaw_text = "{:3.2f}".format(yaw_relative)+" degree" 

                #cv2.putText(img, rel_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, lineType=cv2.LINE_AA)                      
                #cv2.putText(img, yaw_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, lineType=cv2.LINE_AA)                      

                bbox_results.append((cx_on_img,cy_on_img,abs(float(distance)),yaw_relative))
                
    #cv2.imwrite("display.png", img)
    if ret:
        return img, bbox_results
