import cv2
import numpy as np
import tensorflow as tf
import utils_io
import os

from directed_blur import KernelBuilder, DirectedBlurFilter
from pencil_filter import PencilFilter
from logger import Logger

from losses import multi_grid_loss
from metrics import gate_center_mae_error,distance_mae_error,orientation_mae_error

from networks import PencilNet
class PencilNetInference:
    '''
    filters_to_apply - list of filters which will be applied to the frame before inference. The order is preserved.
    '''
    def __init__(self, h=160, w=120, filters_to_apply = []):
        self.shape = (h,w)
        self.filters_to_apply = filters_to_apply
        self.pencil_filter = PencilFilter()
        self.threshold = 0.5
        self.checkpoint_epoch = 789
        
        self.initi_tf('PencilNet/Public_datasets/Trained_models/pencilnet-2022-01-22-13-45')


    def initi_tf(self, model_path):
        print("[*] PencilNet: Model is being loaded from {}...".format(model_path))

        self.logger = Logger()
        self.logger.load(model_path)

        config = tf.compat.v1.ConfigProto(
            device_count={'GPU': 1},
            intra_op_parallelism_threads=2,
            allow_soft_placement=True
        )

        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.6
        tf.compat.v1.enable_eager_execution()
        self.session = tf.compat.v1.Session(config=config)
        checkpoints = self.logger.list_checkpoints()
        self.path_to_model = checkpoints[self.checkpoint_epoch]

        print("[*] PencilNet: The model is loaded!")


    def run(self, bgr_img):
        # Preprocess data
        img = cv2.resize(bgr_img, (self.shape[0],self.shape[1])) # Resize image for the network.

        # Apply filters
        for pre_filter in self.filters_to_apply:
            img = pre_filter.apply(img)

        img = np.reshape(img, (1,120,160,3)) # 1 more channel is added to form a batch.
        img = img.astype(np.float32)/255. # Convert image type int8 to float32

        predictions = None
        # Run the network for inference!
        with self.session.as_default():
            with self.session.graph.as_default():
                json_file = open('PencilNet/Public_datasets/Trained_models/pencilnet-2022-01-22-13-45/model.json','r')
                loaded_model_json = json_file.read()
                json_file.close()
                loaded_model = tf.keras.models.model_from_json(loaded_model_json)
                loaded_model.load_weights('PencilNet/Public_datasets/Trained_models/pencilnet-2022-01-22-13-45/weights/' + self.path_to_model)
                loaded_model.compile(loss=multi_grid_loss,optimizer='adam',metrics=[gate_center_mae_error,distance_mae_error,orientation_mae_error])
                
                # perform the prediction
                predictions = loaded_model.predict(img)
        # -------------------------


        # -------------------------
        # Publish the cx, cy, distance, yaw_relative
        # -------------------------
        # Get results, pred_imd is for debugging, bbox is [(cx_on_img,cy_on_img,distance,yaw_relative)]
        pred_img, bboxes = utils_io.display_target_woWH(np.float32(predictions[0]), 
                                                            img[0], 
                                                            self.logger.config['output_shape'], 
                                                            self.threshold, ret=True)

        return pred_img, bboxes

# Test inference
if __name__ == '__main__':
    kernel_builder = KernelBuilder()
    # With kernel size you can control the simulated velocity of the drone => greater the size, the more the motion.
    input_images = []
    predicted_images = []
    
    for i in range(10):
        kernel_size = 5
        kernel = kernel_builder.custom_rotated_horizontal_kernel(kernel_size, random_angle=True)
        directed_blur_filter = DirectedBlurFilter(kernel)

        pencilFilter = PencilFilter()

        #test_img_path = os.path.join('PencilNet', 'Public_datasets', 'Test data', 'original','rgb_real_N_40', 'images','000735.png')
        #test_img_path = os.path.join('PencilNet', 'Public_datasets', 'Test data', 'original','rgb_real_N_20', 'images','000370.png')
        test_img_path = os.path.join('PencilNet', 'Public_datasets', 'Test data', 'original','rgb_real_N_10', 'images','000000.png')
        test_img_bgr = cv2.imread(test_img_path)

        filters = [directed_blur_filter,pencilFilter]
        
        img = test_img_bgr.copy()
        # Apply filters
        for pre_filter in filters:
            img = pre_filter.apply(img)
        
        input_images.append(img)
        inference = PencilNetInference(filters_to_apply=filters)

        pred_img, bboxes = inference.run(test_img_bgr)
        
        predicted_images.append(pred_img)

        center_x = [bbox[0] for bbox in bboxes]
        center_y = [bbox[1] for bbox in bboxes]
        distance = [bbox[2] for bbox in bboxes]
        yaw_relative = [bbox[3] for bbox in bboxes]

    cv2.imshow('Original w/o filters', test_img_bgr)
    cv2.imshow('Original', np.concatenate(input_images, axis=1))
    cv2.imshow('Predicted', np.concatenate(predicted_images, axis=1))

    cv2.waitKey(0)
    cv2.destroyAllWindows()