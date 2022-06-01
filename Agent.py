import numpy as np
import os
import os.path as osp

from . import recognition_engine as recognition

class Agent():
    def __init__(self, train_dataset, test_dataset, model_path):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        #self.model_list = model_list # list[ckpt1, ckpt2, ckpt3]
        self.model_path = model_path

    def train(self):
        raise NotImplementedError("")

    def evaluate(self) -> list:
        # TODO: sanity check to make sure there's a test_dataset
        for idx, test_img in enumerate(self.test_dataset):

            # Copy code!
            # --- Load image ---
            #image_filepath = inputs['Image'][idx]['resource_path']
            #image = cv2.imread(image_filepath, 1)
            # --- Load image END ---
            analyses = recognition.process_image_msae(test_img, model_paths, height, width, mode = mode)

            # three models
            ret_list = []
            for id_label, _ in enumerate(model_paths):
                if mode == 'masks':
                    mask = ((analyses[id_label] > (threshold / 100.0)) * 255).astype('uint8')
                elif mode == 'logical':
                    label_range = np.array(id_label, dtype=np.uint8)
                    mask = cv2.inRange(analyses, label_range, label_range)
    
                original_masked = cv2.bitwise_and(image, image, mask = mask)
                original_masked[mask == 0] = (255, 255, 255)

                # Alpha = 0 when background
                alpha_channel = np.ones(mask.shape, dtype=mask.dtype) * 255
                alpha_channel[mask == 0] = 0
                b_channel, g_channel, r_channel = cv2.split(original_masked)
                original_masked_alpha = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

                # This is the rodan version
                #if switch[id_label] in outputs:
                #    cv2.imwrite(outputs[switch[id_label]][idx]['resource_path']+'.png', original_masked_alpha)
                #    os.rename(outputs[switch[id_label]][idx]['resource_path']+'.png', outputs[switch[id_label]][idx]['resource_path'])
                ret_list.append(original_masked_alpha)
            yield ret_list