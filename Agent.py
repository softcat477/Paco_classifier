import numpy as np
import os
import os.path as osp
import cv2

import recognition_engine as recognition

class Agent():
    """ Agent is used by Local and Rodan.
    do not put any code specifically handling Rodan job or local GPU training inside!!!
    """
    def __init__(self, train_dataset, test_dataset, cfg):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.cfg = cfg
        # Load ckpt 
        self.ckpt_paths = self.cfg.path_ckpt
        self.patch_height = self.cfg.patch_height
        self.patch_width = self.cfg.patch_width

    def train(self):
        raise NotImplementedError("")

    def evaluate(self) -> list:
        """ The evaluation function shared by Local and Rodan.
        Evaluate test images inside the test set loader. Return layers 
        of each image with yield.

        Returns:
            list
        """
        # TODO: sanity check to make sure there's a test_dataset
        model_paths = self.ckpt_paths
        print ("Load ckpts from ", model_paths)

        # --- RODAN start ---
        mode = 'logical'
        for idx, test_img in enumerate(self.test_dataset):

            print (f"Predicting {idx} model")
            analyses = recognition.process_image_msae(test_img, model_paths, self.patch_height, self.patch_width, mode=mode)

            # three models
            ret_list = []
            for id_label, _ in enumerate(model_paths):
                if mode == 'masks':
                    mask = ((analyses[id_label] > (threshold / 100.0)) * 255).astype('uint8')
                elif mode == 'logical':
                    label_range = np.array(id_label, dtype=np.uint8)
                    mask = cv2.inRange(analyses, label_range, label_range)
    
                original_masked = cv2.bitwise_and(test_img, test_img, mask = mask)
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
        # --- RODAN end ---

    def getTestsetNameList(self):
        """This function is used by local GPU evaluationg task.
        We need the test image name to write results.
        """
        return [p.split("/")[-1].split(".")[0] for p in self.cfg.testset]