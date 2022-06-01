"""
This should be a normal deep learning training/testing script.
"""
import numpy as np
import os.path as osp
import os
import pprint
import cv2

from Utils.ConfigParser import loadConfig
from Agent import Agent
from Loader import TestDatasetLoader

pp = pprint.PrettyPrinter(indent=4)

def train(OmrAgent):
    # ckpts <- agent.train(). ckpts are just paths to saved ckpt

    # write ckpts. You do nothing here
    return

def evaluate(agent):
    testset_name_list = agent.getTestsetNameList()

    for idx_img, pred_img_list in enumerate(agent.evaluate()):
        for idx_layer, pred_img in enumerate(pred_img_list):
            filename = f"./Results/{testset_name_list[idx_img]}_{idx_layer}.png"
            print (f"Write to {filename}")
            cv2.imwrite(filename, pred_img)

    # Do silly masking stuff :<
    return

if __name__ == "__main__":
    # Load configuration
    yaml_path = "./Utils/config.yaml"
    cfg = loadConfig(yaml_path)
    pp.pprint(cfg)

    # Dataset Loader
    test_dataset = TestDatasetLoader(cfg.testset)

    # Get Model

    # Create an OMR agent and inject dataset/model
    agent = Agent(train_dataset=None, test_dataset=test_dataset, cfg=cfg)

    # ckpts <- agent.train()
    evaluate(agent)
    print ("pass")
