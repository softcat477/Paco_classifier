import numpy as np
import os.path as osp
import os
import pprint
import cv2

from Utils.ConfigParser import loadConfig
from Agent import Agent

pp = pprint.PrettyPrinter(indent=4)

def train(OmrAgent):
    # ckpts <- agent.train(). ckpts are just paths to saved ckpt

    # write ckpts. You do nothing here
    return

def evaluate(agent):
    for idx, pred_img_list in enumerate(agent.evaluate()):
        for pred_img in pred_img_list:
            cv2.imwrite(f"./{idx}.png", pred_img)

    # Do silly masking stuff :<
    return

if __name__ == "__main__":
    # Load configuration
    yaml_path = "./Utils/config.yaml"
    cfg = loadConfig(yaml_path)
    pp.pprint(cfg)

    # Dataset Loader

    # Get Model

    # Create an OMR agent and inject dataset/model

    # ckpts <- agent.train()
    agent.evaluate()
