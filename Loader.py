"""
Should support loading data with generator (what calvo's classifier is using)
and Tfrecord.
"""
import cv2
class TestDatasetLoader():
    """ A simple loader to load testset
    """
    def __init__(self, path_to_test_image_list):
        self.test_image_list = path_to_test_image_list
    
    def __len__(self):
        return len(self.test_image_list)

    def __getitem__(self, idx):
        path = self.test_image_list[idx]
        # load image!
        img = cv2.imread(path, 1)
        return img

if __name__ == "__main__":
    from Utils.ConfigParser import loadConfig
    yaml_path = "./Utils/config.yaml"
    cfg = loadConfig(yaml_path)

    print (cfg)

    testset = TestDatasetLoader(cfg.testset)

    for img in testset:
        print (img.shape)
    print ("Pass")