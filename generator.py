


path_to_test_image_list = ["..../.png", ".../.png"]
def generator(path_to_test_image_list:list):
    for path in path_to_test_image_list:
        # load image!
        img = cv2.read(......)
        yield img