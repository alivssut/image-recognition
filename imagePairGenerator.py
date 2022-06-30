import os
import cv2
import numpy as np


class ImagePairGenerator:
    def __init__(self, path, width, height, channel):
        self.path = path
        self.width = width
        self.height = height
        self.channel = channel

    def generate_images_pair(self):
        images_pair = []
        images_label = []
        c = 0
        for person in os.listdir(self.path):
            print(person)
            d = 0
            for personImage in os.listdir(f"{self.path}\\{person}"):
                image1 = cv2.imread(f"{self.path}\\{person}\\{personImage}", cv2.IMREAD_COLOR)
                image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
                image1 = cv2.resize(image1, (self.width, self.height))
                for secondPerson in os.listdir(self.path):
                    for secondPersonImage in os.listdir(f"{self.path}\\{secondPerson}"):
                        image2 = cv2.imread(f"{self.path}\\{secondPerson}\\{secondPersonImage}", cv2.IMREAD_COLOR)
                        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
                        image2 = cv2.resize(image2, (self.width, self.height))
                        images_pair.append((np.array(image1), np.array(image2)))
                        if person == secondPerson:
                            images_label.append(1)
                        else:
                            images_label.append(0)

                    d += 1
                    if d > 3:
                        break
            c += 1
            if c > 10:
                break
        return np.array(images_pair) , np.array(images_label)