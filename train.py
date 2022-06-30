from sklearn.utils import shuffle
from siameseNetwork import SiameseNetwork
from imagePairGenerator import ImagePairGenerator
import matplotlib.pyplot as plt
import seaborn as sns

imagesPair = ImagePairGenerator("E:\Data\lfw_funneled", 64, 64, 1)
data, labels = imagesPair.generate_images_pair()
plt.hist(labels)
plt.show()

siameseNetwork = SiameseNetwork(64, 64, 1, (64, 64, 1))
siameseNetwork.siamese_model.fit([data[:, 0], data[:, 1]], labels, epochs=5, validation_split=0.2, batch_size=16)
siameseNetwork.save("image_recognition_weights.h5")
