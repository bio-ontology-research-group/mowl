import mowl
mowl.init_jvm("10g")

from mowl.visualization import TSNE
from numpy import array, random

data = random.rand(100, 100)
names = [f"name_{i}" for i in range(100)]
classes = [1,2,3]
labels = [random.choice(classes) for _ in range(100)]

name_to_embedding = dict(zip(names, data))
name_to_label = dict(zip(names, labels))

tsne = TSNE(name_to_embedding, name_to_label)
tsne.generate_points(250, workers=4)

tsne.show()
