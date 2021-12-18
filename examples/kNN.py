from easyneuron.neighbours import KNNClassifier
from easyneuron.data import gen_stairs

X, y = gen_stairs()

model = KNNClassifier()
model.fit(X, y)
model.predict(X)
