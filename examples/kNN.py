from easyneuron.data.gen import gen_stairs
from easyneuron.neighbours import KNNClassifier
from easyneuron.metrics import accuracy

X, y = gen_stairs(3, 2, sd=0.1, samples=1000)

model = KNNClassifier()
model.fit(X[:900], y[:900])
preds = model.predict(X[900:])
print(preds)
print(accuracy(preds, y[900:]))
