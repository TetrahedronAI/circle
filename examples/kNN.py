# Copyright 2022 Neuron-AI GitHub Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from pprint import pformat

from easyneuron.data.gen import make_stairs
from easyneuron.metrics import accuracy
from easyneuron.neighbours import KNNClassifier

print("Generating clustered data...".ljust(100), end="\r")
X, y = make_stairs(3, 2, sd=0.1, samples=1000)
X_train, y_train = X[:900], y[:900]
X_test, y_test = X[900:], y[900:]

print("Training kNN model...".ljust(100), end="\r")
model = KNNClassifier()
model.fit(X_train, y_train)

print("Predicting with kNN model...".ljust(100), end="\r")
preds = model.predict(X_test)

print("Predictions: ", pformat(preds, indent=4, width=10))
print(f"Model Accuracy: {accuracy(preds, y_test)}")
