import os

for i in os.listdir("logs/"):
	os.remove(os.path.join("logs", i))