import os


def clear_dir(dir):
    for i in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, i)):
            os.remove(os.path.join(dir, i))
        elif os.path.isdir(os.path.join(dir, i)):
            clear_dir(os.path.join(dir, i))


clear_dir("logs")
