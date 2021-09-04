import os


def search_dir(path, extension):
    result = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                result.append(os.path.join(root, file))
    return result
