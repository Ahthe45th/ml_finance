import os


def search_dir(path, extension):
    """
    Lists all files matching a specified extension in a given path.

    Args:
        path: A string containing a path.
        extension: A string containing an extension name without a dot.

    Return:
        An ordered list of strings containing matched files.

    Example:

        >>> search_dir('/home/user/documents', 'pdf')
        ['/home/user/documents/notes.pdf',
        '/home/user/documents/novel.pdf']
    """
    result = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                result.append(os.path.join(root, file))
    return result
