import zipfile


def unzip(path_to_file, extract_directory):
    zip_file = zipfile.ZipFile(path_to_file, 'r')
    zip_file.extractall(extract_directory)
    zip_file.close()
