import os


def list_all_files_with_extension(files: list,
                                  extension: str,
                                  shallow: bool = False) -> list:
    result = list()

    for file in files:
        if os.path.isfile(file) and file.endswith(f'.{extension}'):
            result.append(file)
        elif os.path.isdir(file) and not shallow:
            sub_files = [os.path.join(file, f) for f in os.listdir(file)]
            result.extend(list_all_files_with_extension(files=sub_files,
                                                        extension=extension,
                                                        shallow=shallow))

    return result
