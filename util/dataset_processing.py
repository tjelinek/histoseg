import bz2
import urllib.request
import requests
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from bs4 import BeautifulSoup
from cfg import *
from PIL import Image
from pathlib import Path

from image.preprocessor import StainNormalizer


def generate_offline_augmented_data(directory: Path) -> None:
    """
    Generates augmented data offline.
    """
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=(0.7, 1.3),
        zoom_range=(0.7, 1.3),
        fill_mode="constant",
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=None,
        cval=0
    )

    augmentation_factor = 10
    counter = 0
    flow_gen = datagen.flow_from_directory(str(directory),
                                           class_mode='binary',
                                           save_prefix='N',
                                           save_format='tif',
                                           batch_size=augmentation_factor,
                                           shuffle=False)

    for inputs, outputs in flow_gen:
        for i in range(len(inputs)):
            dir_with_class = str(directory) + '_augmented/' + str(int(outputs[i]))
            Path(dir_with_class).mkdir(exist_ok=True)
            Image.fromarray(inputs[i].astype('uint8')).save(
                str(dir_with_class) + '/' + str(counter) + '.tif')
            counter += 1
            print(counter)


def apply_img_color_normalization(source_train: Path, source_valid: Path,
                                  destination_train: Path, destination_valid: Path) -> None:
    """

    @param source_train:
    @param source_valid:
    @param destination_train:
    @param destination_valid:
    @return:
    """
    data_raw = [source_train, source_valid]
    data_processed = [destination_train, destination_valid]

    for i in range(len(data_raw)):
        folder = data_raw[i]
        folder_preprocessed = data_processed[i]
        for root, _, files in os.walk(folder, topdown=False):
            save_dir = Path(folder_preprocessed / Path(Path(root).name))
            if len(files) > 0:
                save_dir.mkdir(parents=True, exist_ok=True)

            if Path(root).name == 'BACK':
                continue
            for file in files:
                normalizer = StainNormalizer()
                try:  # TODO: do some better handling of this error
                    img_preprocessed = normalizer.normalize_file(Path(root) / Path(file))
                except:
                    print("Normalizing of image failed")
                    continue
                img_preprocessed = Image.fromarray(img_preprocessed.astype('uint8'))
                img_preprocessed.save(save_dir / Path(file))


def apply_img_preprocessing() -> None:
    """

    @return:
    """
    data = ['data/data_tiles_train', 'data/data_tiles_valid']
    data_preproc = ['data/data_tiles_train_color_sep', 'data/data_tiles_valid_color_sep']

    for i in range(len(data)):
        folder = data[i]
        folder_preproc = data_preproc[i]
        for root, _, files in os.walk(folder, topdown=False):
            save_dir = Path(folder_preproc / Path(Path(root).name))
            if len(files) > 0:
                save_dir.mkdir(parents=True, exist_ok=True)

            for file in files:
                img = Image.open(Path(root) / Path(file))
                # img_preprocessed = Image.fromarray(separate_colors(img).astype('uint8'))
                # TODO implement this function
                img_preprocessed = img
                img_preprocessed.save(save_dir / Path(file))


def load_webpage(url: str) -> str:
    """

    @param url: Link to a web page
    @return:  Source of the web page as a string
    """
    fp = urllib.request.urlopen(url)
    webpage = fp.read()
    web_content: str = webpage.decode("utf8")
    fp.close()

    return web_content


def decompress_bz2(compressed):
    """
    Decompresses .bz2 file, as the annotations may be compressed
    @param compressed:
    @return: decompressed .bz2 file
    """
    return bz2.decompress(compressed)


def fetch_files_prof_feit(url: str, destination_folder: Path) -> None:
    """
    Formerly written for fetching files from https://atlases.muni.cz/atlases/colonannot.html
    It assumes the web page contains only links to files to be downloaded and nothing more.
    @param url: Url to the webpage containing links to the files
    @param destination_folder: Destination folder to save the fetched files. Structure of the links is retained.
    """
    url_parent, _ = os.path.split(url)
    url_parent += '/'

    web_content = load_webpage(url)

    file_paths = []
    soup = BeautifulSoup(web_content, features="html.parser")

    # Get all links
    for file_path in soup.find_all('a'):
        file_paths.append(file_path.get('href'))

    i: int = 0
    for file_path in file_paths:
        i += 1
        print("Processing file " + str(i) + " out of " + str(len(file_paths)))

        full_link = url_parent + file_path
        dest_path = destination_folder / file_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if dest_path.is_file():
            continue  # file has already been downloaded
        r = requests.get(full_link, allow_redirects=True)

        open(dest_path, 'wb').write(r.content)
