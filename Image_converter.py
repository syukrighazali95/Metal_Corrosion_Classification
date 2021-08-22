from PIL import Image
import os

jfif_list = []
dataset_dir = "D:/git/Metal_Corrosion_Classification/Datasets/Google Image/No rust/Imageye - bridge metal - Google Search"

def convert_image_to_jpg(dataset_directory, class_name="", sub_directory=""):
    count = 100000
    for dirpath, dirname, filenames in os.walk(dataset_directory + sub_directory):
        if filenames:
            # print(filenames)
            for file in filenames:
                try:
                    if "jfif" in file:
                        print(file)
                        filename = class_name + str(count) + ".jpg"
                        print(filename)
                        os.rename(f"{dataset_directory}/{sub_directory}/{file}", f"{dataset_directory}/{sub_directory}/{filename}")
                        count += 1
                    elif "png" in file:
                        print(file)
                        print(dataset_directory + "/" + sub_directory + "/" + file)
                        im = Image.open(dataset_directory + "/" + sub_directory + "/" + file)
                        rgb_im = im.convert("RGB")
                        filename = str(count)
                        rgb_im.save(dataset_directory + "/" + sub_directory + "/" + filename + ".jpg")
                        count += 1
                except:
                    print("Other type of images are not supported")

def rename_datasets(dataset_directory, class_name="", sub_directory=""):
    count = 1
    for dirpath, dirname, filenames in os.walk(dataset_directory + sub_directory):
        for file in filenames:
            try:
                filename = class_name + str(count) + ".jpg"
                os.rename(f"{dataset_directory}/{sub_directory}/{file}", f"{dataset_directory}/{sub_directory}/{filename}")
                count += 1
            except:
                print("Something is wrong with renaming the files")


def main():
    convert_image_to_jpg(dataset_dir)
    print("Images have been converted to jpg successfully")
    rename_datasets(dataset_dir, class_name="bridge_metal")
    print("Images have been renamed successfully")

if __name__ == "__main__":
    main()

