# original version: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/util/html.py

import pickle

from prediction_loader import *
from html import Method, CopyImages, HTML, writing_table


def read_id_from_file(file_path):
    with open(file_path) as f:
        id_list = f.readlines()
    id_list = [i.strip() for i in id_list]
    return id_list

if __name__ == '__main__':  # we show an example usage here.
    # Methods
    method_list = []
    method_list.append(Method(title="base",
                              subdir="wo_RGF_wo_RR_CGI+IIW",
                              image_loader=CRefNet("experiments/results_NeurIPS2022/CGI+IIW/visualization/test_with_wo_RGF_wo_RR/IIW")))
    method_list.append(Method(title="+RGF",
                              subdir="w_RGF_wo_RR_CGI+IIW",
                              image_loader=CRefNet("experiments/results_NeurIPS2022/CGI+IIW/visualization/test_with_w_RGF_wo_RR/IIW")))
    method_list.append(Method(title="+RR",
                              subdir="wo_RGF_w_RR_CGI+IIW",
                              image_loader=CRefNet("experiments/results_NeurIPS2022/CGI+IIW/visualization/test_with_wo_RGF_w_RR/IIW")))
    method_list.append(Method(title="+RGF +RR (final)",
                              subdir="w_RGF_w_RR_final_CGI+IIW",
                              image_loader=CRefNet("experiments/results_NeurIPS2022/CGI+IIW/visualization/test_with_final_w_RGF_w_RR/IIW")))

    # Input loader
    input_loader = InputLoader("./data/iiw-dataset")

    # Sampled image index
    test_list_file_path = "./iiw_test_img_batch.p"
    images_list = pickle.load(open(test_list_file_path, "rb"))
    index_list = []
    sample_interv = 3
    for i in range(3):
        for j in range(len(images_list[i])):
            if j % sample_interv == 0 or j % 2 == 0:
                id = str(images_list[i][j].split('/')[-1][0:-7])
                index_list.append(id)
    exclude_list = [
        80141, 113958, 89092, 86288, 83086, 113930, 74872, 85322, 69646, 11389, 105065,
        90699, 71954, 89055, 77273, 113806, 66513, 116535,
        64988, 34280, 116303, 114369, 88001, 116625, 91762, 104333, 70179, 56804, 65639, 67662,
        69896, 67912, 82969, 57082, 113601, 115374, 105924, 114704, 77548,
        115560, 23021, 68023,
        78837, 65500, 109934, 65252, 57242, 75235,
        18680, 101315, 16129, 22351  #toilet
    ]  # with people
    for idx in exclude_list:
        if str(idx) in index_list:
            index_list.remove(str(idx))
    # index_list = [
    #     104639, 93581, 34872, 104084, 55692, 89913, 61999, 114093,
    #     60741, 105478, 90494, 34647, 3916, 102274, 104781, 35713, 74900,
    #     116769, 101315, 58077, 103669, 34735, 104417, 64756, 106788, 65156,
    #     103586, 103336, 110701, 94184, 104457, 89851, 63508, 60104, 102052,
    #     83267, 11225, 96885, 67440, 15263, 108505, 104013, 686, 2548, 82208, 58077, 100861,
    #     2355,
    #     4627, 34647,
    #     61999,
    #     82208,
    #     100861, 103649
    # ]

    # Copy images
    html_dir = "experiments/results_NeurIPS2022/CGI+IIW/web"
    image_rel_dir = "./images"
    html_images_dir = os.path.join(html_dir, image_rel_dir)
    c = CopyImages(method_list, index_list, html_images_dir, image_rel_dir, skip_exist=True, compress_quality=100)
    method_list = c.run_copy_method_images(4)
    input_loader = c.copy_images_from_input_loader(input_loader, "input")

    # Writing html table
    html = HTML(html_dir, 'intrinsic_images_html')
    html.add_header('Intrinsic Image Decomposition')
    html.add_text("We recommend viewing this HTML file using the Chrome browser.")
    html.add_text(f"Visual comparison between our CRefNet and previous methods. "
                  f"We show the first of every {sample_interv} consecutive images on the test split file provided by Li and Snavely (2018). "
                  f"For each sample, reflectance (top) and shading (below) images are presented.")
    writing_table(html, input_loader, index_list, method_list)

