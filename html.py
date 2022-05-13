# original version: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/util/html.py

from collections import namedtuple
import os
import pickle
import shutil
import multiprocessing

import dominate
from dominate import tags
from dominate.tags import meta, h3, tr, td, a, img, br, col
import cv2

from prediction_loader import *


Method = namedtuple("Method", ["title", "subdir", "image_loader"])


class HTML:
    """This HTML class allows us to save images and write texts into a single HTML file.

     It consists of functions such as <add_header> (add a text header to the HTML file),
     <add_images> (add a row of images to the HTML file), and <save> (save the HTML to the disk).
     It is based on Python library 'dominate', a Python library for creating and manipulating HTML documents using a DOM API.
    """

    def __init__(self, web_dir, title, refresh=0):
        """Initialize the HTML classes

        Parameters:
            web_dir (str) -- a directory that stores the webpage. HTML file will be created at <web_dir>/index.html; images will be saved at <web_dir/images/
            title (str)   -- the webpage name
            refresh (int) -- how often the website refresh itself; if 0; no refreshing
        """
        self.title = title
        self.web_dir = web_dir
        # self.img_dir = os.path.join(self.web_dir, 'images')
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        # if not os.path.exists(self.img_dir):
        #     os.makedirs(self.img_dir)

        self.doc = dominate.document(title=title)
        if refresh > 0:
            with self.doc.head:
                meta(http_equiv="refresh", content=str(refresh))

    def create_table(self, border, widths):
        t = tags.table(border=border, style="table-layout: fixed;")  # Insert a table
        self.doc.add(t)
        for width in widths:
            col(width=f"{width}px")
        return t

    # def get_image_dir(self):
    #     """Return the directory that stores images"""
    #     return self.img_dir

    def add_header(self, text):
        """Insert a header to the HTML file

        Parameters:
            text (str) -- the header text
        """
        with self.doc:
            h3(text)

    def add_text(self, text):
        with self.doc:
            tags.p(text)

    def add_images(self, t, ims, txts, links, widths, hw_ratio):
        """add images to the HTML file

        Parameters:
            ims (str list)   -- a list of image paths
            txts (str list)  -- a list of image names shown on the website
            links (str list) --  a list of hyperref links; when you click an image, it will redirect you to a new page
        """
        with t:
            with tr():
                for im, txt, link, width in zip(ims, txts, links, widths):
                    with td(style="word-wrap: break-word;", align="center", valign="top"):
                        if txt is not None:
                            tags.p(txt)
                            # br()
                        if im is not None:
                            with a(href=link):
                                img(width=width, height=int(width*hw_ratio), src=im)

    def save(self):
        """save the current content to the HMTL file"""
        html_file = '%s/index.html' % self.web_dir
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()


class CopyImages(object):
    def __init__(self, method_list, index_list, dst_dir, rel_dir, skip_exist=False, compress_quality=100):
        self.method_list = method_list
        self.index_list = index_list
        self.dst_dir = dst_dir
        self.rel_dir = rel_dir
        self.dst_postfix = "jpg"
        self.compress_quality = compress_quality
        self.skip_exist = skip_exist

    def _copy_single_image(self, src_img_path, subdir, postfix):
        assert postfix in ["jpg", "jpeg"]
        file, o_postfix = src_img_path.split('/')[-1].split('.')
        file_name = f"{file}.{postfix}"
        out_dir = os.path.join(self.dst_dir, subdir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_img_path = os.path.join(out_dir, file_name)
        if not self.skip_exist or not os.path.exists(out_img_path):
            if o_postfix == postfix:
                shutil.copy(src_img_path, out_img_path)
            else:
                src_img = cv2.imread(src_img_path, cv2.IMREAD_UNCHANGED)
                cv2.imwrite(out_img_path, src_img, [cv2.IMWRITE_JPEG_QUALITY, self.compress_quality])
        # img = cv2.imread(out_img_path)
        # h, w = img.shape[:2]
        # return os.path.join(os.path.join(subdir, file)), h / w

    def copy_images_from_method(self, m_idx):
        m = self.method_list[m_idx]
        for id in self.index_list:
            r_path, s_path = m.image_loader.get_pred_rs_img_path(id)
            for p in [r_path, s_path]:
                self._copy_single_image(p, m.subdir, self.dst_postfix)
        m.image_loader.set_img_dir(os.path.join(self.rel_dir, m.subdir), self.dst_postfix)
        return m_idx, m
        # print("copy", m.image_loader.image_dir, self.method_list[m_idx].image_loader.image_dir)

    def copy_images_from_input_loader(self, loader: InputLoader, subdir):
        for id in self.index_list:
            img_path = loader.get_input_img_path(id)
            self._copy_single_image(img_path, subdir, self.dst_postfix)
        loader.set_img_dir(os.path.join(self.rel_dir, subdir), self.dst_postfix)
        return loader

    def run_copy_method_images(self, num_workers):
        print(f"Multiple threads: {num_workers}")
        with multiprocessing.Pool(processes=num_workers) as pool:
            cnt = 0
            for index, m in pool.imap_unordered(self.copy_images_from_method, range(len(self.method_list))):
                self.method_list[index] = m
                cnt += 1
                print(f"Finish: {cnt}/{len(self.method_list)}")
        # for m in self.method_list:
        #     print("run", m.image_loader.image_dir)
        return self.method_list


def writing_table(html, input_loader, index_list, method_list):
    print(f"Total samples: {len(index_list)}")

    text_width = 50
    img_width = 200

    # table header
    header = ["Index", "Input"]
    for m in method_list:
        header.append(m.title)
    column_widths = [text_width, img_width] + [img_width] * len(method_list)
    # create a table
    tab = html.create_table(0, column_widths)

    # add image
    for i in range(len(index_list)):
        # print header
        if i % 4 == 0:
            html.add_images(tab, [None]*len(header), header, [None]*len(header), column_widths, 1)
        id = index_list[i]
        # input image + reflectance
        input_path = input_loader.get_input_img_path(id)
        full_input_path = os.path.join(html.web_dir, input_path)
        input_img = cv2.imread(full_input_path, cv2.IMREAD_UNCHANGED)
        print(full_input_path)
        hw_ratio = input_img.shape[0] / input_img.shape[1]
        ims = [None, input_path]
        txts = [id, None]
        for m in method_list:
            img_path = m.image_loader.get_pred_rs_img_path(id)[0]
            ims.append(img_path)
            txts.append(None)
        html.add_images(tab, ims, txts, ims, column_widths, hw_ratio)
        # shading
        ims = [None, None]
        txts = [None, None]
        for m in method_list:
            img_path = m.image_loader.get_pred_rs_img_path(id)[1]
            ims.append(img_path)
            txts.append(None)
        html.add_images(tab, ims, txts, ims, column_widths, hw_ratio)
        # print(f"{i}/{len(index_list)}")
    html.save()
    print("Finish writing table.")


if __name__ == '__main__':  # we show an example usage here.
    # Methods
    method_list = []
    method_list.append(Method(title="CRefNet (ours)",
                              subdir="crefnet",
                              image_loader=CRefNet("./CRefNet/final_CGI+IIW")))
    method_list.append(Method(title="Wang & Lu 2019",
                              subdir="wang_lu_2019",
                              image_loader=Wang_2019_Discriminative_Loader("./Wang_2019_Single Image Intrinsic Decomposition with Discriminative Feature Encoding")))
    method_list.append(Method(title="Luo et al. 2020",
                              subdir="luo_2020",
                              image_loader=Luo_2020_NIID_Net_Loader("./Luo_2020_NIID-Net")))
    method_list.append(Method(title="Li & Snavely 2018",
                              subdir="li_snavely_2018",
                              image_loader=Li_2018_CGI_Loader("./Li_2018_CGIntrinsics/CGI+IIW+SAW/cgi_iiw")))
    method_list.append(Method(title="Bi et al. 2015",
                              subdir="bi_2015",
                              image_loader=Bi_2015_L1smoothing_Loader("./Bi_2015_An L1 image transform for edge-preserving smoothing and scene-level intrinsic decomposition")))
    method_list.append(Method(title="Bell et al. 2014",
                              subdir="bell_2014",
                              image_loader=General_Loader("./saw_decomps/methods/bell2014_densecrf-1141")))
    method_list.append(Method(title="Garces et al. 2012",
                              subdir="garces_2012",
                              image_loader=General_Loader("./saw_decomps/methods/garces2012_clustering-1221")))
    method_list.append(Method(title="Grosse et al. 2009",
                              subdir="grosse_2009",
                              image_loader=General_Loader("./saw_decomps/methods/grosse2009_color_retinex-633")))
    method_list.append(Method(title="Zhao et al. 2012",
                              subdir="zhao_2012",
                              image_loader=General_Loader("./saw_decomps/methods/zhao2012_nonlocal-709")))
    method_list.append(Method(title="Zhou et al. 2015",
                              subdir="zhou_2015",
                              image_loader=General_Loader("./saw_decomps/methods/zhou2015_reflprior-1281")))
    # Input loader
    input_loader = InputLoader("./data/iiw-dataset")

    # Sampled image index
    test_list_file_path = "./iiw_test_img_batch.p"
    images_list = pickle.load(open(test_list_file_path, "rb"))
    index_list = []
    sample_interv = 5
    for i in range(3):
        for j in range(0, len(images_list[i]), sample_interv):
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

    # Copy images
    # html_dir = "./web"
    html_dir = "experiments/faster/refine_on_the_IIW/web"
    image_rel_dir = "./images"
    html_images_dir = os.path.join(html_dir, image_rel_dir)
    c = CopyImages(method_list, index_list, html_images_dir, image_rel_dir, skip_exist=True, compress_quality=90)
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

