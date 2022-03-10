import numpy as np
import pickle
import sys
import json
import h5py
import argparse
import os
from collections import namedtuple
from skimage.transform import resize
from skimage import io

from average_meter import AverageMeter
from prediction_loader import *
import metrics_iiw


class WHDRAverageMeter(object):
    Result = namedtuple("Result", ["WHDR", "WHDR_eq", "WHDR_ineq"])

    def __init__(self, name: str):
        self.name = name
        self.whdr_meter = AverageMeter("WHDR")
        self.whdr_eq_meter = AverageMeter("WHDR_eq")
        self.whdr_ineq_meter = AverageMeter("WHDR_ineq")

    def update(self, whdr, whdr_eq, whdr_ineq, count, count_eq, count_ineq):
        self.whdr_meter.update(whdr, count)
        self.whdr_eq_meter.update(whdr_eq, count_eq)
        self.whdr_ineq_meter.update(whdr_ineq, count_ineq)

    def get_results(self):
        return self.Result(WHDR=self.whdr_meter.avg, WHDR_eq=self.whdr_eq_meter.avg, WHDR_ineq=self.whdr_ineq_meter.avg)

    def __str__(self):
        return f"WHDR {self.whdr_meter.avg: .6f}, " \
               f"WHDR_eq {self.whdr_eq_meter.avg: .6f}, " \
               f"WHDR_ineq {self.whdr_ineq_meter.avg: .6f}"


def evaluate_predictions(file_list_path, iiw_dir, eq_delta, loader: PredictionLoader):
    images_list = pickle.load(open(file_list_path, "rb"))

    # whdr_rgb_meter = WHDRAverageMeter("whdr_rgb")
    whdr_srgb_meter = WHDRAverageMeter("whdr_srgb")
    # count = 0.0
    # whdr_sum =0.0

    for j in range(0, 3):
        img_list = images_list[j]
        for i in range(len(img_list)):
            id = str(img_list[i].split('/')[-1][0:-7])
            img_path = os.path.join(iiw_dir, "data", f"{id}.png")
            judgement_path = os.path.join(iiw_dir, "data", f"{id}.json")
            judgements = json.load(open(judgement_path))

            img = np.float32(io.imread(img_path)) / 255.0
            o_h, o_w = img.shape[0], img.shape[1]

            pred_r = loader.get_pred_r(id, "srgb")
            pred_r = resize(pred_r, (o_h, o_w),
                            order=1, preserve_range=True, anti_aliasing=True)

            (whdr, _), (whdr_eq, valid_eq), (whdr_ineq, valid_ineq) =\
                metrics_iiw.compute_whdr(pred_r, judgements, eq_delta)
            # whdr_sum += whdr
            # count+=1.0
            # whdr_mean =whdr_sum/count
            # print(whdr_mean)
            whdr_srgb_meter.update(whdr,    whdr_eq if valid_eq else 0, whdr_ineq if valid_ineq else 0,
                                   1,       1 if valid_eq else 0,       1 if valid_ineq else 0)
            if i%100 == 0:
                print(f"Evaluate {j}-{i}: \n"
                      # f"\tWHDR(rgb) {whdr_rgb_meter} \n"
                      f"\tWHDR(srgb) {whdr_srgb_meter}")

    # whdr_mean = whdr_sum/count
    # print('whdr_mean %f', whdr_mean)
    print(f"\nWHDR(srgb) {whdr_srgb_meter}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        default="./iiw_test_img_batch.p",
        metavar="FILE",
        help="Path to test list file",
        type=str,
    )
    parser.add_argument(
        "--iiwdir",
        default="./data/iiw-dataset",
        metavar="FILE",
        help="Path to test list file",
        type=str,
    )
    parser.add_argument(
        "--method",
        default="Li_2018_full",
        metavar="FILE",
        help="Method to be evaluated",
        type=str,
    )
    parser.add_argument(
        "--t",
        default=0.10,
        metavar="Number",
        help="Equality threshold",
        type=float,
    )

    args = parser.parse_args()
    print(f"\ntest list file path:{args.file} ")
    print(f"IIW image directory: {args.iiwdir}")
    for p in [args.file, args.iiwdir]:
        if not os.path.exists(p):
            print(f"Not exsists: {p}")
            exit(0)

    loader_dicts = {
        "Li_2018_full": Li_2018_CGI_Loader("./Li_2018_CGIntrinsics/CGI+IIW+SAW/cgi_pred_iiw/release_iiw"),
        "Li_2018_cgi": Li_2018_CGI_Loader("./Li_2018_CGIntrinsics/CGI/cgi_iiw/release_iiw_cgi"),
        "Luo_2020": Luo_2020_NIID_Net_Loader("./Luo_2020_NIID-Net/final_raw")
    }
    if args.method not in loader_dicts.keys():
        print(f"Undefined method: {args.method}")

    print(f"\nEvaluate {args.method} with threshold: {args.t}")
    evaluate_predictions(args.file, args.iiwdir, args.t, loader_dicts[args.method])


