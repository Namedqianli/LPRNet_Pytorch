import argparse
import torch
import onnx
import numpy as np
from model.LPRNet import build_lprnet
from data.load_data import CHARS, CHARS_DICT

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--img_size', default=[94, 24], help='the image size')
    parser.add_argument('--dropout_rate', default=0, help='dropout rate.')
    parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
    parser.add_argument('--phase_train', default=False, type=bool, help='train or test phase flag.')
    parser.add_argument('--pretrained_model', default='./weights/exp2/Final_LPRNet_model.pth', help='pretrained base model')
    parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_parser()
    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    lprnet.cuda()
    dummy_input = torch.randn(1, 3, 24, 94).to(device)
    torch.onnx.export(
        lprnet,
        dummy_input,
        "./lprnet_kr_lp.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True
    )
