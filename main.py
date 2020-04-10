from options.test_options import TestOptions
from cartoonGan.test import cartoonize
from test import pixelize

import argparse


def main():

    opt = TestOptions().parse()

    print('------------ Cartoonizing -------------')
    cartoonize(opt)
    print('------------ Pixelizing -------------')
    pixelize(opt)

if __name__ == '__main__':
    main()