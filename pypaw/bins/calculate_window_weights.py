#!/usr/bin/env pythoiin
# -*- coding: utf-8 -*-
"""
Calculate the adjoint source weighting based on the station and source
distribution.

:copyright:
    Wenjie Lei (lei@princeton.edu), 2016
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)
"""
from __future__ import print_function, division, absolute_import

import argparse
from pypaw.window_weights import WindowWeight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', action='store', dest='path_file', required=True,
                        help="path file")
    parser.add_argument('-p', action='store', dest='param_file', required=True,
                        help="param file")
    args = parser.parse_args()

    weightobj = WindowWeight(args.path_file, args.param_file)
    weightobj.run()


if __name__ == "__main__":
    main()
