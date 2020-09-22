#!/usr/bin/env python
import matplotlib as mpl
mpl.use('Agg')  # NOQA
import argparse
from pypaw.adjoint import AdjointASDFMPI
from pypaw.adjoint_serial import AdjointASDFSerial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', action='store', dest='params_file',
                        required=True, help="parameter file")
    parser.add_argument('-f', action='store', dest='path_file', required=True,
                        help="path file")
    parser.add_argument('-v', action='store_true', dest='verbose',
                        help="verbose flag")
    parser.add_argument('--serial', action='store_true',
                        dest='serial_mode',
                        help="processing in serial mode flag")
    args = parser.parse_args()

    print("user input args: ", args)

    if args.serial_mode:
        proc = AdjointASDFSerial(
            args.path_file, args.params_file, verbose=args.verbose)
        proc.smart_run()
    else:
        proc = AdjointASDFMPI(
            args.path_file, args.params_file, verbose=args.verbose)
        proc.smart_run()


if __name__ == '__main__':
    main()
