#!/usr/bin/env python

import argparse
from pypaw.process import ProcASDFMPI
from pypaw.process_serial import ProcASDFSerial


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
                        help="serial mode flag")
    args = parser.parse_args()

    print("user input args:", args)

    if args.serial_mode:
        print("Processing in serial mode")
        proc = ProcASDFSerial(args.path_file, args.params_file, args.verbose)
        proc.smart_run()
    else:
        print("Processing in parallel (mpi) mode")
        # proc = ProcASDF(args.path_file, args.params_file, args.verbose)
        # proc.smart_run()
        proc = ProcASDFMPI(args.path_file, args.params_file, args.verbose)
        proc.smart_run()


if __name__ == '__main__':
    main()
