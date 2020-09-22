#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class that calculate adjoint source using asdf

:copyright:
    Wenjie Lei (lei@princeton.edu), 2016
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)
"""
from __future__ import (absolute_import, division, print_function)
from functools import partial

from pytomo3d.adjoint import measure_adjoint_on_stream
from .adjoint import load_adjoint_config, AdjointASDF
from .utils import dump_json
from .asdf_container import process_two_asdf_mpi


def write_measurements(content, filename):
    content_filter = dict(
        (k, v) for k, v in content.items() if v is not None)
    dump_json(content_filter, filename)


def measure_adjoint_wrapper(
        obsd_station_group, synt_station_group, config=None,
        obsd_tag=None, synt_tag=None, windows=None,
        adj_src_type="multitaper_misfit"):

    # Make sure everything thats required is there.
    if not hasattr(obsd_station_group, obsd_tag):
        print("Missing tag '%s' from obsd_station_group %s. Skipped." %
              (obsd_tag, obsd_station_group._station_name))
        return
    if not hasattr(synt_station_group, synt_tag):
        print("Missing tag '%s' from synt_station_group %s. Skipped." %
              (synt_tag, synt_station_group._station_name))
        return
    if not hasattr(obsd_station_group, "StationXML"):
        print("Missing tag 'STATIONXML' from obsd_station_group %s. Skipped" %
              (obsd_tag, obsd_station_group._station_name))

    try:
        window_sta = windows[obsd_station_group._station_name]
    except Exception as exp:
        print("Missing station in windows: {}".format(exp))
        return

    observed = getattr(obsd_station_group, obsd_tag)
    synthetic = getattr(synt_station_group, synt_tag)

    results = measure_adjoint_on_stream(
        observed, synthetic, window_sta, config, adj_src_type,
        figure_mode=False, figure_dir=None)

    return results


def measure_adjoint_wrapper_2(
        obsd_station_group, synt_station_group, config=None,
        obsd_tag=None, synt_tag=None, windows=None,
        adj_src_type="multitaper_misfit"):
    """
    To be used in ASDFContainer
    """

    # Make sure everything thats required is there.
    if obsd_tag not in obsd_station_group:
        print("Missing tag '%s' from obsd_station_group %s. Skipped." %
              (obsd_tag, obsd_station_group._station_name))
        return
    if synt_tag not in synt_station_group:
        print("Missing tag '%s' from synt_station_group %s. Skipped." %
              (synt_tag, synt_station_group._station_name))
        return
    if "StationXML" not in obsd_station_group:
        print("Missing tag 'STATIONXML' from obsd_station_group %s. Skipped" %
              (obsd_tag, obsd_station_group._station_name))
        return

    try:
        window_sta = windows[obsd_station_group["_station_name"]]
    except Exception as exp:
        print("Missing station in windows: {}".format(exp))
        return

    observed = obsd_station_group[obsd_tag]
    synthetic = synt_station_group[synt_tag]

    results = measure_adjoint_on_stream(
        observed, synthetic, window_sta, config, adj_src_type,
        figure_mode=False, figure_dir=None)

    return results


class MeasureAdjointASDF(AdjointASDF):
    """
    Make measurements on ASDF file. The output file is the json
    file which contains measurements for all the windows in
    the window file
    """
    def __init__(self, path, param, verbose=False, debug=False):
        super(MeasureAdjointASDFMPI, self).__init__(
            path, param, verbose=verbose, debug=debug)

    def _core(self, path, param):
        """
        Core function that handles one pair of asdf file(observed and
        synthetic), windows and configuration for adjoint source

        :param path: path information, path of observed asdf, synthetic
            asdf, windows files, observed tag, synthetic tag, output adjoint
            file, figure mode and figure directory
        :type path: dict
        :param param: parameter information for constructing adjoint source
        :type param: dict
        :return:
        """
        adjoint_param = param["adjoint_config"]

        obsd_file = path["obsd_asdf"]
        synt_file = path["synt_asdf"]
        obsd_tag = path["obsd_tag"]
        synt_tag = path["synt_tag"]
        window_file = path["window_file"]
        output_filename = path["output_file"]

        self.check_input_file(obsd_file)
        self.check_input_file(synt_file)
        self.check_input_file(window_file)
        self.check_output_file(output_filename)

        obsd_ds = self.load_asdf(obsd_file, mode="r")
        synt_ds = self.load_asdf(synt_file, mode="r")

        windows = self.load_windows(window_file)

        adj_src_type = adjoint_param["adj_src_type"]
        adjoint_param.pop("adj_src_type", None)

        config = load_adjoint_config(adjoint_param, adj_src_type)

        if self.mpi_mode:
            self.comm.barrier()

        measure_adj_func = \
            partial(measure_adjoint_wrapper, config=config,
                    obsd_tag=obsd_tag, synt_tag=synt_tag,
                    windows=windows,
                    adj_src_type=adj_src_type)

        results = obsd_ds.process_two_files(synt_ds, measure_adj_func)

        if self.rank == 0:
            print("output filename: %s" % output_filename)
            write_measurements(results, output_filename)


class MeasureAdjointASDFMPI(MeasureAdjointASDF):
    """
    Make measurements on ASDF file. The output file is the json
    file which contains measurements for all the windows in
    the window file
    """
    def __init__(self, path, param, verbose=False, debug=False):
        super(MeasureAdjointASDF, self).__init__(
            path, param, verbose=verbose, debug=debug)

    def _core(self, path, param):
        """
        Core function that handles one pair of asdf file(observed and
        synthetic), windows and configuration for adjoint source

        :param path: path information, path of observed asdf, synthetic
            asdf, windows files, observed tag, synthetic tag, output adjoint
            file, figure mode and figure directory
        :type path: dict
        :param param: parameter information for constructing adjoint source
        :type param: dict
        :return:
        """
        adjoint_param = param["adjoint_config"]

        obsd_file = path["obsd_asdf"]
        synt_file = path["synt_asdf"]
        obsd_tag = path["obsd_tag"]
        synt_tag = path["synt_tag"]
        window_file = path["window_file"]
        output_filename = path["output_file"]

        self.check_input_file(obsd_file)
        self.check_input_file(synt_file)
        self.check_input_file(window_file)
        self.check_output_file(output_filename)

        windows = self.load_windows(window_file)

        adj_src_type = adjoint_param["adj_src_type"]
        adjoint_param.pop("adj_src_type", None)

        config = load_adjoint_config(adjoint_param, adj_src_type)

        measure_adj_func = \
            partial(measure_adjoint_wrapper_2,
                    config=config,
                    obsd_tag=obsd_tag,
                    synt_tag=synt_tag,
                    windows=windows,
                    adj_src_type=adj_src_type)

        if self.mpi_mode:
            self.comm.barrier()

        local_results = process_two_asdf_mpi(
            obsd_file, synt_file, measure_adj_func)

        results = self.gather_data_to_master(local_results)

        if self.mpi.rank == 0:
            # filter the None values
            results = dict((k, v) for k, v in results.items() if v is not None)

            # write out on master node
            print("output filename: %s" % output_filename)
            write_measurements(results, output_filename)
