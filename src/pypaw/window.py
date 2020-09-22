#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class for window selection on asdf file and handles parallel I/O
so they are invisible to users.

:copyright:
    Wenjie Lei (lei@princeton.edu), 2016
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)
"""
from __future__ import (absolute_import, division, print_function)
from functools import partial
import os
import inspect
from copy import deepcopy
import json
import pyflex

from pytomo3d.window.window import window_on_stream
from pytomo3d.window.utils import merge_windows, stats_all_windows
from pytomo3d.window.io import get_json_content, WindowEncoder

from . import logger
from .utils import smart_mkdir
from .procbase import ProcASDFBase
from .asdf_container import process_two_asdf_mpi


def check_param_keywords(config):
    deletes = ["self", "noise_start_index", "noise_end_index",
               "signal_start_index", "signal_end_index",
               "window_weight_fct"]

    default_keywords = inspect.getargspec(pyflex.Config.__init__).args
    for d in deletes:
        default_keywords.remove(d)

    if set(default_keywords) != set(config.keys()):
        logger.warning(
            "Missing: %s" % (set(default_keywords) - set(config.keys())))
        logger.warning(
            "Redundant: %s" % (set(config.keys()) - set(default_keywords)))
        raise ValueError("config file is missing values compared to "
                         "pyflex.Config")


def load_window_config(param):
    config_dict = {}
    other_params = {}

    for key, value in param.items():
        for _k in ["instrument_merge_flag", "write_window_with_phase"]:
            # pop the key out
            other_params[_k] = value.pop(_k)

        check_param_keywords(value)
        config_dict[key] = pyflex.Config(**value)

    return config_dict, other_params


def write_window_json(results, output_file, with_phase=False):
    window_all = {}
    for station, sta_win in results.items():
        if sta_win is None:
            continue
        window_all[station] = {}
        _window_comp = {}
        for trace_id, trace_win in sta_win.items():
            _window = [get_json_content(_i, with_phase=with_phase)
                       for _i in trace_win]
            _window_comp[trace_id] = _window
        window_all[station] = _window_comp

    with open(output_file, 'w') as fh:
        j = json.dumps(window_all, cls=WindowEncoder, sort_keys=True,
                       indent=2, separators=(',', ':'))
        try:
            fh.write(j)
        except TypeError:
            fh.write(j.encode())


def window_wrapper(obsd_station_group, synt_station_group, config_dict=None,
                   obsd_tag=None, synt_tag=None, user_modules=None,
                   event=None, figure_mode=False, figure_dir=None):
    """
    Wrapper for asdf I/O
    """
    # Make sure everything thats required is there.
    if not hasattr(synt_station_group, "StationXML"):
        logger.warning("{}".format(synt_station_group))
        logger.warning("Missing StationXML from synt_staiton_group")
        return
    if not hasattr(obsd_station_group, obsd_tag):
        logger.warning("Missing tag '%s' from obsd_station_group" % obsd_tag)
        return
    if not hasattr(synt_station_group, synt_tag):
        logger.warning("Missing tag '%s' from synt_station_group" % synt_tag)
        return

    inv = synt_station_group.StationXML
    observed = getattr(obsd_station_group, obsd_tag)
    synthetic = getattr(synt_station_group, synt_tag)
    config_dict = deepcopy(config_dict)

    return window_on_stream(
        observed, synthetic, config_dict, station=inv,
        event=event, user_modules=user_modules,
        figure_mode=figure_mode, figure_dir=figure_dir)


def window_wrapper_2(
        obsd_station_group, synt_station_group, config_dict=None,
        obsd_tag=None, synt_tag=None, event=None,
        figure_mode=False, figure_dir=None):
    """
    Wrapper for ADSFContainer
    """
    # Make sure everything thats required is there.
    if "StationXML" not in synt_station_group:
        logger.warning("Missing 'StationXML' from synt_staiton_group: "
                       "{}".format(synt_station_group))
        return
    if obsd_tag not in obsd_station_group:
        logger.warning("Missing tag '%s' from obsd_station_group" % obsd_tag)
        return
    if synt_tag not in synt_station_group:
        logger.warning("Missing tag '%s' from synt_station_group" % synt_tag)
        return

    inv = synt_station_group["StationXML"]
    observed = obsd_station_group[obsd_tag]
    synthetic = synt_station_group[synt_tag]
    config_dict = deepcopy(config_dict)

    return window_on_stream(
        observed, synthetic, config_dict, station=inv,
        event=event, figure_mode=figure_mode, figure_dir=figure_dir)


class WindowASDF(ProcASDFBase):

    def __init__(self, path, param, debug=False):
        super(WindowASDF, self).__init__(
            path, param, debug=debug)

    def _parse_param(self):
        myrank = self.comm.Get_rank()
        param = self._parse_yaml(self.param)

        # reform the param from default
        default = param["default"]
        comp_settings = param["components"]
        results = {}
        for _comp, _settings in comp_settings.items():
            if myrank == 0:
                logger.info("Preapring params for components: %s" % _comp)
            results[_comp] = deepcopy(default)
            if _settings is None:
                continue
            for k, v in _settings.items():
                if myrank == 0:
                    logger.info("--> Modify key[%s] to value: %s --> %s"
                                % (k, results[_comp][k], v))
                results[_comp][k] = v

        return results

    def _validate_path(self, path):
        necessary_keys = ["obsd_asdf", "obsd_tag", "synt_asdf", "synt_tag",
                          "output_file", "figure_mode"]
        self._missing_keys(necessary_keys, path)

    def _validate_param(self, param):
        for key, value in param.items():
            necessary_keys = ["min_period", "max_period", "selection_mode"]
            self._missing_keys(necessary_keys, value)
            minp = value["min_period"]
            maxp = value["max_period"]
            if minp > maxp:
                raise ValueError("min_period(%6.2f) is larger than max_period"
                                 "(%6.2f)" % (minp, maxp))

    def _core(self, path, param):

        obsd_file = path["obsd_asdf"]
        synt_file = path["synt_asdf"]
        output_file = path["output_file"]
        output_dir = os.path.dirname(output_file)

        self.check_input_file(obsd_file)
        self.check_input_file(synt_file)
        smart_mkdir(output_dir, mpi_mode=self.mpi_mode,
                    comm=self.comm)

        obsd_tag = path["obsd_tag"]
        synt_tag = path["synt_tag"]
        figure_mode = path["figure_mode"]
        figure_dir = output_dir

        obsd_ds = self.load_asdf(obsd_file, mode='r')
        synt_ds = self.load_asdf(synt_file, mode='r')

        event = obsd_ds.events[0]

        # Ridvan Orsvuran, 2016
        # take out the user module values
        user_modules = {}
        for key, value in param.items():
            user_modules[key] = value.pop("user_module", None)

        config_dict, other_params = load_window_config(param)

        winfunc = partial(window_wrapper, config_dict=config_dict,
                          obsd_tag=obsd_tag, synt_tag=synt_tag,
                          user_modules=user_modules,
                          event=event, figure_mode=figure_mode,
                          figure_dir=figure_dir)

        windows = \
            obsd_ds.process_two_files(synt_ds, winfunc)

        myrank = self.comm.Get_rank()
        if myrank == 0:
            logger.info("other params: {}".format(other_params))

        instrument_merge_flag = other_params["instrument_merge_flag"]
        with_phase = other_params["write_window_with_phase"]

        if self.rank == 0:
            if instrument_merge_flag:
                # merge multiple instruments
                results = merge_windows(windows)
            else:
                # nothing is done
                results = windows

            stats_logfile = os.path.join(output_dir, "windows.stats.json")
            # stats windows on rand 0
            win_stats = stats_all_windows(
                results, obsd_tag, synt_tag, instrument_merge_flag,
                stats_logfile)

            logger.info("window statistics information: {}".format(win_stats))

            write_window_json(results, output_file, with_phase=with_phase)


def postprocess_windows(windows, instrument_merge_flag):
    if instrument_merge_flag:
        # merge multiple instruments
        results = merge_windows(windows)
    else:
        # nothing is done
        results = windows
    return results


class WindowASDFMPI(WindowASDF):
    def __init__(self, path, param, debug=False):
        super(WindowASDF, self).__init__(
            path, param, debug=debug)

    def gather_windows_to_master(self, local_windows):
        # gather all windows to the master(rank=0)
        gathered_windows = self.mpi.comm.gather(local_windows, root=0)

        all_windows = {}
        if self.mpi.rank == 0:
            for wins in gathered_windows:
                all_windows.update(wins)

        return all_windows

    def _core(self, path, param):

        obsd_file = path["obsd_asdf"]
        synt_file = path["synt_asdf"]
        output_file = path["output_file"]
        output_dir = os.path.dirname(output_file)

        self.check_input_file(obsd_file)
        self.check_input_file(synt_file)
        smart_mkdir(output_dir, mpi_mode=True,
                    comm=self.comm)

        obsd_tag = path["obsd_tag"]
        synt_tag = path["synt_tag"]
        figure_mode = path["figure_mode"]
        figure_dir = output_dir

        events = self.get_events_mpi(obsd_file)

        config_dict, other_params = load_window_config(param)

        winfunc = partial(window_wrapper_2,
                          config_dict=config_dict,
                          obsd_tag=obsd_tag,
                          synt_tag=synt_tag,
                          event=events[0].copy(),
                          figure_mode=figure_mode,
                          figure_dir=figure_dir)

        local_windows = process_two_asdf_mpi(obsd_file, synt_file, winfunc)

        windows = self.gather_windows_to_master(local_windows)
        # filter out None values
        windows = dict((k, v) for k, v in windows.items() if v is not None)

        if self.mpi.rank == 0:
            logger.info("other params: {}".format(other_params))
            merge_flag = other_params["instrument_merge_flag"]
            # clean up all the windows on master (rank=0)
            windows = postprocess_windows(windows, merge_flag)

        # write out windows on master (rank=0)
        if self.mpi.rank == 0:
            stats_logfile = os.path.join(output_dir, "windows.stats.json")
            win_stats = stats_all_windows(
                windows, obsd_tag, synt_tag,
                merge_flag, stats_logfile)
            logger.info("window statistics information: {}".format(win_stats))

            with_phase = other_params["write_window_with_phase"]
            write_window_json(windows, output_file, with_phase=with_phase)
