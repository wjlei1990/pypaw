#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parent class for singal processing asdf file and
handles parallel I/O so they are invisible to users.

:copyright:
    Wenjie Lei (lei@princeton.edu), 2016
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)
"""
from __future__ import (print_function, division, absolute_import)
import inspect
import time
from functools import partial
from pytomo3d.signal.process import process_stream
from .process import ProcASDF


def check_param_keywords(param):
    """
    Check the param keywords are the same with the keywords list of
    the function of process_stream
    """
    default_param = inspect.getargspec(process_stream).args
    default_param.remove("st")
    default_param.remove("inventory")
    if not param["remove_response_flag"]:
        # water_level is only used in remove instrument response
        default_param.remove("water_level")
    if set(default_param) != set(param.keys()):
        print("Missing: %s" % (set(default_param) - set(param.keys())))
        print("Redundant: %s" % (set(param.keys()) - set(default_param)))
        raise ValueError("Param is not consistent with function argument list")


def process_wrapper(stream, inv, param=None):
    """
    Process function wrapper for pyasdf

    :param stream:
    :param inv:
    :param param:
    :return:
    """
    param["inventory"] = inv
    return process_stream(stream, **param)


def update_param(event, param):
    """ update the param based on event information """
    origin = event.preferred_origin()
    origin = event.preferred_origin()
    event_latitude = origin.latitude
    event_longitude = origin.longitude
    event_time = origin.time

    # figure out interpolation parameter
    param["starttime"] = event_time + param["relative_starttime"]
    param.pop("relative_starttime")
    param["endtime"] = event_time + param["relative_endtime"]
    param.pop("relative_endtime")
    param["event_latitude"] = event_latitude
    param["event_longitude"] = event_longitude


class ProcASDFSerial(ProcASDF):

    def __init__(self, path, param, verbose=False, debug=True):
        super(ProcASDFSerial, self).__init__(
            path, param, verbose=verbose, debug=debug)

    def _core(self, path, param):
        if self.mpi_mode:
            raise ValueError("ProcASDFSerial only works in None-mpi mode")

        input_asdf = path["input_asdf"]
        input_tag = path["input_tag"]
        output_asdf = path["output_asdf"]
        output_tag = path["output_tag"]

        self.check_input_file(input_asdf)
        self.check_output_file(output_asdf, remove_flag=True)

        # WJ: set to 'a' for now since SPECFEM output is
        # a incomplete asdf file, missing the "auxiliary_data"
        # part. So give it 'a' permission to add the part.
        # otherwise, it there will be errors
        input_ds = self.load_asdf(input_asdf, mode='a')

        output_ds = self.load_asdf(output_asdf, mode='a')
        # add event information to output asdf
        output_ds.add_quakeml(input_ds.events)

        # update param based on event information
        update_param(input_ds.events[0], param)
        # check final param to see if the keys are right
        check_param_keywords(param)

        process_function = \
            partial(process_wrapper, param=param)

        station_tags = input_ds.waveforms.list()
        n = len(station_tags)
        print("Total Number of waveforms: {}".format(n))
        t0 = time.time()
        for _i, station in enumerate(station_tags):
            if _i > 0 and _i % max(int(n / 10), 2) == 0:
                print("Serial processing {0:5d}/{1:5d} stations using {2:.2f} "
                      "sec".format(_i, n, time.time() - t0))
            inv = input_ds.waveforms[station]["StationXML"]
            st = input_ds.waveforms[station][input_tag]
            try:
                st_new = process_function(st, inv)
                output_ds.add_waveforms(st_new, tag=output_tag)
                output_ds.add_stationxml(inv)
            except Exception as exp:
                print("Failed to processing {}.{} due to: {}".format(
                    station, input_tag, exp))

        output_ds.flush()
