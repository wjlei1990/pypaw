#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parent class for general asdf processing. Wraps things like MPI
and parallel I/O so they are invisible to users.

:copyright:
    Wenjie Lei (lei@princeton.edu), 2016
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)
"""
from __future__ import (absolute_import, division, print_function)
import os
import socket
from mpi4py import MPI

from pyasdf import ASDFDataSet
from . import mpi_ns
from .utils import smart_read_yaml, smart_read_json, is_mpi_env
from .utils import smart_check_path, smart_remove_file, smart_mkdir


class ProcASDFBase(object):

    def __init__(self, path, param, verbose=False, debug=False):

        self.comm = None
        self.rank = None
        self.size = None
        self.mpi = None

        self.path = path
        self.param = param
        self._verbose = verbose
        self._debug = debug

    def _parse_yaml(self, content):
        """
        Parse yaml file

        :param content:
        :return:
        """
        if isinstance(content, dict):
            # already in the memory
            return content
        elif isinstance(content, str):
            return smart_read_yaml(content, mpi_mode=self.mpi_mode,
                                   comm=self.comm)
        else:
            raise ValueError("Not recogonized input: %s" % content)

    def _parse_json(self, content):
        """
        Parse json file

        :param content:
        :return:
        """
        if isinstance(content, dict):
            # already in the memory
            return content
        elif isinstance(content, str):
            return smart_read_json(content, mpi_mode=self.mpi_mode,
                                   comm=self.comm)
        else:
            raise ValueError("Not recogonized input: %s" % content)

    def _parse_path(self):
        """
        How you parse the path arugment to fit your requirements
        """
        return self._parse_json(self.path)

    def _parse_param(self):
        """
        How you parse the param argument to fit your requirements
        """
        return self._parse_yaml(self.param)

    def detect_env(self):
        """
        Detect environment, mpi or not

        :return:
        """
        self.mpi_mode = is_mpi_env()
        hostname = socket.gethostname()
        if not self.mpi_mode:
            print("[Host {}] None-MPI environment detected".format(hostname))
        else:
            comm = MPI.COMM_WORLD
            self.comm = comm
            self.rank = comm.rank
            self.size = comm.size
            self.mpi = mpi_ns(
                comm=MPI.COMM_WORLD, rank=comm.rank, size=comm.size, MPI=MPI,
                processor=MPI.Get_processor_name())
            print("[Host {}] MPI environment detected: {} / {}".format(
                    hostname, self.rank, self.size))

    def print_info(self, dict_obj, title=""):
        """
        Print dict. You can use it to print out information
        for path and param

        :param dict_obj:
        :param title:
        :return:
        """
        def _print_subs(_dict, title):
            print("-"*10 + title + "-"*10)
            sorted_dict = sorted(((v, k) for v, k in _dict.items()))
            for key, value in sorted_dict:
                print("%s:  %s" % (key, value))

        if not isinstance(dict_obj, dict):
            raise ValueError("Input dict_obj should be type of dict")

        if not self.mpi_mode:
            _print_subs(dict_obj, title)
        else:
            if self.rank != 0:
                return
            _print_subs(dict_obj, title)

    def load_asdf(self, filename, mode="r"):
        """
        Load asdf file

        :param filename:
        :param mode:
        :return:
        """
        if self.mpi_mode:
            print("Load asdf file {} in mpi mode: rank/size: {}/{}".format(
                    filename, self.rank, self.size))
            return ASDFDataSet(filename, compression=None, debug=self._debug,
                               mode=mode, mpi=True)
        else:
            print("Load asdf file {} in none-mpi mode".format(filename))
            return ASDFDataSet(filename, debug=self._debug, mode=mode,
                               mpi=False)

    def _get_event_preferred_origin(self, event):
        pid = event.preferred_origin_id

        res = None
        for origin in event.origins:
            if origin.resource_id == pid:
                res = origin

        if res is None:
            raise ValueError("Failed to fetch preferred_origin by id")
        return res

    def get_events_mpi_2(self, asdf_fn):
        if self.mpi.rank == 0:
            ds = ASDFDataSet(asdf_fn, mode='r', mpi=False)
            events = ds.events
            del ds
        else:
            events = None

        events = self.mpi.comm.bcast(events, root=0)
        return events

    def get_events_mpi(self, asdf_fn):
        ds = ASDFDataSet(asdf_fn, mode='r', mpi=True)
        events = ds.events
        del ds
        self.mpi.comm.barrier()

        return events

    def check_input_file(self, filename):
        """
        Check existance of input file. If not, raise ValueError
        """
        if not smart_check_path(filename, mpi_mode=self.mpi_mode,
                                comm=self.comm):
            raise ValueError("Input file not exists: %s" % filename)

    def check_output_file(self, filename, remove_flag=True):
        """
        Check existance of output file. If directory of output file
        not exists, raise ValueError; If output file exists, remove it
        """
        dirname = os.path.dirname(filename)
        if not smart_check_path(dirname, mpi_mode=self.mpi_mode,
                                comm=self.comm):
            print("Output dir not exists and created: %s" % dirname)
            smart_mkdir(dirname, mpi_mode=self.mpi_mode,
                        comm=self.comm)

        if smart_check_path(filename, mpi_mode=self.mpi_mode,
                            comm=self.comm):
            if remove_flag:
                if self.rank == 0:
                    print("Output file already exists and removed:%s"
                          % filename)
                smart_remove_file(filename)

    @staticmethod
    def clean_memory(asdf_ds):
        """
        Delete asdf dataset
        """
        del asdf_ds

    @staticmethod
    def _missing_keys(necessary_keys, _dict):
        """
        Check if necessary_keys exists in _dict

        :param necessary_keys:
        :param _dict:
        :return:
        """
        if not isinstance(_dict, dict):
            raise ValueError("Input _dict must be type of dict")
        error_code = 0
        for _key in necessary_keys:
            if _key not in _dict.keys():
                print("%s must be specified in parameter file" % _key)
                error_code = 1
        if error_code:
            raise ValueError("Key values missing in paramter file")

    def _core(self, par_obj, file_obj):
        """
        Pure virtual function. Needs to be implemented in the
        child class.
        """
        raise NotImplementedError()

    def _validate_path(self, path):
        pass

    def _validate_param(self, param):
        pass

    def gather_data_to_master(self, local_values):
        gathered = self.mpi.comm.gather(local_values, root=0)

        if isinstance(local_values, list):
            # flattern list of list
            results = []
            if self.mpi.rank == 0:
                results = [x for l in gathered for x in l]
        elif isinstance(local_values, dict):
            # flattern list of dict
            results = {}
            if self.mpi.rank == 0:
                results = {k: v for d in gathered for k, v in d.items()}
        else:
            raise ValueError("unkonw type for local_values: {}".format(
                type(local_values)))

        return results

    def smart_run(self):
        """
        Job launch method

        :return:
        """
        self.detect_env()

        path = self._parse_path()
        self.print_info(path, title="Path Info")
        self._validate_path(path)

        param = self._parse_param()
        self.print_info(param, title="Param Info")
        self._validate_param(param)

        self._core(path, param)
