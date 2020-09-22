import os
import sys
import re
import time
import collections
import itertools
from mpi4py import MPI

from pyasdf import ASDFDataSet


# mpi class
mpi_ns = collections.namedtuple(
    "mpi_ns", ["comm", "rank", "size", "MPI", "processor"])


# Global error handler for MPI environment
def global_except_hook(exctype, value, traceback):
    import sys
    try:
        import mpi4py.MPI
        sys.stderr.write("\n" + "*" * 32 + "\n")
        sys.stderr.write("Uncaught exception was detected on rank "
                         "{}.\n".format(mpi4py.MPI.COMM_WORLD.Get_rank()))
        from traceback import print_exception
        print_exception(exctype, value, traceback)
        sys.stderr.write("*" * 32 + "\n\n\n")
        sys.stderr.write("Calling MPI_Abort() to shut down "
                         "MPI processes...\n")
        sys.stderr.flush()
    finally:
        try:
            import mpi4py.MPI
            mpi4py.MPI.COMM_WORLD.Abort(1)
        except Exception as e:
            sys.stderr.write("*" * 32 + "\n")
            sys.stderr.write("Sorry, we failed to stop MPI, "
                             "this process will hang.\n")
            sys.stderr.write("*" * 32 + "\n")
            sys.stderr.flush()
            raise e


sys.excepthook = global_except_hook


class ASDFContainer(object):
    """
    ASDF file in-memory model
    """
    def __init__(self, mpi=None):
        self.events = []
        self.stream_buffer = {}
        self.staxml_buffer = {}
        self.auxiliary_buffer = {}

        # load all statxml to rank 0
        self.all_staxml_buffer = {}

        self.mpi = None
        self.__init_mpi__(mpi)

        self._filename = None

    def __init_mpi__(self, mpi):
        if mpi is None:
            comm = MPI.COMM_WORLD
            self.mpi = mpi_ns(
                comm=comm, rank=comm.rank, size=comm.size, MPI=MPI,
                processor=MPI.Get_processor_name()
            )
        elif isinstance(mpi, mpi_ns):
            self.mpi = mpi
        else:
            raise ValueError(f"wrong type of mpi ({type(mpi)}): {mpi}")

    def __repr__(self):
        return '{} | Events: {} | Streams: {} | StationXML: {} | ' \
            'Auxiliary: {}'.format(
                self.mpi_info(), len(self.events), len(self.stream_buffer),
                len(self.staxml_buffer), len(self.auxiliary_buffer))

    def add_events(self, events):
        self.events = events.copy()

    @property
    def station_names(self):
        return self.stream_buffer.keys()

    def get_stationxml(self, station_name):
        return self.staxml_buffer.get(station_name, None)

    @property
    def content_summary(self):
        n0 = len(self.events)
        n1 = self.total_stream_count()
        n2 = self.total_staxml_count()
        n3 = self.total_auxiliary_count()
        return {"events": n0, "stream": n1, "staxml": n2, "auxiliary": n3}

    @property
    def content_str(self):
        v = self.content_summary
        return "Events: {}| Stream: {}| Staxml: {}| Auxiliary: {}".format(
            v["events"], v["stream"], v["staxml"], v["auxiliary"])

    def add_stationxml(self, staxml, station_name):
        if staxml is None:
            return

        if station_name not in self.staxml_buffer:
            self.staxml_buffer[station_name] = {}
        self.staxml_buffer[station_name] = staxml

    def get_waveforms(self, station_name, waveform_tag):
        if station_name not in self.stream_buffer:
            return None

        return self.stream_buffer[station_name].get(waveform_tag, None)

    def add_waveforms(self, stream, station_name, waveform_tag):
        if stream is None:
            return

        if station_name not in self.stream_buffer:
            self.stream_buffer[station_name] = {}
        self.stream_buffer[station_name][waveform_tag] = stream

    def add_station_group(self, station_group, station_name, waveform_tags):

        station_group_waveform_tags = station_group.get_waveform_tags()

        if waveform_tags is None:
            waveform_tags = station_group_waveform_tags

        for wtag in waveform_tags:
            if wtag in station_group_waveform_tags:
                st = station_group[wtag]
                self.add_waveforms(st, station_name, wtag)

        if "StationXML" in station_group.list():
            self.add_stationxml(station_group["StationXML"], station_name)

    def get_station_group(self, station_name):
        """
        Return the station group as dict
        """
        res = {}

        if station_name in self.staxml_buffer:
            res["StationXML"] = self.staxml_buffer[station_name]

        if station_name in self.stream_buffer:
            res.update(self.stream_buffer[station_name])

        res["_station_name"] = station_name
        return res

    def mpi_info(self):
        return f"[MPI:{self.mpi.rank}/{self.mpi.size} | {self.mpi.processor}]"

    def _mpi_split_list_(self, data):
        rank_data = data[self.mpi.rank::self.mpi.size]
        return rank_data

    def _mpi_allgather(self, proc_data):
        gathered_data = self.mpi.comm.allgather(proc_data)

        if isinstance(proc_data, list):
            res = []
            for d in gathered_data:
                res.extend(d)
        elif isinstance(proc_data, dict):
            res = {}
            for d in gathered_data:
                res.update(d)

        self.mpi.comm.barrier()
        return res

    def total_stream_count(self):
        _len = len(self.stream_buffer)
        return self.mpi.comm.allreduce(_len, op=MPI.SUM)

    def total_staxml_count(self):
        _len = len(self.staxml_buffer)
        return self.mpi.comm.allreduce(_len, op=MPI.SUM)

    def total_auxiliary_count(self):
        _len = len(self.auxiliary_buffer)
        return self.mpi.comm.allreduce(_len, op=MPI.SUM)

    @property
    def file_basename(self):
        if self._filename is None:
            return "None"
        return os.path.basename(self._filename)

    def load_from_file(self, fn, waveform_tags=None, station_names=None,
                       mode='r'):
        """
        :param fn: asdf file name
        :param waveform_tags: load specific waveform within waveform_tags.
            if you want to load all waveform tags, leave it to None
        :param station_names: load specific station_names on the MPI
            processor. if set to None, then ASDFContainer will split
            it automatically.
        :param mode: the mode to open asdf file
        """

        t1 = time.time()
        self._filename = fn

        if isinstance(waveform_tags, str):
            waveform_tags = [waveform_tags]

        # parallel load
        ds = ASDFDataSet(fn, mode=mode, mpi=True)
        self.events = ds.events.copy()

        total_stations = ds.waveforms.list()
        if self.mpi.rank == 0:
            print(f"{self.mpi_info()} Total number of station groups "
                  f"({self.file_basename}): {len(total_stations)}")

        if station_names is None:
            # split the total_stations to each processor
            station_names = self._mpi_split_list_(total_stations)
        elif not isinstance(station_names, list):
            raise ValueError("station_names must be None or list of "
                             "station names (type of str)")
        station_names.sort()

        for station_name in station_names:
            station_group = ds.waveforms[station_name]
            try:
                self.add_station_group(station_group, station_name,
                                       waveform_tags)
            except Exception as exp:
                print("Failed to add station group '{}' due to: {}".format(
                    station_name, exp))

        self.mpi.comm.barrier()

        print(f"{self.mpi_info()} Number of streams loaded "
              f"({self.file_basename}): {len(self.stream_buffer)}")

        t2 = time.time()

        content = self.content_str
        if self.mpi.rank == 0:
            print("{} [Timer] Load takes time {:.2f} sec: ({}) {}".format(
                self.mpi_info(), t2 - t1, content, fn))

        del ds

    def _sync_metadata(self, ds):
        sendobj = []
        for sta_group in self.stream_buffer.values():
            if sta_group is None:
                continue
            for k, st in sta_group.items():
                for tr in st:
                    info = ds._add_trace_get_collective_information(tr, k)
                    sendobj.append(info)
                    tr.stats.__info = info

        data = self.mpi.comm.allgather(sendobj)

        trace_info = filter(
            lambda x: x is not None, itertools.chain.from_iterable(data))

        for info in trace_info:
            ds._add_trace_write_collective_information(info)

        self.mpi.comm.Barrier()

    def _sync_stationxml(self):
        staxmls = self.mpi.comm.gather(self.staxml_buffer, root=0)

        all_staxmls = {}
        if self.mpi.rank == 0:
            for v in staxmls:
                all_staxmls.update(v)
            print(f"{self.mpi_info()} Total number of staxml gathered: "
                  f"{len(all_staxmls)}")

        return all_staxmls

    def _write_events_and_staxml(self, outputfn):
        """
        Write events and staxml on rank 0.

        Gather all staxml to rank 0 so it can be written out later
        """
        all_staxmls = self._sync_stationxml()

        # add event and staxml on master node only
        if self.mpi.rank == 0:
            ds = ASDFDataSet(outputfn, mode='a', mpi=False)

            if self.events is not None and len(self.events) > 0:
                ds.add_quakeml(self.events)

            for staxml in all_staxmls.values():
                ds.add_stationxml(staxml)
            del ds

        # barrier to sync
        self.mpi.comm.Barrier()

    def _write_waveform_mpi(self, ds):
        # write out stream data
        self._sync_metadata(ds)

        for sta_group in self.stream_buffer.values():
            for k, st in sta_group.items():
                for tr in st:
                    ds._add_trace_write_independent_information(
                        tr.stats.__info, tr)

        self.mpi.comm.Barrier()

    def _write_auxiliary_mpi(self, ds):
        """
        Auxiliary data will be stored in default group name of
        "AuxiliaryData" in hdf5
        """
        aux_group = ds._auxiliary_data_group
        self._write_auxiliary_collective_data(aux_group)
        self._write_auxiliary_independent_data(aux_group)

    def write_to_file(self, outputfn):
        t1 = time.time()

        # write events and staxml on rank 0 only
        self._write_events_and_staxml(outputfn)

        # parallel write out
        ds = ASDFDataSet(outputfn, mode='a', mpi=True, compression=None)
        self._write_waveform_mpi(ds)
        self._write_auxiliary_mpi(ds)

        ds.flush()
        del ds

        t2 = time.time()
        content = self.content_str
        if self.mpi.rank == 0:
            print("{} [Timer] Write takes {:.2f} sec: ({}) {}".format(
                      self.mpi_info(), t2 - t1, content, outputfn))

    def add_auxiliary_data(self, aux_data):
        """
        :param data: dict with key of station name and value of data (list
            for each channel dict data)
            The channel dict data must be: {
                'object': value,
                'path': path of dataset,
                'parameters': metadata and dataset definiation parameters
            }
        """
        def __validate_path(path):
            tag_pattern = r"^[a-zA-Z0-9][a-zA-Z0-9_]*[a-zA-Z0-9]$"

            tag_path = path.strip("/").split("/")
            for path in tag_path:
                if re.match(tag_pattern, path) is None:
                    raise ValueError(
                        f"Tag name '{path}' is invalid. It must validate"
                        f"agains the regular experssion '{tag_pattern}'")

        # filter none values
        aux_data = dict((k, v) for k, v in aux_data.items() if v is not None)

        # check the data is valid
        local_paths = []
        for sta, sta_info in aux_data.items():
            for d in sta_info:
                if not isinstance(d, dict):
                    raise ValueError("channel data in auxliary must be type"
                                     "of dict")
                _keys = set(["object", "type", "path", "parameters"])
                if set(d.keys()) != _keys:
                    raise ValueError(f"keys{d.keys()} should be {_keys}")

                if d["type"] != "AuxiliaryData":
                    raise ValueError("type must be AuxiliaryData")

                __validate_path(d["path"])
                local_paths.append(d["path"])

        # check no duplicat path found
        all_paths = self.mpi.comm.reduce(local_paths, root=0)
        if self.mpi.rank == 0:
            if len(all_paths) != len(set(all_paths)):
                raise ValueError("Duplicate path found")

        # copy the aux data
        self.auxiliary_buffer = aux_data
        print("{} added {} auxiliary group (out of initial {} group)".format(
            self.mpi_info(), len(self.auxiliary_buffer), len(aux_data)))

    def _collect_auxiliary_metadata(self, aux_data):
        """
        Collect meta information over all processors

        The "dataset_creation_params" could be found here:
        https://docs.h5py.org/en/stable/high/group.html#Group.create_dataset
        """
        def __create_dataset_params(chan_info):
            info = {
                "data_name": chan_info["path"],
                "creation_params": {
                    "name": chan_info["path"],
                    "shape": chan_info["object"].shape,
                    "dtype": chan_info["object"].dtype,
                    "compression": None,
                    "compression_opts": None,
                    "shuffle": False,
                    "fletcher32": False,
                },
                "attrs": chan_info["parameters"],
            }
            return info

        local_metas = []
        for _sta, _sta_info in aux_data.items():
            for _chan_info in _sta_info:
                _meta = __create_dataset_params(_chan_info)
                local_metas.append(_meta)

        all_meta = self._mpi_allgather(local_metas)

        return all_meta

    def _write_auxiliary_collective_data(self, group):
        """
        Write meta information of auxiliary data into hdf5 collectivelly,
        including the dataset_creation_params and dataset_attrs
        """
        # collect necessary information without data
        collect_info = self._collect_auxiliary_metadata(self.auxiliary_buffer)

        for _sta_info in collect_info:
            ds = group.create_dataset(**_sta_info["creation_params"])

            for key, value in _sta_info["attrs"].items():
                ds.attrs[key] = value

    def _write_auxiliary_independent_data(self, group):
        """
        Write auxiliary independent data into hdf5, i.e., the data array
        """
        for _sta, _sta_info in self.auxiliary_buffer.items():
            for _chan_info in _sta_info:
                group[_chan_info["path"]][:] = _chan_info["object"]


def process_asdf_mpi(inputfile, outputfile, process_function, tag_map,
                     inputfile_mode='r'):
    """
    :param tag_map: input to output waveform tag, for example,
        {"raw_observed": "proc_obsd_17_40"}
    """
    t0 = time.time()
    input_dcon = ASDFContainer()
    input_dcon.load_from_file(inputfile,
                              waveform_tags=tag_map.keys(),
                              mode=inputfile_mode)
    print("input_dcon:", input_dcon)

    mpi = input_dcon.mpi

    output_dcon = ASDFContainer()
    output_dcon.events = input_dcon.events

    t1 = time.time()
    nstations = len(input_dcon.station_names)
    for idx, station_name in enumerate(input_dcon.station_names):
        if idx % int(max(nstations/20, 10)) == 0:
            print(f"{input_dcon.mpi_info()} processed {idx+1}/{nstations} "
                  f"stations")

        inv = input_dcon.get_stationxml(station_name)
        output_dcon.add_stationxml(inv, station_name)

        for input_tag, output_tag in tag_map.items():
            try:
                st = input_dcon.get_waveforms(station_name, input_tag)
                new_st = process_function(st, inv)
                output_dcon.add_waveforms(new_st, station_name, output_tag)
            except Exception as exp:
                print("Failed to process stream: {}".format(exp))
                continue

    mpi.comm.barrier()
    t2 = time.time()
    if mpi.rank == 0:
        print("{} [Timer] Processing takes {:.2f} sec".format(
            input_dcon.mpi_info(), t2 - t1))

    print("output_dcon: ", output_dcon)
    output_dcon.write_to_file(outputfile)

    t3 = time.time()
    if mpi.rank == 0:
        print("{} [Timer] Overall takes {:.2f} sec".format(
            input_dcon.mpi_info(), t3 - t0))


def process_two_asdf_mpi(asdf_file1, asdf_file2, process_function):
    dcon1 = ASDFContainer()
    dcon1.load_from_file(asdf_file1)

    mpi = dcon1.mpi

    if mpi.rank == 0:
        ds2 = ASDFDataSet(asdf_file2, mode='r', mpi=False)
        ds2_stations = ds2.waveforms.list()
        del ds2
    else:
        ds2_stations = None

    ds2_stations = mpi.comm.bcast(ds2_stations, root=0)

    ds1_stations = dcon1.station_names
    usable_stations = list(set(ds1_stations).intersection(set(ds2_stations)))

    dcon2 = ASDFContainer()
    dcon2.load_from_file(asdf_file2, station_names=usable_stations)

    t0 = time.time()

    results = {}
    for station_name in usable_stations:
        sta_group1 = dcon1.get_station_group(station_name)
        sta_group2 = dcon2.get_station_group(station_name)
        try:
            result = process_function(sta_group1, sta_group2)
        except Exception as exp:
            print("Failed to processing of station '{}' on rank {}: "
                  "{}".format(station_name, mpi.rank, exp))
        else:
            results[station_name] = result

    mpi.comm.barrier()

    t1 = time.time()
    if mpi.rank == 0:
        print("{} [Timer] Processing takes {:.2f} sec".format(
            dcon1.mpi_info(), t1 - t0))

    return results
