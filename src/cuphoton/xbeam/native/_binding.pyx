# SPDX-License-Identifier: Apache-2.0
# cython: language_level=3
from libc.stddef cimport size_t
from libc.stdint cimport uint32_t, uint64_t
from libcpp.string cimport string
cdef extern from "xbeam/radio.hpp" namespace "xbeam":
    cdef cppclass VdifHeader:
        bint invalid, complex_data
        uint32_t seconds_from_epoch, reference_epoch, frame_number
        uint32_t version, log2_channels, frame_length_bytes
        uint32_t bits_per_sample, thread_id, station_id, extended_data_version
        uint64_t samples_per_frame
    VdifHeader parse_vdif_header(const unsigned char*, size_t, bint, bint) except + nogil
    string inspect_vdif_json(const string&) except + nogil

def parse_header(const unsigned char[::1] data, bint strict=True, bint require_two_bit=True):
    cdef VdifHeader h
    cdef size_t n=data.shape[0]
    cdef const unsigned char* pointer=NULL
    if n:
        pointer=&data[0]
    with nogil:
        h=parse_vdif_header(pointer,n,strict,require_two_bit)
    return dict(invalid=h.invalid, complex_data=h.complex_data,
                seconds_from_epoch=h.seconds_from_epoch, reference_epoch=h.reference_epoch,
                frame_number=h.frame_number, version=h.version, log2_channels=h.log2_channels,
                frame_length_bytes=h.frame_length_bytes, bits_per_sample=h.bits_per_sample,
                thread_id=h.thread_id, station_id=h.station_id,
                extended_data_version=h.extended_data_version, samples_per_frame=h.samples_per_frame)

def inspect_recording(path):
    import json
    import os
    cdef string encoded=os.fsencode(path)
    cdef string result
    with nogil:
        result=inspect_vdif_json(encoded)
    return json.loads(result.decode('utf-8'))

cdef extern from "xbeam/radio.hpp" namespace "xbeam":
    cdef cppclass SwinMetadata:
        int baseline, configuration_id, source_id, frequency_id
        unsigned int mjd, pulsar_bin
        double seconds, weight, u, v, w
        char polarization_a, polarization_b
    string encode_swin(const SwinMetadata&, const float*, size_t) except + nogil
    void write_swin_file(const string&, const string&) except + nogil

def encode_visibility(int baseline, unsigned int mjd, double seconds,
                      int configuration_id, int source_id, int frequency_id,
                      bytes polarization, unsigned int pulsar_bin,
                      double weight, double u, double v, double w,
                      const float[::1] interleaved):
    """Encode an observable SWIN record using the native codec."""
    if len(polarization)!=2:
        raise ValueError("polarization requires two ASCII bytes")
    if interleaved.shape[0]%2:
        raise ValueError("spectrum requires interleaved complex pairs")
    cdef SwinMetadata m
    m.baseline=baseline; m.mjd=mjd; m.seconds=seconds
    m.configuration_id=configuration_id; m.source_id=source_id
    m.frequency_id=frequency_id; m.pulsar_bin=pulsar_bin
    m.polarization_a=polarization[0]; m.polarization_b=polarization[1]
    m.weight=weight; m.u=u; m.v=v; m.w=w
    cdef const float* pointer=NULL
    cdef size_t n=interleaved.shape[0]//2
    if n:
        pointer=&interleaved[0]
    cdef string result
    with nogil:
        result=encode_swin(m,pointer,n)
    return <bytes>result

def write_visibility_bytes(path, bytes records):
    """Create a new SWIN stream; existing paths are never overwritten."""
    import os
    cdef string name=os.fsencode(path)
    cdef string payload=records
    with nogil:
        write_swin_file(name,payload)

cdef extern from "xbeam/radio.hpp" namespace "xbeam":
    cdef cppclass RequestGeometry:
        long long ratio_numerator, ratio_denominator, fft_samples
        long long ffts_per_request, alignment_bytes, word_bytes
    cdef cppclass RequestCursor:
        bint valid
        long long origin_bytes, skipped_ffts, decoder_delta_samples
    RequestCursor request_cursor(long long, double, const RequestGeometry&, double) except + nogil

def buffered_cursor(long long local_samples, double drift_seconds,
                    long long ratio_numerator, long long ratio_denominator,
                    long long fft_samples, long long ffts_per_request,
                    long long alignment_bytes, long long word_bytes,
                    double sample_rate):
    cdef RequestGeometry g
    g.ratio_numerator=ratio_numerator; g.ratio_denominator=ratio_denominator
    g.fft_samples=fft_samples; g.ffts_per_request=ffts_per_request
    g.alignment_bytes=alignment_bytes; g.word_bytes=word_bytes
    cdef RequestCursor result
    with nogil:
        result=request_cursor(local_samples,drift_seconds,g,sample_rate)
    if not result.valid:
        return None
    return (result.origin_bytes,result.skipped_ffts,result.decoder_delta_samples)

from libcpp.vector cimport vector
cdef extern from "xbeam/radio.hpp" namespace "xbeam":
    cdef cppclass TextEntry:
        string key, value
        size_t line_number
    vector[TextEntry] parse_job_text(const string&) except + nogil

def parse_text(str text):
    cdef string encoded=text.encode('utf-8')
    cdef vector[TextEntry] entries
    cdef size_t i
    with nogil:
        entries=parse_job_text(encoded)
    return [(entries[i].key.decode('utf-8'),entries[i].value.decode('utf-8'),
             entries[i].line_number) for i in range(entries.size())]

cdef extern from "xbeam/radio.hpp" namespace "xbeam":
    string load_interop_job_json(const string&) except + nogil

def load_text_job(path):
    import json
    import os
    cdef string name=os.fsencode(path)
    cdef string result
    with nogil:
        result=load_interop_job_json(name)
    return json.loads(result.decode('utf-8'))

cdef extern from "xbeam/radio.hpp" namespace "xbeam":
    cdef cppclass IndexFrame:
        unsigned long long timestamp, physical_frame
        unsigned short thread
        unsigned char state
    cdef cppclass PhysicalIndexRun:
        uint64_t first,stop,packed_timestamp
    cdef cppclass IndexReader:
        IndexReader(const string&, const string&) except + nogil
        string metadata_json() except + nogil
        vector[PhysicalIndexRun] physical_runs() except + nogil
        vector[unsigned char] frame_validity(uint64_t,uint64_t) except + nogil
        vector[IndexFrame] query(unsigned short, unsigned long long, unsigned long long) except + nogil

cdef class NativeIndex:
    cdef IndexReader* reader
    def __cinit__(self, source, index):
        import os
        cdef string a=os.fsencode(source)
        cdef string b=os.fsencode(index)
        self.reader=NULL
        with nogil:
            self.reader=new IndexReader(a,b)
    def __dealloc__(self):
        del self.reader
    def metadata(self):
        import json
        cdef string result
        with nogil:
            result=self.reader.metadata_json()
        return json.loads(result.decode('utf-8'))
    def physical_runs(self):
        cdef vector[PhysicalIndexRun] runs
        with nogil:
            runs=self.reader.physical_runs()
        return [(r.first,r.stop,r.packed_timestamp) for r in runs]
    def frame_validity(self,uint64_t first,uint64_t count):
        cdef vector[unsigned char] result
        with nogil:
            result=self.reader.frame_validity(first,count)
        return list(result)
    def query(self, unsigned short thread, unsigned long long first, unsigned long long stop):
        cdef vector[IndexFrame] rows
        cdef size_t i
        with nogil:
            rows=self.reader.query(thread,first,stop)
        return [(rows[i].timestamp,rows[i].physical_frame,rows[i].thread,rows[i].state)
                for i in range(rows.size())]

cdef extern from "xbeam/radio.hpp" namespace "xbeam":
    cdef cppclass SourceBlock:
        long long fft_samples, output_sample, coarse_sample
        double centre_delay
    cdef cppclass RadioJob:
        RadioJob(const string&) except + nogil
        string normalized_json() except + nogil
        long long frame_slot(long long,int) except + nogil
        SwinMetadata record_metadata(const string&,long long,int,int,int,char,char) except + nogil
        vector[SourceBlock] source_blocks(const string&,int,int,const double*,size_t,bint) except + nogil
        JobRequestInputs requests(const string&,int,int,const RequestGeometry&,long long,long long,long long,long long,long long,bint,long long) except + nogil

cdef class NativeJob:
    cdef RadioJob* job
    def __cinit__(self,path):
        import os
        cdef string name=os.fsencode(path)
        self.job=NULL
        with nogil:
            self.job=new RadioJob(name)
    def __dealloc__(self):
        del self.job
    def record_metadata(self,str scan,long long integration,int station_a,int station_b,int frequency,bytes polarization):
        if len(polarization)!=2:
            raise ValueError("polarization requires two ASCII bytes")
        cdef string name=scan.encode('utf-8')
        cdef char pa=polarization[0],pb=polarization[1]
        cdef SwinMetadata m
        with nogil:
            m=self.job.record_metadata(name,integration,station_a,station_b,frequency,pa,pb)
        return dict(baseline=m.baseline,mjd=m.mjd,seconds=m.seconds,
                    configuration_id=m.configuration_id,source_id=m.source_id,
                    frequency_id=m.frequency_id,uvw=(m.u,m.v,m.w))
    def frame_slot(self,long long fps,int epoch):
        return self.job.frame_slot(fps,epoch)
    def source_blocks(self,str scan,int stream,int band,const double[::1] starts):
        cdef string name=scan.encode('utf-8')
        cdef vector[SourceBlock] result
        cdef const double* pointer=NULL
        cdef size_t n=starts.shape[0],i
        if n:
            pointer=&starts[0]
        with nogil:
            result=self.job.source_blocks(name,stream,band,pointer,n,False)
        return [(result[i].fft_samples,result[i].output_sample,
                 result[i].coarse_sample,result[i].centre_delay) for i in range(result.size())]

    def requests(self,str scan,int stream,int band,long long num,long long den,
                 long long fft,long long width,long long alignment,long long word,
                 long long first,long long stop,long long timeline_count,
                 long long job_slot,long long timestamp_origin,bint endpoint_mean=True,
                 long long maximum_requests=4096):
        cdef string name=scan.encode('utf-8')
        cdef RequestGeometry g
        g.ratio_numerator=num;g.ratio_denominator=den;g.fft_samples=fft
        g.ffts_per_request=width;g.alignment_bytes=alignment;g.word_bytes=word
        cdef JobRequestInputs result
        cdef size_t i
        with nogil:
            result=self.job.requests(name,stream,band,g,first,stop,timeline_count,
                                     job_slot,timestamp_origin,endpoint_mean,maximum_requests)
        return dict(first_fft=result.first_fft,stop_fft=result.stop_fft,
                    timeline_fft_count=result.timeline_fft_count,
                    timestamp_origin_frame=result.timestamp_origin,sample_rate=result.sample_rate,
                    request_fft_starts=[r.first_fft for r in result.requests],
                    request_fft_counts=[r.active_count for r in result.requests],
                    desired_sample_starts=[r.desired_sample for r in result.requests],
                    endpoint_drifts=[r.endpoint_drift for r in result.requests],
                    ordinary_source_starts=[list(r.ordinary_samples) for r in result.requests],
                    output_sample_starts=[list(r) for r in result.output_samples],
                    centre_delays_seconds=[list(r) for r in result.centre_delays],
                    read_average_delays_seconds=[list(r) for r in result.read_average],
                    left_delays=list(result.left_delays))

from libc.stdint cimport int64_t
cdef extern from "xbeam/timeline.hpp" namespace "xbeam":
    cdef cppclass PhysicalRun:
        int64_t first, stop, timestamp
    cdef cppclass SampleSpan:
        int64_t request_first, request_stop, physical_first
    cdef cppclass AnchorGap:
        int64_t allocation, physical_boundary, first, stop
    cdef cppclass FrameInterval:
        int64_t first,stop
    cdef cppclass PhysicalTimeline:
        PhysicalTimeline(const vector[PhysicalRun]&,int64_t) except + nogil
        const vector[int64_t]& allocation_starts() noexcept nogil
        const vector[int64_t]& anchors() noexcept nogil
        vector[FrameInterval] uncovered_heads() except + nogil
        vector[AnchorGap] deferred_gaps() except + nogil
        vector[SampleSpan] source_spans(int64_t,int64_t,int64_t) except + nogil
        vector[SampleSpan] latched_spans(int64_t,int64_t,int64_t,int64_t) except + nogil

cdef class NativeTimeline:
    cdef PhysicalTimeline* timeline
    def __cinit__(self, runs, int64_t allocation_frames):
        cdef vector[PhysicalRun] values
        cdef PhysicalRun item
        self.timeline=NULL
        for first,stop,timestamp in runs:
            item.first=first;item.stop=stop;item.timestamp=timestamp
            values.push_back(item)
        with nogil:
            self.timeline=new PhysicalTimeline(values,allocation_frames)
    def __dealloc__(self):
        del self.timeline
    def snapshot(self):
        cdef vector[int64_t] first=self.timeline.allocation_starts()
        cdef vector[int64_t] anchors=self.timeline.anchors()
        cdef vector[FrameInterval] heads
        cdef vector[AnchorGap] gaps
        cdef size_t i
        with nogil:
            heads=self.timeline.uncovered_heads()
            gaps=self.timeline.deferred_gaps()
        return dict(allocation_starts=list(first),anchors=list(anchors),
                    uncovered_heads=[(heads[i].first,heads[i].stop) for i in range(heads.size())],
                    deferred_gaps=[(gaps[i].allocation,gaps[i].physical_boundary,gaps[i].first,gaps[i].stop) for i in range(gaps.size())])
    def source_spans(self,int64_t first,int64_t count,int64_t samples_per_frame):
        cdef vector[SampleSpan] result
        cdef size_t i
        with nogil:
            result=self.timeline.source_spans(first,count,samples_per_frame)
        return [(result[i].request_first,result[i].request_stop,result[i].physical_first) for i in range(result.size())]
    def latched_spans(self,int64_t allocation,int64_t first,int64_t count,int64_t samples_per_frame):
        cdef vector[SampleSpan] result
        cdef size_t i
        with nogil:
            result=self.timeline.latched_spans(allocation,first,count,samples_per_frame)
        return [(result[i].request_first,result[i].request_stop,result[i].physical_first) for i in range(result.size())]

cdef extern from "xbeam/replay.hpp" namespace "xbeam":
    cdef cppclass RawRequestState:
        RawRequestState() except +
        vector[int64_t] anchors
        vector[unsigned char] events
    cdef cppclass RequestFact:
        int64_t first_fft,desired_sample,active_count
        double endpoint_drift
        vector[int64_t] ordinary_samples
    cdef cppclass JobRequestInputs:
        long long first_fft,stop_fft,timeline_fft_count,timestamp_origin
        double sample_rate
        vector[RequestFact] requests
        vector[vector[int64_t]] output_samples
        vector[vector[double]] centre_delays,read_average
        vector[double] left_delays
    cdef cppclass DiscardBatch:
        vector[int64_t] allocations
        vector[RequestCursor] cursors
        vector[unsigned char] valid,events
        vector[vector[unsigned char]] eligible
        vector[FrameInterval] excluded
    RawRequestState raw_request_state(const PhysicalTimeline&,int64_t,int64_t,bint) except + nogil
    DiscardBatch compile_discards(const RawRequestState&,const vector[RequestFact]&,const RequestGeometry&,double,int64_t,int64_t) except + nogil
    cdef cppclass WindowBounds:
        int64_t nearest,valid_first,valid_stop
        bint before,after
    WindowBounds window_bounds(int64_t,int64_t,double,int64_t,int64_t,int64_t,int64_t) except + nogil

cdef class NativeRawState:
    cdef RawRequestState state
    def __cinit__(self,NativeTimeline timeline not None,int64_t samples_per_frame,int64_t timestamp_origin,bint clock_frame_owner):
        with nogil:
            self.state=raw_request_state(timeline.timeline[0],samples_per_frame,timestamp_origin,clock_frame_owner)
    def snapshot(self):
        return dict(anchors=list(self.state.anchors),events=list(self.state.events))
    def discards(self,const int64_t[::1] first,const int64_t[::1] desired,
                 const double[::1] drifts,const int64_t[:,::1] ordinary,
                 const int64_t[::1] counts,
                 int64_t num,int64_t den,int64_t fft,int64_t ffts,
                 int64_t alignment,int64_t word,double rate,
                 int64_t frame_bytes,int64_t control_bits):
        cdef size_t n=first.shape[0],i,j
        if desired.shape[0]!=n or drifts.shape[0]!=n or ordinary.shape[0]!=n or counts.shape[0]!=n:
            raise ValueError("request vectors must have equal row counts")
        cdef vector[RequestFact] facts
        cdef RequestFact fact
        for i in range(n):
            if counts[i]<1 or counts[i]>ordinary.shape[1]:
                raise ValueError("request count exceeds row width")
            fact.first_fft=first[i];fact.desired_sample=desired[i];fact.endpoint_drift=drifts[i]
            fact.ordinary_samples.clear()
            for j in range(counts[i]):
                fact.ordinary_samples.push_back(ordinary[i,j])
            facts.push_back(fact)
        cdef RequestGeometry g
        g.ratio_numerator=num;g.ratio_denominator=den;g.fft_samples=fft
        g.ffts_per_request=ffts;g.alignment_bytes=alignment;g.word_bytes=word
        cdef DiscardBatch result
        with nogil:
            result=compile_discards(self.state,facts,g,rate,frame_bytes,control_bits)
        return dict(allocations=list(result.allocations),valid=list(result.valid),
                    events=list(result.events),eligible=[list(result.eligible[i]) for i in range(n)],
                    cursors=[(result.cursors[i].origin_bytes,result.cursors[i].skipped_ffts,result.cursors[i].decoder_delta_samples) for i in range(n)],
                    excluded=[(result.excluded[i].first,result.excluded[i].stop) for i in range(result.excluded.size())])

def returned_window_bounds(int64_t output,int64_t head,double delay,int64_t rate,
                           int64_t framed_extent,int64_t decoded_extent,int64_t fft):
    cdef WindowBounds result
    with nogil:
        result=window_bounds(output,head,delay,rate,framed_extent,decoded_extent,fft)
    return (result.nearest,result.before,result.after,result.valid_first,result.valid_stop)

cdef extern from "xbeam/window.hpp" namespace "xbeam":
    int64_t packed_frame_slot(uint64_t,int64_t) except + nogil
    cdef cppclass WindowGeometry:
        int64_t frames_per_fft,ffts_per_request,buffered_ffts,ring_segments
        int64_t allocation_frames,request_frames,read_buffer_ratio
    WindowGeometry size_windows(int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,double) except + nogil
    cdef cppclass NormalExtent:
        int64_t frame_bytes,payload_bytes,request_frames,read_buffer_ratio
        int64_t ratio_num,ratio_den,samples_per_frame
        int64_t framed_admission_samples,decoded_samples,allocation_bytes
    NormalExtent normal_extent(int64_t,int64_t,int64_t,int64_t,int64_t,int64_t) except + nogil
    int64_t leading_head(int64_t,double,int64_t,int64_t,int64_t) except + nogil
    cdef cppclass ReturnedWindow:
        int64_t fft,output,logical_head,physical_head
        double centre_delay
        int64_t framed_extent,decoded_extent
        bint published,clamp_negative
    vector[ReturnedWindow] returned_windows(const PhysicalTimeline&,const RawRequestState&,
                                            const JobRequestInputs&,const RequestGeometry&,
                                            const NormalExtent&,int64_t,int64_t,int64_t) except + nogil

def window_geometry(int64_t spf,int64_t fft,int64_t rate,int64_t subint_num,int64_t subint_den,
                    int64_t factor,int64_t segments,int64_t buffered,double allowance):
    cdef WindowGeometry g
    with nogil:
        g=size_windows(spf,fft,rate,subint_num,subint_den,factor,segments,buffered,allowance)
    return (g.frames_per_fft,g.ffts_per_request,g.buffered_ffts,g.ring_segments,
            g.allocation_frames,g.request_frames,g.read_buffer_ratio)

def leading_frame_head(int64_t output,double delay,int64_t rate,int64_t spf,int64_t quantum):
    return leading_head(output,delay,rate,spf,quantum)

def returned_window_contract(NativeTimeline timeline not None,NativeRawState state not None,dict inputs,
                             tuple geometry,tuple extent,int64_t source_stop,
                             int64_t quantum=1,int64_t control=30):
    cdef RequestGeometry g
    g.ratio_numerator,g.ratio_denominator,g.fft_samples,g.ffts_per_request,g.alignment_bytes,g.word_bytes=geometry
    cdef NormalExtent e=normal_extent(extent[0],extent[1],extent[2],extent[3],extent[4],extent[5])
    cdef JobRequestInputs data
    data.first_fft=inputs['first_fft'];data.stop_fft=inputs['stop_fft']
    data.timeline_fft_count=inputs['timeline_fft_count'];data.timestamp_origin=inputs['timestamp_origin_frame']
    data.sample_rate=inputs['sample_rate']
    cdef RequestFact f
    cdef size_t i,j
    for i in range(len(inputs['request_fft_starts'])):
        f.first_fft=inputs['request_fft_starts'][i];f.active_count=inputs['request_fft_counts'][i]
        f.desired_sample=inputs['desired_sample_starts'][i];f.endpoint_drift=inputs['endpoint_drifts'][i]
        f.ordinary_samples=inputs['ordinary_source_starts'][i]
        data.requests.push_back(f)
    data.output_samples=inputs['output_sample_starts'];data.centre_delays=inputs['centre_delays_seconds']
    data.left_delays=inputs['left_delays']
    cdef vector[ReturnedWindow] windows
    with nogil:
        windows=returned_windows(timeline.timeline[0],state.state,data,g,e,control,quantum,source_stop)
    return [(w.fft,w.output,w.logical_head,w.physical_head,w.centre_delay,
             w.framed_extent,w.decoded_extent,bool(w.published),bool(w.clamp_negative)) for w in windows]

def packed_timestamp_slot(uint64_t timestamp,int64_t fps):
    return packed_frame_slot(timestamp,fps)

cdef extern from "xbeam/window.hpp" namespace "xbeam":
    cdef cppclass ReplayRow:
        int64_t fft,physical_start,timing_displacement,valid_first,valid_stop
        bint accepted
    vector[ReplayRow] compile_replay_rows(const vector[ReturnedWindow]&,int64_t,int64_t,int64_t) except + nogil

def replay_rows(windows,int64_t fft,int64_t rate,int64_t source_stop):
    cdef vector[ReturnedWindow] rows
    cdef ReturnedWindow w
    for row in windows:
        w.fft,w.output,w.logical_head,w.physical_head,w.centre_delay,w.framed_extent,w.decoded_extent,w.published,w.clamp_negative=row
        rows.push_back(w)
    cdef vector[ReplayRow] result
    with nogil:
        result=compile_replay_rows(rows,fft,rate,source_stop)
    return [(r.fft,r.physical_start,r.timing_displacement,r.valid_first,r.valid_stop,bool(r.accepted)) for r in result]

cdef extern from "xbeam/execution.hpp" namespace "xbeam":
    string correlate_native(const vector[string]&) except + nogil
    bint native_cuda_available() noexcept

def cuda_available():
    return bool(native_cuda_available())

def correlate(job,output,index_dir,str scan,long long first_integration=0,
              int fft_tile=1250,int device=0,str vdif_io="posix",*,
              long long integration_count,str profile,int upload_workers=4,bint prefetch=True):
    """Run native radio execution; release the GIL for the complete run."""
    import os
    import json
    cdef vector[string] arguments
    for argument in [b"xbeam-native-correlate",os.fsencode(job),b"--output",os.fsencode(output),
                     b"--index-dir",os.fsencode(index_dir),b"--scan",scan.encode('utf-8'),
                     b"--first-integration",str(first_integration).encode('ascii'),
                     b"--integration-count",str(integration_count).encode('ascii'),
                     b"--fft-tile",str(fft_tile).encode('ascii'),b"--device",str(device).encode('ascii'),
                     b"--vdif-io",vdif_io.encode('ascii'),b"--profile",profile.encode('ascii'),
                     b"--upload-workers",str(upload_workers).encode('ascii'),
                     b"--prefetch",b"on" if prefetch else b"off"]:
        arguments.push_back(argument)
    cdef string report
    with nogil:
        report=correlate_native(arguments)
    return json.loads(report.decode('utf-8'))
