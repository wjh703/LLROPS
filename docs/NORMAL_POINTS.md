# Normal-point inputs

`NptRecord` is the only normal-point record consumed by the observation and
estimation layers. Source formats are adapted at the file boundary:

```text
MINI file  -> parse_mini_file -> NptDataset
CRD file   -> parse_crd_file  -> NptDataset
LLROPS file -> read_llrops_npt -> NptDataset
```

`read_normal_points(path)` detects and dispatches all three formats. It never
creates an intermediate file. In particular, CRD processing does not pass
through MINI and therefore retains the original CRD floating-point pressure,
temperature, humidity, wavelength, epoch, light time, and bin RMS values.

`CrdToMini` remains available only when a MINI file is required by external
software. Its fixed-width quantization is not part of the LLROPS computation
path.

## Canonical LLROPS files

Repeated or production processing can pre-convert source files into a
versioned JSON Lines file:

```yaml
- program: NormalPointsToLlrops
  inputNormalPoints:
    - /data/llr/crd
    - /data/llr/archive.mini.gz
  datasetName: campaign-2025
  outputFile: /data/llr/campaign-2025.llnpt.gz
```

The resulting `.llnpt`, `.llnpt.gz`, `.llrops.jsonl`, or
`.llrops.jsonl.gz` file can be passed directly to `inputNormalPoints`. Writing
uses a temporary file in the destination directory followed by an atomic
replace, so an interrupted conversion cannot leave a partial canonical file.

The first JSON object is a schema header:

```json
{"record_type":"header","schema":"llrops.normal_points","version":1,"dataset_name":"campaign-2025","n_records":1,"n_input_records":1,"n_invalid_records":0}
```

Each following line is one canonical record. Physical field names include
their units. UTC is stored as an explicit two-part Julian date so precision and
ERFA leap-second quasi-JD semantics are preserved:

```json
{"record_type":"normal_point","station_name":"APOL","reflector_name":"Apollo 15","transmit_epoch":{"jd1":2458850.5,"jd2":0.001158834,"scale":"utc"},"round_trip_time_s":2.5,"uncertainty_two_way_s":1.2e-11,"pressure_hpa":900.1,"temperature_k":280.1,"humidity_percent":50.0,"wavelength_nm":532.1,"index":0,"station_code":"70610","reflector_code":"3"}
```

Readers reject unknown schema versions, malformed records, non-finite values,
non-positive required physical quantities, invalid humidity, and a record
count that disagrees with the header.

## CRD interpretation

CRD record `20` meteorology is matched to each record `11` normal point by the
nearest circular seconds-of-day distance. Record `11` epoch event 2 is already
the ground transmit epoch. Epoch event 1 is a bounce epoch and is converted to
an approximate transmit epoch by subtracting half the recorded time of flight;
this existing approximation is recorded here because `NptRecord` requires a
transmit epoch.

