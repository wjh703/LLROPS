# GROOPS-aligned File and Program Design

## Status

Design only.  This document defines the target artifact and program model for
LLROPS.  It does not introduce a new reader, writer, CLI option, or file
extension by itself.

## Decision summary

LLROPS will adopt the **semantic architecture** of GROOPS, not its byte-level
file encodings.

GROOPS gets composability from a small set of typed, versioned files shared by
many programs.  In particular, parameter names bind columns across normal
equations, matrices are first-class data, and a normal equation contains the
normal matrix, right-hand side, parameter names, and statistics.  Programs use
semantic input/output names such as `outputfileNormalEquation` and
`outputfileSolution`; they are not limited to a generic `inputfile` and
`outputfile` pair.

LLROPS will preserve this model with a GROOPS-like separation between the
scientific file type and its physical encoding.  Human-readable ASCII is the
default for observations, results, parameter names, solutions, metadata, and
catalogs.  Dense or blocked numerical matrices may use a binary encoding and
must be convertible to ASCII without changing their scientific type.  The
initial scope is LLR only.  It deliberately excludes GROOPS
formats for GNSS, GRACE, gravity fields, grids, and orbit variational equations
until an LLROPS program needs them.

The following principles are binding:

1. A new program-private JSON report is not a file format.  A reusable result
   has a named artifact type, reader, writer, version, validation, and tests.
2. Program configuration names describe the artifact, not its serialization.
   For example, `inputFileNormalPoints`, not `inputJson`.
3. A format is introduced only when it has a clear producer and at least one
   independent consumer, or when it is a scientific reproducibility boundary.
4. All persisted numerical data records units, time scale, frame where
   applicable, and enough provenance to reject incompatible inputs.
5. Parameter ordering is never an implicit cross-program contract.  Persistent
   parameter vectors and covariance matrices always carry `ParameterName`s.
6. The efficient direct path remains supported.  Persisting observation
   equations is optional, not a prerequisite for producing normal equations.
7. Encoding is selected by the file extension and is not part of the program's
   scientific contract.  Converting `matrix.dat.gz` to `matrix.txt.gz` must not
   change the matrix type, shape, values, or metadata.

## GROOPS reference model

The relevant GROOPS core formats are:

| GROOPS type | Role in a processing graph | LLR relevance |
|---|---|---|
| `Instrument` | Epoch-wise, typed observations, optionally split into arcs | Reference for a typed observation sequence, but not a direct LLR normal-point replacement |
| `Matrix` | General numeric matrix/vector; symmetric matrices can be triangular | Required conceptually for covariance and generic numerical exports |
| `ParameterName` | Stable object/type/temporal/interval identity for unknowns | Already present in LLROPS and must become a file-level contract |
| `NormalEquation` | Matrix, RHS, parameter names, observation count and quadratic sum | Already partly present; first priority for formalization |
| `StringList` / `StringTable` | Small typed auxiliary tables | Useful only for diagnostics and external interchange, not an initial core artifact |
| `EarthOrientationParameter`, `Ephemerides`, `Platform` | Domain model inputs | External standards or catalog/config data in the current LLR scope |

GROOPS itself does not use only TXT files.  Its documented extension rule is
`.txt` and unrecognized extensions for ASCII, `.dat` for binary, `.json` for
JSON, and `.xml` for XML; appending `.gz` enables transparent compression.
ASCII comments begin with `#`.  LLROPS adopts the useful part of this rule:
ASCII and binary are two encodings of a typed artifact, and gzip is transparent.
JSON remains available for reports and external interchange, but is not the
default scientific data encoding.

GROOPS has both fused and decomposed programs.  `SlrProcessing` computes
linearized equations, accumulates normal equations, and solves as one
configurable processing sequence.  Separately, `ObservationEquations2Files`,
`NormalsBuild`, and `NormalsSolverVCE` allow the same scientific stages to be
stored, inspected, recombined, and solved independently.  LLROPS should have
the same choice: a fused nonlinear adjustment for normal use and explicit file
boundaries for reproducible batch processing.

Sources:

- [GROOPS file formats](https://groops-devs.github.io/groops/html/general.fileFormat.html)
- [Instrument format](https://groops-devs.github.io/groops/html/fileFormat_instrument.html)
- [Matrix format](https://groops-devs.github.io/groops/html/fileFormat_matrix.html)
- [ParameterName format](https://groops-devs.github.io/groops/html/fileFormat_parameterName.html)
- [NormalEquation format](https://groops-devs.github.io/groops/html/fileFormat_normalEquation.html)
- [SLR processing](https://groops-devs.github.io/groops/html/SlrProcessing.html)
- [Observation equations to files](https://groops-devs.github.io/groops/html/ObservationEquations2Files.html)
- [Normal-equation build](https://groops-devs.github.io/groops/html/NormalsBuild.html)
- [Normal-equation solver/VCE](https://groops-devs.github.io/groops/html/NormalsSolverVCE.html)

### Disposition of the GROOPS format catalog

The table below prevents two opposite mistakes: overlooking a useful GROOPS
abstraction, or importing formats that have no LLR consumer.

| GROOPS format family | LLROPS disposition |
|---|---|
| `Instrument` | Adopt the epoch/arc/type concepts; retain an LLR-specific normal-point schema |
| `Matrix` | Adopt as a constrained generic numeric artifact in P2 |
| `ParameterName` | Adopt directly at the semantic level; current four-part LLROPS name already follows it |
| `NormalEquation` | Adopt directly at the semantic level and formalize in P1 |
| `ArcList` | Defer; source grouping in `NptDataset` is sufficient until arc programs consume it |
| `StringList`, `StringTable` | Defer; use only for explicit interchange needs, not as a universal escape format |
| `EarthOrientationParameter`, `MeanPolarMotion` | Keep IERS C04 and model-specific readers as external inputs; add a converter only if LLROPS starts producing EOP estimates |
| `Ephemerides` | Keep Calceph/SPICE-compatible kernels as external inputs; do not define an LLROPS ephemeris format |
| `Platform` | Map only the station subset to `StationCatalogFile`; satellite platform metadata is out of current scope |
| `EarthTide`, `OceanPoleTide`, `TideGeneratingPotential` | Keep model coefficient files external; no native output consumer exists yet |
| `Admittance`, `DoodsonEarthOrientationParameter`, `DoodsonHarmonic` | Out of current LLR processing scope |
| `GnssAntennaDefinition`, `GnssReceiverDefinition`, `GnssSignalBias` | Out of scope |
| `GriddedData`, `GriddedDataTimeSeries`, `Polygon` | Out of scope until a spatial product program is proposed |
| `PotentialCoefficients`, `TimeSplinesGravityField`, `TimeSplinesCovariance` | Out of scope |
| `SatelliteModel`, `VariationalEquation` | Out of scope until LLROPS estimates or integrates lunar/satellite dynamics |

The GROOPS program families used as design references are similarly limited:

| GROOPS program family | LLROPS use |
|---|---|
| `SlrProcessing` | Reference for the fused, staged `LlrAdjustment` workflow |
| `ObservationEquations2Files` | Reference for optional frozen equation export |
| `NormalsBuild`, `NormalsAccumulate`, `NormalsReorderAndAccumulate` | Reference for build, name alignment, and accumulation |
| `NormalsSolverVCE` | Reference for separate solution, sigma, covariance, variance-factor, and combined-normal outputs |
| `NormalsEliminate`, `NormalsScale`, `NormalsRegularization*` | Deferred program contracts; artifact design must not prevent them |
| `InstrumentConcatenate`, `InstrumentFilter`, `InstrumentStatistics*` | Reference for normal-point/result concatenate, filter, and QC programs |
| `ParameterNamesCreate`, `ParameterSelection2IndexVector` | Reference for future parameter inspection and selection programs |
| `MatrixCalculate`, `FileConvert` | Reference for explicit generic conversion, never implicit format guessing |

### Compatibility boundary

LLROPS will not initially promise to read or write GROOPS archive files.
GROOPS uses its own archive grammar and multi-file normal-equation layout.
LLROPS will use a smaller independently implemented ASCII/binary archive
grammar.  The compatibility promise is:

```text
same scientific object -> same required metadata -> same program composition
```

Interchange with GROOPS, if needed later, is a dedicated converter program. It
must not leak GROOPS parsing rules into LLROPS domain classes.

## Current LLROPS inventory

| LLROPS object/output | Current implementation | Target decision |
|---|---|---|
| `NptRecord` / `NptDataset` | CRD, MINI, and versioned LLROPS JSONL readers; LLROPS JSONL writer | Add native ASCII `NormalPointFile`; retain JSONL as legacy/interchange |
| `ObservationEquation` | Typed in-memory object with residual, sigma, partial blocks, identity, station, reflector, UTC epoch | Add optional `ObservationEquationFile` |
| O-C rows | Ad-hoc CSV or JSON output only | Add versioned ASCII `ObservationResultFile`; CSV remains export-only |
| `NormalEquations` | NPZ plus parameter-name JSON sidecar; supports load/add/solve | Migrate to GROOPS-like matrix/RHS/names/info group, retaining legacy reader |
| `ParameterName` | Structured in-memory name with GROOPS-compatible four parts | Add `ParameterVectorFile` and use names in all matrix-like artifacts |
| Adjustment result | Report JSON containing a mixture of diagnostics and machine state | Split into a text report and a restartable typed state file |
| Station/reflector catalog | Builtin or JSON/YAML reader | Define canonical ASCII catalog and writer before catalog-producing programs |

This assessment follows the current implementations in
`llrops/fileio/llrops_normal_point_file.py`,
`llrops/fileio/normal_equations.py`,
`llrops/fileio/observation_result_writer.py`,
`llrops/classes/observation/equations.py`, and
`llrops/fileio/catalogs.py`.

## Target artifact catalog

`File` below means an artifact reference in a program configuration.  A
physical artifact may be one compressed ASCII file or a directory file group.
The group contains several typed ASCII/binary members and allows atomic
publication without untracked sidecars.

| Artifact | Proposed name and storage | Producer -> consumer | Priority |
|---|---|---|---|
| Normal points | `NormalPointFile`; `normalPoints.txt[.gz]` | converters -> residuals/equations/adjustment | P1; JSONL legacy exists |
| Observation results | `ObservationResultFile`; `residuals.txt[.gz]` | residual calculation -> QC/statistics/export | P1 |
| Observation equations | `ObservationEquationFile`; `equations/` file group with ASCII info/names and ASCII or binary CSR arrays | equation calculation -> normals/QC | P3 |
| Normal equations | `NormalEquationFile`; `normals/` file group with ASCII info/names and ASCII or binary matrices | build/accumulate -> solve | P1 |
| Parameter vector | `ParameterVectorFile`; `solution.txt[.gz]` | solve/adjustment -> apply/report | P2 |
| Covariance matrix | `CovarianceMatrixFile`; `covariance.txt[.gz]` or `.dat[.gz]` | solve -> uncertainty/report | P2 |
| Generic matrix | `MatrixFile`; `matrix.txt[.gz]` or `.dat[.gz]` | diagnostics/converters only | P2 |
| Adjustment state | `AdjustmentStateFile`; `adjustmentState.txt[.gz]` | adjustment checkpoint -> resume | P3 |
| LLR catalog | `StationCatalogFile`, `ReflectorCatalogFile`; `stations.txt`, `reflectors.txt` | catalog conversion/apply solution -> observation modeling | P3 |

`NormalPointFile` is intentionally not replaced by a generic Instrument file.
An LLR normal point has domain-specific uncertainty, meteorology, wavelength,
station and reflector semantics.  It is closer to a strongly typed GROOPS
instrument subtype than to an untyped matrix.

### Encoding policy

The artifact type comes from the archive header, not the suffix alone.  The
suffix selects the encoding:

| Suffix | Encoding | Intended use |
|---|---|---|
| `.txt` | UTF-8 ASCII-style scientific text with `#` comments | Default, reviewable and archivable |
| `.txt.gz` | The same text through gzip | Default for large observation/result files |
| `.dat` | Versioned little-endian binary archive | Dense/symmetric/blocked matrices |
| `.dat.gz` | The same binary archive through gzip | Large persisted matrices where compression helps |
| `.json` | JSON representation of the same logical data, where implemented | External tools and debugging, not canonical by default |

XML is not introduced because LLROPS has no XML dependency or current consumer.
NPZ remains a `legacy-v0` encoding only.  A future `FileConvert` converts among
supported encodings by reading the typed object and writing it again; it never
performs text substitution.

All ASCII files begin with one machine-readable header line:

```text
llrops <artifactType> version=<YYYYMMDD>
```

They then contain type-specific metadata records, comment lines, counts, and
fixed-order values in that order.  The type reader owns the grammar; generic
tools only identify the header and encoding.
Floating-point output uses enough significant digits for an exact IEEE-754
double round trip.  Readers ignore blank lines and text from `#` to end of line.
Column names always include units in comments.  Locale-dependent formatting,
thousands separators, and `NaN`/`Inf` are forbidden.

Examples:

```text
llrops parameterName version=20260728
# object:type:temporal:interval unit
4
APOLLO15:position.x:: m
APOLLO15:position.y:: m
APOLLO15:position.z:: m
APOLLO:rangeBias:: m
```

```text
llrops normalPoint version=20260728
datasetName POLAC
arcCount 1
recordCount 1
# jd1_utc jd2_utc station reflector rtt_s sigma2w_s pressure_hPa temperature_K humidity_pct wavelength_nm stationCode reflectorCode
2451545.00000000000000000 1.42889802083333333e-01 APOLLO APOLLO15 2.51234567890123457e+00 1.20000000000000003e-10 8.72000000000000000e+02 2.93149999999999977e+02 2.50000000000000000e+01 5.32000000000000000e+02 71110 3
```

```text
llrops matrix version=20260728
# matrixType rows columns
LowerSymmetricMatrix 3 3
1.00000000000000000e+00
2.00000000000000000e-01 2.00000000000000000e+00
3.00000000000000000e-01 4.00000000000000000e-01 3.00000000000000000e+00
```

### Common metadata contract

Every derived artifact has typed metadata.  A single-file ASCII artifact stores
it as explicit key/value records after the header; comments are descriptive and
are ignored by the reader.  A multi-file artifact stores metadata in
`info.txt`, which must validate before matrix payloads are opened.

```text
llrops normalEquationInfo version=20260728
producerProgram LlrNormalEquations
llropsVersion 0.1.0
createdUtc 2026-07-28T00:00:00Z
observationUnit m
timeScale UTC
recordCount 6924
configurationSha256 <hex>
inputCount 1
input normalPoints.txt.gz <sha256> llrops.normalPoint
payload normalMatrix.dat.gz <sha256>
payload rightHandSide.dat.gz <sha256>
```

The exact units, `timeScale`, `frame`, and payload fields are type-specific.
Parameter-bound arrays store a unit for every parameter, aligned with the
ordered parameter-name vector.  Consequently `N[i,j]` has units
`1/(parameterUnits[i]*parameterUnits[j])`, the normal RHS element `n[i]` has
units `1/parameterUnits[i]`, and covariance element `Q[i,j]` has units
`parameterUnits[i]*parameterUnits[j]`.  A single global normal-matrix unit is
invalid once heterogeneous parameter types are introduced.

`configurationSha256` is calculated internally from a canonicalized resolved
configuration.  Its serialization algorithm is an implementation detail; JSON
does not appear in the scientific file merely because it is convenient for
hashing.  The hash excludes output paths, progress settings, and MPI chunk
size.  Input fingerprints are required for derived scientific artifacts, but
optional for imported external source files.

Combination does not require input files or time selections to have identical
hashes.  Each derived artifact also contains a `compatibility` block with the
observation convention, parameter convention, model family/version, reference
frames, and time-system conventions.  Programs compare those fields and
per-parameter units.  Full configuration and input hashes remain provenance,
not an overly strict equality key.

Writers create a sibling temporary file or directory, write and validate the
metadata and payload checksums, then rename it to the target.  A reader never
accepts a partial file group.

### NormalPointFile

P1 defines the native ASCII normal-point file.  It follows the GROOPS
Instrument pattern: a typed header, dataset/arc count, a commented column
definition, and one numeric/text record per epoch.  Calendar time is stored as
the lossless two-part Julian date plus an explicit `UTC` scale; an ISO timestamp
may appear in comments or exports but is not the precision-bearing field.
The physical fields of `NptRecord` remain:

```text
station/reflector identity, UTC transmit epoch, RTT [s], RTT uncertainty [s],
pressure [hPa], temperature [K], relative humidity [%], wavelength [nm]
```

CRD and MINI stay supported as import/export formats.  They are not required
to acquire LLROPS provenance semantics.  Existing `.llnpt[.gz]` JSONL files
remain readable as `legacy-v1`; the new default writer emits `.txt[.gz]`.

### ObservationResultFile

This is the formal persistent form of output from `LlrResiduals`.  It is a
versioned ASCII table with one row per normal point and a stable minimum schema:

```text
identity, source, epoch_utc, station_key, reflector_key,
observed_rtt_s, computed_rtt_s, oc_one_way_m, sigma_one_way_m,
elevation_up_deg, elevation_down_deg, convergence_status
```

`standard` and `full` are views of the same artifact type.  `full` may add
model diagnostics, but must not change the meaning or units of required
columns.  Variable-length status and identity fields are either enumerated
integers with a documented lookup section or whitespace-free escaped tokens;
free-form JSON objects are not embedded in rows.  CSV and JSON are export
formats generated by
`ObservationResultsExport`; they are never the authoritative interchange
format.

### ObservationEquationFile

This optional artifact represents one frozen linearization, not raw
observations.  It must contain:

```text
identity, source, epoch UTC, station/reflection keys,
l = observed_minus_computed [m], sigma [m], active/converged flags,
parameter names, and sparse design matrix A in CSR form.
```

The file group's `info.txt` also stores the model-state fingerprint, resolved
model configuration fingerprint, catalog fingerprint, ephemeris identity,
parameter names, and the linearization iteration.  A consumer rejects it when
the requested parametrization or model state differs.

CSR row pointers, column indices, values, observation vectors, and sigmas use
typed `MatrixFile`s.  `.txt.gz` is available for inspection; `.dat.gz` is the
recommended encoding for production-sized equations.

This is not the default implementation route: `LlrNormalEquations` may
continue to stream equations directly into normal equations.  The file exists
for inspection, reproducibility, partitioned processing, and independent
normal-equation construction.

### NormalEquationFile

The scientific payload remains:

```text
N = A^T P A
n = A^T P l
lPl, observation count, ParameterName vector, source/component metadata
```

Following GROOPS, the v1 native artifact is a related file group published in
one directory:

```text
normals/
  info.txt
  normalMatrix.dat.gz          # or .txt.gz
  rightHandSide.dat.gz         # or .txt.gz
  parameterNames.txt
```

`info.txt` includes `lPl`, observation count, payload hashes, and weight/model
metadata.  The normal matrix is stored densely or lower-symmetric initially
because current LLR systems are small; the API must not expose this choice.
Later, numbered block matrix members can be introduced without changing the
program contract.

The present `<stem>.npz` and `<stem>.parameters.json` pair remains readable as
`legacy-v0`.  All new writers emit the typed file group.  Normal equations can
only be accumulated after exact parameter-name alignment and compatible
unit/time/model checks.  A forced incompatible combination is never implicit;
it needs an explicit conversion/reweighting program.

### ParameterVectorFile, CovarianceMatrixFile, and MatrixFile

`ParameterVectorFile` is an ASCII table binding every value to its complete
`ParameterName` and unit on the same row.  It is the human- and
machine-readable answer of a solve.

`CovarianceMatrixFile` stores its matching ordered names, covariance/cofactor
kind, and per-parameter units in the same file group as the matrix.  Neither
may rely on the surrounding solution report.

`MatrixFile` is a general numerical interchange format for vectors or matrices
without parameter semantics.  Like GROOPS, it writes only one triangle for a
symmetric/triangular matrix and supports both `.txt[.gz]` and `.dat[.gz]`.
It does not replace the two typed formats above.  Using generic matrices for
parameter solutions would recreate the column-order bug that `ParameterName`
was designed to prevent.

### AdjustmentStateFile and report

The current adjustment JSON is valuable as a legacy report but mixes
presentation, diagnostics, state, and final normals.  It splits into:

| Output | Purpose | Required reader |
|---|---|---|
| `AdjustmentReportFile` | ASCII human/audit summary, iteration history, residual statistics | Optional report reader only |
| `AdjustmentStateFile` | Typed ASCII state plus optional matrix members: parameters, stochastic scales/factors, active set, stage and fingerprints | `LlrAdjustment --resume` |
| `ParameterVectorFile` / `CovarianceMatrixFile` | Final numerical estimate and uncertainty | solve/apply/report programs |
| `NormalEquationFile` | Final weighted normal equations, when requested | normal-equation programs |

Resume is valid only after the input and resolved model fingerprints match.

## Program I/O contract

### Declarative registry

The existing program decorator registers only a callable.  It becomes a
`ProgramSpec` containing:

```python
ProgramSpec(
    name="LlrNormalEquations",
    summary="Build normal equations at one fixed linearization.",
    inputs=(ArtifactSlot("inputFileNormalPoints", "NormalPointFile", many=True),),
    outputs=(ArtifactSlot("outputFileNormalEquations", "NormalEquationFile"),),
    config_schema=...,  # required/optional scalar and class settings
)
```

The runtime still passes a resolved `dict` to existing program functions during
migration.  Before the function starts, the registry validates required slots,
path cardinality, known keys, and artifact types.  This enables:

```bash
python -m llrops describe-program LlrNormalEquations
python -m llrops validate config.yml
```

and makes the program graph inspectable without running the models.

### Naming policy

Use singular/plural semantic keys:

```yaml
inputFileNormalPoints: normalPoints.txt.gz
inputFilesNormalPoints: [arc-a.txt.gz, arc-b.txt.gz]
outputFileNormalEquations: normals
outputFileSolution: solution.txt
outputFileCovariance: covariance.dat.gz
```

`outputDirectoryMini` is allowed where a conversion naturally creates one file
per input.  A program with several products uses several explicit output keys;
there is no untyped `outputFile` catch-all.

The existing keys (`inputNormalPoints`, `inputNormals`, `outputNormals`,
`outputCsv`, `outputJson`, and `outputSolutionJson`) remain supported as
deprecated aliases for one major format generation.  A config validator reports
the replacement but does not silently reinterpret an ambiguous key.

### Target program graph

```text
CRD / MINI / external LLR source
            |
            v
   NormalPointsConvert / NormalPointsConcatenate / NormalPointsFilter
            |
            v
      NormalPointFile (normalPoints.txt.gz)
         |                 |
         |                 +--> LlrResiduals --> ObservationResultFile --> QC/export
         |
         +--> LlrNormalEquations --------------------------> NormalEquationFile
         |                                                    |
         +--> LlrObservationEquations --> ObservationEquationFile
                                                 |             |
                                                 +--> ObservationEquationsToNormals
                                                               |
                                                               v
                                      NormalsAccumulate --> NormalEquationFile
                                                               |
                                                               v
                                                        NormalsSolve
                                                        |          |
                                                        v          v
                                             ParameterVector   Covariance

NormalPointFile + classes + parametrization --> LlrAdjustment
                 |                 |              |       |
                 |                 |              v       v
                 |                 |        report/state  final NormalEquationFile
                 |                 |              |
                 +-----------------+--------------+--> ParameterVector/Covariance
```

The direct `LlrNormalEquations` route and the persisted-equation route must
produce equal normal equations for a fixed model state, within documented
floating-point tolerance.

### Program evolution table

| Current program | Target contract | Action |
|---|---|---|
| `CrdToMini` | CRD files -> MINI directory | Keep as external-format conversion; add spec and aliases only |
| `NormalPointsToLlrops` | external/normal-point files -> `NormalPointFile` | Keep, rename output contract, add provenance |
| `LlrResiduals` | `NormalPointFile` -> `ObservationResultFile` | Keep computation; replace ad-hoc canonical output while retaining CSV export |
| `LlrNormalEquations` | `NormalPointFile` + model + parametrization -> `NormalEquationFile` | Keep as high-performance fused path |
| `NormalsCombineSolve` | normal-equation files -> optional combined normals plus JSON report | Deprecate in favor of `NormalsAccumulate` and `NormalsSolve` |
| `LlrAdjustment` | normal points + model + parametrization -> report/state/solution/covariance/normals | Keep as nonlinear fused workflow; add typed outputs and checkpointing |
| `LlrObservationEquations` | normal points + fixed model -> `ObservationEquationFile` | New, optional path |
| `ObservationEquationsToNormals` | equation file -> normal-equation file | New |
| `NormalsAccumulate` | many normal-equation files -> normal-equation file | New |
| `NormalsSolve` | normal-equation file -> solution/covariance/report | New |
| `LlrApplySolution` | solution + catalog/model state -> updated catalog/state | New only after catalog schema exists |

## Format introduction order

### P0: contract and validation foundation

1. Introduce `ArtifactType`, `ArtifactSlot`, and `ProgramSpec` without changing
   existing program behavior.
2. Add `describe-program` and static configuration validation.
3. Test that every registered program declares inputs, outputs, required keys,
   and aliases.
4. Publish a program/artifact reference generated from the registry.

### P1: make existing boundaries authoritative

1. Add the native ASCII `NormalPointFile` while preserving the current JSONL reader.
2. Replace the normal-equation sidecar pair with a v1 ASCII/binary file-group writer and
   `legacy-v0` reader.
3. Add the ASCII `ObservationResultFile` and make CSV/JSON presentation exports explicit.
4. Add normal-equation compatibility checks and `NormalsAccumulate`.

P1 yields a useful GROOPS-like file graph without persisting a potentially huge
design matrix.

### P2: complete the linear solution boundary

1. Add ASCII `ParameterVectorFile` plus dual-encoding
   `CovarianceMatrixFile` and constrained `MatrixFile`.
2. Split `NormalsCombineSolve` into `NormalsAccumulate` and `NormalsSolve`.
3. Make `LlrAdjustment` publish final solution and covariance through the same
   writers as `NormalsSolve`.
4. Add import/export adapters only when an external consumer is identified.

### P3: reproducibility and inspection boundaries

1. Add frozen `ObservationEquationFile` and equivalence tests against streaming
   normal-equation construction.
2. Add `AdjustmentStateFile` checkpoint/resume.
3. Define canonical station/reflector catalog files and `LlrApplySolution`.
4. Add QC, statistics, filter, and inspection programs that consume the typed
   artifacts rather than CSV.

## Validation and test requirements

Every new artifact must have:

1. Round-trip tests, including compressed and atomic-publish paths where
   supported.
2. Rejection tests for an unknown version, missing payload, checksum mismatch,
   non-finite number, wrong unit, incorrect time scale/frame, and duplicate
   parameter name.
3. Golden tiny fixtures stored under `tests/data/`, not fixtures generated by
   the writer under test.
4. Cross-program tests: read a producer output in an independent consumer.
5. Equivalence tests: direct LLR normal accumulation versus persisted equations
   for the same fixed linearization.
6. Compatibility tests for every supported `legacy-v0` reader until its stated
   removal date.

Normal-equation combination tests must prove that permuting parameter order in
one input yields the same result after name-based alignment.  This is the most
important GROOPS behavior to preserve.

## Scientific and operational guardrails

| Risk | Required control |
|---|---|
| Reusing equations at a different linearization | Model/catalog/config fingerprints; consumer rejection |
| Combining mismatched units or time systems | Mandatory metadata fields and compatibility check |
| Incorrect parameter column order | Parameter names stored with every parameter-bound array |
| Partial multi-file output | Directory file group plus atomic rename |
| CSV changes silently breaking a downstream workflow | CSV classified as export-only, not canonical input |
| ASCII precision loss | 17-significant-digit float output and exact round-trip tests |
| ASCII matrix size/performance | Same typed artifact supports `.dat.gz`; `FileConvert` provides inspectable `.txt.gz` |
| Large equation artifacts | Persist only by request; default to streaming normal accumulation |
| Schema growth for every diagnostic | Required core columns fixed; diagnostics are namespaced optional fields |
| External GROOPS interoperability pressure | Dedicated converter program, never ad-hoc parser branches |

## Explicit non-goals

- Reimplement the full GROOPS file-format catalog.
- Make all programs have exactly one input and one output.
- Replace CRD, MINI, C04, SPICE/Calceph, or IERS source formats.
- Persist every internal object.
- Force nonlinear `LlrAdjustment` through an observation-equation file on each
  iteration.
- Promise byte-for-byte GROOPS compatibility before a converter is designed and
  validated.

## Acceptance criteria for implementation approval

Implementation may start only after the first P0/P1 pull request can answer
these questions in code and documentation:

1. What exact scientific artifact does every input/output config key represent?
2. Which reader validates every proposed output type?
3. Can a normal equation be accumulated and solved without relying on a JSON
   report or positional parameter ordering?
4. Can a user discover program I/O and validate a config without running the
   forward model?
5. Does the direct path produce the same normal equations as the persisted
   path at one fixed linearization?

Until those answers are yes, adding more program-specific output files is not
considered progress toward GROOPS-style composability.
