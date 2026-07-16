"""
I/O for POLAC / OCA "MINI" fixed-width LLR normal-point files.

The MINI format is a one-line-per-normal-point fixed-width ASCII format.
Each line carries *exactly* the following fields (1-based columns):

    cols  width  field                          unit / encoding
    ----- ------ ------------------------------ -------------------------------
    1     1      format code                    1 = MINI                [required]
    2     1      laser color code               1 = green, 2 = infrared (optional)
    3-10  8      launch date (UTC)              YYYYMMDD                [required]
    11-23 13     launch time of day (UTC)       HHMMSSsssssss (100 ns)  [required]
    24-37 14     observed two-way light time    integer, 0.1 ps         [required]
    38    1      reflector id                   0=A11 1=L1 2=A14 3=A15 4=L2 [required]
    39-43 5      station id                     ILRS-style 5-char code  [required]
    44-46 3      number of returns              integer (optional)
    47-52 6      uncertainty (two-way)          integer, 0.1 ps         [optional; not used for weights]
    53-55 3      signal-to-noise ratio          integer, S/N * 10 (optional)
    56    1      quality code                   single char (optional)
    57-62 6      surface pressure               integer, hPa * 100, > 0 [required]
    63-66 4      surface temperature            integer, 0.1 deg        [required]
    67-68 2      relative humidity              integer percent, 0-100  [required]
    69-73 5      laser wavelength               integer, 0.1 nm, > 0    [required]
    74    1      version code                   single char (optional)
    75-78 4      session duration               integer seconds (optional)
    79-80 2      (unused / blank)
    81-89 9      source format tag              free text, e.g. original format

References:
    https://polac.obspm.fr/llrdatae.html
    http://www.geoazur.fr/astrogeo/observations/donnees/lune/mini-format.html

Everything the parser produces is either (a) one of the raw MINI fields above,
or (b) a derived convenience quantity computed from them (SI scaling, the
unified UTC Epoch, station / reflector display names).  No CRD-style flags
(troposphere applied, center-of-mass applied, ...) exist in this module:
MINI data never carries such corrections, so they are never represented.

VALIDATION CONTRACT: every quantity carried by the MINI file and needed by
the downstream O-C / fit computation - launch epoch, light time, reflector,
station, MINI uncertainty, pressure, temperature, humidity, and laser wavelength
- is *guaranteed present and physically plausible* after parsing.  Invalid
records are never allowed into the returned NPT dataset: each invalid input line
writes a warning-level log entry that includes the 1-based line number, the
validation reason, and the original line content, then that record is skipped.
At the end of a file with skipped records, a summary warning-level log entry
reports how many nonblank data lines were read and how many valid records were
produced.  Downstream code
therefore never needs None checks, meteo lookups, uncertainty fallbacks, or
default wavelengths for MINI-owned fields.

NOTE on the temperature unit: this module follows the convention used by the
existing processing chain, ``temperature_c = raw / 10`` (i.e. 0.1 deg C).
Some MINI documentation describes the field as 0.1 K instead.  The conversion
is isolated in MINI-to-NPT conversion so that it
can be flipped in a single place if your data files use the 0.1 K convention.
"""
from __future__ import annotations

import gzip
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from llrops.base.epoch import Epoch, TimeScale

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TIME_UNIT_S = 1.0e-13          # light-time / uncertainty unit: 0.1 ps
TIME_100NS_S = 1.0e-7          # launch-time fractional unit: 100 ns
C_LIGHT_M_PER_S = 299_792_458.0
SECONDS_PER_DAY = 86400.0

MINI_LINE_MIN_LENGTH = 78      # duration field ends at col 78
MINI_LINE_FULL_LENGTH = 89     # source-format field ends at col 89

LASER_COLORS = {
    1: "green",
    2: "red",
}

REFLECTOR_NAMES = {
    0: "Apollo 11",
    1: "Lunokhod 1",
    2: "Apollo 14",
    3: "Apollo 15",
    4: "Lunokhod 2",
}

STATION_FULL_NAMES = {
    "71110": "McDonald 2.70",
    "71111": "McDonald MLRS1",
    "71112": "McDonald MLRS2",
    "01910": "Grasse",
    "56610": "Haleakala",
    "07941": "Matera",
    "70610": "Apache Point Observatory",
    "08834": "Wettzell",
}

# Keys / aliases used by sample_catalogs.py.
STATION_CATALOG_NAMES = {
    "71110": "MCDONALD",
    "71111": "MLRS1",
    "71112": "MLRS2",
    "01910": "GRASSE",
    "56610": "HALEAKALA",
    "07941": "MATERA",
    "70610": "APOL",
    "08834": "WETTZELL",
}


DEFAULT_MINI_IO_WARNING_LOG = "llr_mini_io_warnings.log"
_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())
_MINI_IO_LOG_HANDLERS: dict[Path, logging.Handler] = {}


def _resolve_mini_io_log_path(log_path=None) -> Path:
    """Return the concrete warning-log path used for skipped MINI records."""
    target = Path(log_path or DEFAULT_MINI_IO_WARNING_LOG).expanduser()
    if not target.is_absolute():
        target = Path.cwd() / target
    return target.resolve()


def _mini_io_warning_logger(log_path=None) -> logging.Logger:
    """Logger that writes invalid-record diagnostics to a file only."""
    target = _resolve_mini_io_log_path(log_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    logger = _LOGGER
    logger.setLevel(logging.WARNING)
    logger.propagate = False  # keep invalid-record details off stderr/stdout

    if target not in _MINI_IO_LOG_HANDLERS:
        handler = logging.FileHandler(target, mode="a", encoding="utf-8")
        handler.setLevel(logging.WARNING)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(handler)
        _MINI_IO_LOG_HANDLERS[target] = handler

    return logger


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def _open_text(path, *, encoding: str = "ascii", errors: str = "strict"):
    """Open a plain-text or .gz MINI file with universal newlines."""
    path = Path(path)
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding=encoding, errors=errors, newline=None)
    return path.open("r", encoding=encoding, errors=errors, newline=None)


def _blank_to_none(text: str) -> Optional[str]:
    value = str(text).strip()
    return value if value else None


def _parse_int(text: str, *, field: str, line_no: int, required: bool = True) -> Optional[int]:
    value = _blank_to_none(text)
    if value is None:
        if required:
            raise ValueError(f"line {line_no}: required MINI field {field!r} is blank")
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"line {line_no}: invalid MINI integer field {field!r}: {text!r}") from exc


def looks_like_mini_line(raw_line: str) -> bool:
    """Cheap structural test: does this line have the MINI fixed-width layout?"""
    raw = raw_line.rstrip("\r\n")
    if len(raw) < MINI_LINE_MIN_LENGTH or not raw.strip():
        return False
    padded = raw.ljust(MINI_LINE_FULL_LENGTH)
    return (
        padded[0:1].isdigit()
        and (padded[1:2].strip() == "" or padded[1:2].isdigit())
        and padded[2:10].isdigit()
        and padded[10:23].isdigit()
        and padded[23:37].strip().isdigit()
        and padded[37:38].strip().isdigit()
        and padded[38:43].strip().isdigit()
    )


def looks_like_mini_file(path) -> bool:
    """True if the first non-blank line of *path* looks like a MINI record."""
    with _open_text(path, encoding="ascii", errors="ignore") as fh:
        for raw in fh:
            if not raw.strip():
                continue
            return looks_like_mini_line(raw)
    return False


# ---------------------------------------------------------------------------
# Record / dataset model
# ---------------------------------------------------------------------------
@dataclass
class MiniRecord:
    """One MINI normal point: the raw file fields plus derived conveniences.

    The stored attributes correspond 1:1 to the MINI columns; derived
    quantities (SI scalings, display names) are read-only properties so they
    cannot drift out of sync with the raw values.
    """

    # --- raw MINI fields, in column order -------------------------------
    format_code: int
    laser_color_code: Optional[int]
    launch_date: str               # YYYYMMDD as written in the file
    launch_time: str               # HHMMSSsssssss as written in the file
    light_time_raw: int            # two-way light time, 0.1 ps
    reflector_id: int
    station_id: str                # normalized 5-char ILRS code
    number_of_returns: Optional[int]
    uncertainty_raw: int                   # original MINI two-way sigma, 0.1 ps
    signal_noise_ratio_raw: Optional[int]  # S/N * 10
    quality_code: Optional[str]
    pressure_raw: int                      # hPa * 100 (> 0)
    temperature_raw: int                   # 0.1 deg (see module docstring)
    humidity_percent: int                  # %, 0..100
    wavelength_raw: int                    # 0.1 nm (> 0)
    version_code: Optional[str]
    duration_s: Optional[int]
    source_format: Optional[str]

    # --- derived, computed once at parse time ---------------------------
    launch_epoch: Epoch = field(repr=False)
    seconds_of_day: float = 0.0
    index: int = 0                 # 0-based record index within the file
    source_line_no: int = 0        # 1-based line number in the source MINI file
    source_line: str = ""         # original source line without trailing newline

    def __post_init__(self) -> None:
        if not isinstance(self.launch_epoch, Epoch):
            raise TypeError("launch_epoch must be an Epoch.")
        self.launch_epoch.require_scale(TimeScale.UTC, name="launch_epoch")

    # ------------------------------------------------------------------
    # Derived scalar conveniences (computed from the raw fields)
    # ------------------------------------------------------------------
    @property
    def observed_round_trip_time_s(self) -> float:
        """Observed two-way light time in seconds."""
        return float(self.light_time_raw) * TIME_UNIT_S

    @property
    def observed_range_m(self) -> float:
        """One-way range equivalent, 0.5 * c * round-trip time."""
        return 0.5 * C_LIGHT_M_PER_S * self.observed_round_trip_time_s

    @property
    def uncertainty_two_way_s(self) -> float:
        """Original MINI uncertainty as two-way round-trip light-time sigma [s]."""
        return float(self.uncertainty_raw) * TIME_UNIT_S

    @property
    def uncertainty_two_way_ps(self) -> float:
        return float(self.uncertainty_raw) * 0.1

    @property
    def range_uncertainty_one_way_m(self) -> float:
        """Original MINI one-way range sigma [m], retained for diagnostics."""
        return 0.5 * C_LIGHT_M_PER_S * self.uncertainty_two_way_s

    @property
    def signal_noise_ratio(self) -> Optional[float]:
        if self.signal_noise_ratio_raw is None:
            return None
        return float(self.signal_noise_ratio_raw) / 10.0

    @property
    def pressure_hpa(self) -> float:
        return float(self.pressure_raw) / 100.0

    @property
    def temperature_c(self) -> float:
        # See the module docstring for the 0.1 degC vs 0.1 K caveat.
        return float(self.temperature_raw) / 10.0

    @property
    def temperature_k(self) -> float:
        return self.temperature_c + 273.15

    @property
    def wavelength_nm(self) -> float:
        return float(self.wavelength_raw) / 10.0

    @property
    def wavelength_um(self) -> float:
        return self.wavelength_nm / 1000.0

    @property
    def laser_color(self) -> Optional[str]:
        if self.laser_color_code is None:
            return None
        return LASER_COLORS.get(self.laser_color_code)

    @property
    def reflector_name(self) -> str:
        return REFLECTOR_NAMES.get(self.reflector_id, str(self.reflector_id))

    @property
    def station_full_name(self) -> str:
        return STATION_FULL_NAMES.get(self.station_id, self.station_id)

    @property
    def station_name(self) -> str:
        """Catalog token used to resolve the station in sample_catalogs."""
        return STATION_CATALOG_NAMES.get(self.station_id, self.station_full_name)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _parse_launch_epoch(
    date_text: str,
    time_text: str,
    *,
    line_no: int,
) -> tuple[Epoch, float]:
    """Return (UTC Epoch, seconds-of-day) for the MINI launch fields."""
    date_value = date_text.strip()
    time_value = time_text.strip()

    if len(date_value) != 8 or not date_value.isdigit():
        raise ValueError(f"line {line_no}: invalid MINI launch date {date_text!r}; expected YYYYMMDD")
    if len(time_value) != 13 or not time_value.isdigit():
        raise ValueError(f"line {line_no}: invalid MINI launch time {time_text!r}; expected HHMMSSsssssss")

    hour = int(time_value[0:2])
    minute = int(time_value[2:4])
    second = int(time_value[4:6])
    frac_100ns = int(time_value[6:13])

    seconds_of_day = hour * 3600.0 + minute * 60.0 + second + frac_100ns * TIME_100NS_S
    return (
        Epoch.from_date_seconds(date_value, seconds_of_day, scale=TimeScale.UTC),
        seconds_of_day,
    )


def parse_mini_line(raw_line: str, *, line_no: int = 0, index: int = 0) -> MiniRecord:
    """Parse one MINI fixed-width line into a :class:`MiniRecord`."""
    raw = raw_line.rstrip("\r\n")
    if len(raw) < MINI_LINE_MIN_LENGTH:
        raise ValueError(
            f"line {line_no}: MINI line is too short "
            f"({len(raw)} chars; expected at least {MINI_LINE_MIN_LENGTH})"
        )
    padded = raw.ljust(MINI_LINE_FULL_LENGTH)

    launch_date = padded[2:10].strip()
    launch_time = padded[10:23].strip()
    launch_epoch, seconds_of_day = _parse_launch_epoch(launch_date, launch_time, line_no=line_no)

    station_id_raw = padded[38:43].strip()
    if not station_id_raw:
        raise ValueError(f"line {line_no}: required MINI field 'station_id' is blank")
    # Some MINI files encode leading-zero station IDs with a blank first
    # column, e.g. " 1910" for Grasse.  Normalize numeric IDs to the
    # canonical 5-character code (01910 / 07941 / 08834 / ...).
    station_id = station_id_raw.zfill(5) if station_id_raw.isdigit() else station_id_raw

    # Fields the downstream computation depends on are *required*: a blank
    # value raises here, with the offending line number, so that no later
    # stage ever needs to handle missing MINI-owned data.
    uncertainty_raw = _parse_int(padded[46:52], field="uncertainty", line_no=line_no)
    pressure_raw = _parse_int(padded[56:62], field="pressure", line_no=line_no)
    temperature_raw = _parse_int(padded[62:66], field="temperature", line_no=line_no)
    humidity_percent = _parse_int(padded[66:68], field="humidity", line_no=line_no)
    wavelength_raw = _parse_int(padded[68:73], field="wavelength", line_no=line_no)

    # Physical sanity: zero / negative values in these fields are placeholders
    # for missing data and must be rejected just like blanks.
    if uncertainty_raw <= 0:
        raise ValueError(
            f"line {line_no}: MINI uncertainty must be > 0 (0.1 ps), got {uncertainty_raw}"
        )
    if pressure_raw <= 0:
        raise ValueError(
            f"line {line_no}: MINI pressure must be > 0 (hPa*100), got {pressure_raw}"
        )
    if not (0 <= humidity_percent <= 100):
        raise ValueError(
            f"line {line_no}: MINI humidity must be within 0..100 %, got {humidity_percent}"
        )
    if wavelength_raw <= 0:
        raise ValueError(
            f"line {line_no}: MINI wavelength must be > 0 (0.1 nm), got {wavelength_raw}"
        )

    return MiniRecord(
        format_code=_parse_int(padded[0:1], field="format_code", line_no=line_no),
        laser_color_code=_parse_int(padded[1:2], field="laser_color_code", line_no=line_no, required=False),
        launch_date=launch_date,
        launch_time=launch_time,
        light_time_raw=_parse_int(padded[23:37], field="light_time", line_no=line_no),
        reflector_id=_parse_int(padded[37:38], field="reflector_id", line_no=line_no),
        station_id=station_id,
        number_of_returns=_parse_int(padded[43:46], field="number_of_returns", line_no=line_no, required=False),
        uncertainty_raw=uncertainty_raw,
        signal_noise_ratio_raw=_parse_int(padded[52:55], field="signal_noise_ratio", line_no=line_no, required=False),
        quality_code=_blank_to_none(padded[55:56]),
        pressure_raw=pressure_raw,
        temperature_raw=temperature_raw,
        humidity_percent=humidity_percent,
        wavelength_raw=wavelength_raw,
        version_code=_blank_to_none(padded[73:74]),
        duration_s=_parse_int(padded[74:78], field="duration", line_no=line_no, required=False),
        source_format=_blank_to_none(padded[80:89]),
        launch_epoch=launch_epoch,
        seconds_of_day=seconds_of_day,
        index=index,
    )


def parse_mini_file(path, *, mini_io_log_path=None):
    """Parse a MINI normal-point file (.dat / .mini, optionally gzipped).

    Invalid nonblank records are written to ``mini_io_log_path`` and skipped;
    there is deliberately no ``raise`` / ``skip`` policy switch.  Every log
    entry contains the 1-based line number, the validation reason, and the
    original line content so data problems can be traced back to the source file.
    If ``mini_io_log_path`` is omitted, invalid records are appended to
    ``llr_mini_io_warnings.log`` in the current working directory.

    After this function returns, every record in the dataset is guaranteed to
    carry complete MINI-owned data: launch epoch, observed two-way light time,
    reflector id, station id, MINI uncertainty, pressure, temperature, humidity,
    and wavelength.  Catalog resolution and WRMS-table matching depend on the
    processing catalogs and are validated once at the
    :meth:`LlrObservationProcessor.process` boundary.
    """
    path = Path(path)
    if not looks_like_mini_file(path):
        raise ValueError(f"Input does not look like a MINI fixed-width normal-point file: {path}")

    records: List[MiniRecord] = []
    n_input_records = 0
    n_invalid_records = 0
    invalid_record_logger = None

    with _open_text(path, encoding="ascii", errors="strict") as fh:
        for line_no, raw in enumerate(fh, start=1):
            if not raw.strip():
                continue
            n_input_records += 1
            try:
                record = parse_mini_line(raw, line_no=line_no, index=len(records))
                record.source_line_no = line_no
                record.source_line = raw.rstrip("\r\n")
                records.append(record)
            except ValueError as exc:
                n_invalid_records += 1
                if invalid_record_logger is None:
                    invalid_record_logger = _mini_io_warning_logger(mini_io_log_path)
                original = raw.rstrip("\r\n")
                invalid_record_logger.warning(
                    "%s: invalid MINI record at line %s: %s; content=%r; skipping.",
                    path,
                    line_no,
                    exc,
                    original,
                )

    if n_invalid_records:
        if invalid_record_logger is None:
            invalid_record_logger = _mini_io_warning_logger(mini_io_log_path)
        invalid_record_logger.warning(
            "%s: MINI read summary: data lines read=%s, valid records=%s, "
            "invalid records skipped=%s.",
            path,
            n_input_records,
            len(records),
            n_invalid_records,
        )

    if not records:
        raise ValueError(
            f"No valid MINI normal-point records found in {path}. "
            f"Data lines read={n_input_records}, invalid records skipped={n_invalid_records}."
        )

    from llrops.fileio.npt import NptDataset, npt_records_from_mini

    return NptDataset(
        records=npt_records_from_mini(records),
        name=path.stem,
        n_input_records=n_input_records,
        n_invalid_records=n_invalid_records,
    )


# ---------------------------------------------------------------------------
# Writing (used by the CRD -> MINI converter and for round-tripping)
# ---------------------------------------------------------------------------
def _format_opt_int(value: Optional[int], width: int) -> str:
    if value is None:
        return " " * width
    text = f"{int(value):0{width}d}" if value >= 0 else f"{int(value):d}"
    if len(text) > width:
        raise ValueError(f"Integer value {value} does not fit in {width} MINI columns")
    return text.rjust(width)


def format_mini_line(record: MiniRecord) -> str:
    """Serialize a :class:`MiniRecord` to one 89-character MINI line."""
    parts = [
        f"{int(record.format_code):1d}",
        " " if record.laser_color_code is None else f"{int(record.laser_color_code):1d}",
        f"{record.launch_date:>8s}",
        f"{record.launch_time:>13s}",
        f"{int(record.light_time_raw):014d}",
        f"{int(record.reflector_id):1d}",
        f"{record.station_id:>5s}",
        _format_opt_int(record.number_of_returns, 3),
        f"{int(record.uncertainty_raw):06d}",
        _format_opt_int(record.signal_noise_ratio_raw, 3),
        (record.quality_code or " ")[:1],
        f"{int(record.pressure_raw):06d}",
        f"{int(record.temperature_raw):04d}" if record.temperature_raw >= 0 else f"{int(record.temperature_raw):4d}",
        f"{int(record.humidity_percent):02d}",
        f"{int(record.wavelength_raw):05d}",
        (record.version_code or " ")[:1],
        _format_opt_int(record.duration_s, 4),
        "  ",  # cols 79-80 unused
        f"{(record.source_format or ''):<9s}",
    ]
    line = "".join(parts)
    assert len(line) == MINI_LINE_FULL_LENGTH, f"internal error: MINI line length {len(line)}"
    return line


def write_mini_file(records: Sequence[MiniRecord], path) -> None:
    """Write MINI records to a fixed-width file (gzip if path ends in .gz)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    with opener(path, "wt", encoding="ascii", newline="\n") as fh:
        for record in records:
            fh.write(format_mini_line(record))
            fh.write("\n")
