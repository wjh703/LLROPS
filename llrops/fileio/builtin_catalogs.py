from __future__ import annotations

from llrops.fileio.catalogs import ReflectorRecord, StationRecord

# Station coordinates and velocities from the INPOP21a station table.
# Coordinates are ITRF XYZ in meters. Velocities are meters/year.
# IMPORTANT: If your source table defines a different reference epoch, change
# position_epoch_utc below. The default used here is J2000.0.
STATION_POSITION_EPOCH_UTC = "2000-01-01T00:00:00"

STATIONS = {
    "APOL": StationRecord(
        name="APOL",
        aliases=("APOLLO", "7045", "70610"),
        itrf_xyz_m=(-1463998.9079, -5166632.7663, 3435012.8921),
        itrf_velocity_m_per_year=(-0.0139, -0.0003, -0.0023),
        position_epoch_utc=STATION_POSITION_EPOCH_UTC,
    ),
    "APOLLO": StationRecord(
        name="APOLLO",
        aliases=("APOL", "7045", "70610"),
        itrf_xyz_m=(-1463998.9079, -5166632.7663, 3435012.8921),
        itrf_velocity_m_per_year=(-0.0139, -0.0003, -0.0023),
        position_epoch_utc=STATION_POSITION_EPOCH_UTC,
    ),
    "GRASSE": StationRecord(
        name="Grasse",
        aliases=("GRSM", "GRASSE", "7845", "01910"),
        itrf_xyz_m=(4581692.1686, 556196.0742, 4389355.1225),
        itrf_velocity_m_per_year=(-0.0151, 0.0193, 0.0114),
        position_epoch_utc=STATION_POSITION_EPOCH_UTC,
    ),
    "HALEAKALA": StationRecord(
        name="Haleakala",
        aliases=("HALE", "HALL", "HALEAKALA", "56610"),
        itrf_xyz_m=(-5466003.7272, -2404425.9189, 2242197.8916),
        itrf_velocity_m_per_year=(-0.0122, 0.0622, 0.0310),
        position_epoch_utc=STATION_POSITION_EPOCH_UTC,
    ),
    "MATERA": StationRecord(
        name="Matera",
        aliases=("MATM", "MATE", "MATERA", "07941"),
        itrf_xyz_m=(4641978.8100, 1393067.5310, 4133249.4800),
        itrf_velocity_m_per_year=(-0.0180, 0.0192, 0.0140),
        position_epoch_utc=STATION_POSITION_EPOCH_UTC,
    ),
    "MCDONALD": StationRecord(
        name="McDonald",
        aliases=("MDOL", "MCDONALD", "71110"),
        itrf_xyz_m=(-1330781.6134, -5328756.4702, 3235697.8262),
        itrf_velocity_m_per_year=(-0.0244, -0.0319, 0.0091),
        position_epoch_utc=STATION_POSITION_EPOCH_UTC,
    ),
    "MLRS1": StationRecord(
        name="MLRS1",
        aliases=("MLRS1", "71111"),
        itrf_xyz_m=(-1330121.0057, -5328532.3595, 3236146.0225),
        itrf_velocity_m_per_year=(-0.0124, 0.0009, -0.0053),
        position_epoch_utc=STATION_POSITION_EPOCH_UTC,
    ),
    "MLRS2": StationRecord(
        name="MLRS2",
        aliases=("MLRS2", "71112"),
        # itrf_xyz_m=(-1330021.1222, -5328401.8695, 3236480.7584),
        itrf_xyz_m=(-1330021.4931, -5328403.3401, 3236481.6472), # inpop19a
        # itrf_xyz_m=(-1330021.6061, -5328403.3158, 3236481.6069), # petrel19
        itrf_velocity_m_per_year=(-0.0121, 0.0015, -0.0036),
        position_epoch_utc=STATION_POSITION_EPOCH_UTC,
    ),
    "WETTZELL": StationRecord(
        name="Wettzell",
        aliases=("WETL", "WETTZELL", "08834"),
        itrf_xyz_m=(4075576.7721, 931785.5248, 4801583.5601),
        itrf_velocity_m_per_year=(-0.0139, 0.0170, 0.0124),
        position_epoch_utc=STATION_POSITION_EPOCH_UTC,
    ),
}

# Add common aliases and ILRS numeric station codes as direct catalog keys.
STATIONS["GRSM"] = STATIONS["GRASSE"]
STATIONS["7845"] = STATIONS["GRASSE"]
STATIONS["01910"] = STATIONS["GRASSE"]
STATIONS["7045"] = STATIONS["APOL"]
STATIONS["70610"] = STATIONS["APOL"]
STATIONS["HALE"] = STATIONS["HALEAKALA"]
STATIONS["56610"] = STATIONS["HALEAKALA"]
STATIONS["MATM"] = STATIONS["MATERA"]
STATIONS["07941"] = STATIONS["MATERA"]
STATIONS["MDOL"] = STATIONS["MCDONALD"]
STATIONS["71110"] = STATIONS["MCDONALD"]
STATIONS["71111"] = STATIONS["MLRS1"]
STATIONS["71112"] = STATIONS["MLRS2"]
STATIONS["08834"] = STATIONS["WETTZELL"]

# Lunar reflector coordinates in the DE440 PA frame, meters.
# REFLECTORS = {
#     "APOLLO11": ReflectorRecord(
#         name="Apollo 11",
#         aliases=("apollo11", "Apollo11", "A11", "AP11"),
#         moon_fixed_xyz_m=(1591967.049, 690698.573, 21004.461),
#         # moon_fixed_xyz_m=(1591966.6407, 690699.4669, 21003.7578),
#     ),
#     "APOLLO14": ReflectorRecord(
#         name="Apollo 14",
#         aliases=("apollo14", "Apollo14", "A14", "AP14"),
#         moon_fixed_xyz_m=(1652689.369, -520998.431, -109729.869),
#         # moon_fixed_xyz_m=(1652689.5625, -520997.5929, -109730.5181),
#     ),
#     "APOLLO15": ReflectorRecord(
#         name="Apollo 15",
#         aliases=("apollo15", "Apollo15", "A15", "AP15"),
#         moon_fixed_xyz_m=(1554678.104, 98094.498, 765005.863),
#         # moon_fixed_xyz_m=(1554678.3071, 98095.5262, 765005.2077),
#     ),
#     "LUNOKHOD2": ReflectorRecord(
#         name="Lunokhod 2",
#         aliases=("lunokhod2", "Luna21", "L2", "LUNA21", "luna21"),
#         moon_fixed_xyz_m=(1339363.598, 801870.995, 756359.260),
#         # moon_fixed_xyz_m=(1339363.3937, 801871.9437, 756358.6633),
#     ),
#     "LUNOKHOD1": ReflectorRecord(
#         name="Lunokhod 1",
#         aliases=("lunokhod1", "Luna17", "L1", "LUNA17", "luna17"),
#         moon_fixed_xyz_m=(1114291.452, -781299.273, 1076059.049),
#         # moon_fixed_xyz_m=(1114292.2303, -781298.4355, 1076058.6227),
#     ),
# }

# inpop21a for test
REFLECTORS = {
    "APOLLO11": ReflectorRecord(
        name="Apollo 11",
        aliases=("apollo11", "Apollo11", "A11", "AP11"),
        moon_fixed_xyz_m=(1591966.6407, 690699.4669, 21003.7578),
    ),
    "LUNOKHOD1": ReflectorRecord(
        name="Lunokhod 1",
        aliases=("lunokhod1", "Luna17", "L1", "LUNA17", "luna17"),
        moon_fixed_xyz_m=(1114292.2303, -781298.4355, 1076058.6227),
    ),
    "APOLLO14": ReflectorRecord(
        name="Apollo 14",
        aliases=("apollo14", "Apollo14", "A14", "AP14"),
        moon_fixed_xyz_m=(1652689.5625, -520997.5929, -109730.5181),
    ),
    "APOLLO15": ReflectorRecord(
        name="Apollo 15",
        aliases=("apollo15", "Apollo15", "A15", "AP15"),
        moon_fixed_xyz_m=(1554678.3071, 98095.5262, 765005.2077),
    ),
    "LUNOKHOD2": ReflectorRecord(
        name="Lunokhod 2",
        aliases=("lunokhod2", "Luna21", "L2", "LUNA21", "luna21"),
        moon_fixed_xyz_m=(1339363.3937, 801871.9437, 756358.6633),
    ),
}


# ILRF for test
# REFLECTORS = {
#     "APOLLO11": ReflectorRecord(
#         name="Apollo 11",
#         aliases=("apollo11", "Apollo11", "A11", "AP11"),
#         moon_fixed_xyz_m=(1591966.745, 690699.384, 21003.764),
#     ),
#     "LUNOKHOD1": ReflectorRecord(
#         name="Lunokhod 1",
#         aliases=("lunokhod1", "Luna17", "L1", "LUNA17", "luna17"),
#         moon_fixed_xyz_m=(1114292.301, -781298.502, 1076058.718),
#     ),
#     "APOLLO14": ReflectorRecord(
#         name="Apollo 14",
#         aliases=("apollo14", "Apollo14", "A14", "AP14"),
#         moon_fixed_xyz_m=(1652689.627, -520997.633, -109730.514),
#     ),
#     "APOLLO15": ReflectorRecord(
#         name="Apollo 15",
#         aliases=("apollo15", "Apollo15", "A15", "AP15"),
#         moon_fixed_xyz_m=(1554678.397, 98095.451, 765005.257),
#     ),
#     "LUNOKHOD2": ReflectorRecord(
#         name="Lunokhod 2",
#         aliases=("lunokhod2", "Luna21", "L2", "LUNA21", "luna21"),
#         moon_fixed_xyz_m=(1339363.512, 801871.855, 756358.706),
#     ),
# }
