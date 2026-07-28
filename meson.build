project(
  'llrops',
  'c',
  version: '0.1.0',
  meson_version: '>=1.2.3',
  default_options: ['warning_level=2'],
)

add_languages('fortran', native: false)

py = import('python').find_installation(pure: false)
py_dep = py.dependency()
numpy_dep = dependency('numpy')

f2py_include = run_command(
  py,
  ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'],
  check: true,
).stdout().strip()

# Keep generated build inputs outside an in-repository virtual environment.
f2py_support = custom_target(
  'f2py-support',
  output: ['fortranobject.c', 'fortranobject.h'],
  command: [
    py,
    '-c',
    'import pathlib, shutil, sys; source = pathlib.Path(sys.argv[1]); shutil.copyfile(source / "fortranobject.c", sys.argv[2]); shutil.copyfile(source / "fortranobject.h", sys.argv[3])',
    f2py_include,
    '@OUTPUT0@',
    '@OUTPUT1@',
  ],
)

iers_fortran_sources = files(
  'external/iers2010/src/FCUL_A.F',
  'external/iers2010/src/FCUL_ZD_HPA.F',
  'external/iers2010/src/ORTHO_EOP.F',
  'external/iers2010/src/CNMTX.F',
  'external/iers2010/src/PMSDNUT2.F',
  'external/iers2010/src/UTLIBR.F',
  'external/iers2010/src/FUNDARG.F',
  'external/iers2010/src/DEHANTTIDEINEL.F',
  'external/iers2010/src/CAL2JD.F',
  'external/iers2010/src/DAT.F',
  'external/iers2010/src/NORM8.F',
  'external/iers2010/src/SPROD.F',
  'external/iers2010/src/ST1IDIU.F',
  'external/iers2010/src/ST1ISEM.F',
  'external/iers2010/src/ST1L1.F',
  'external/iers2010/src/STEP2DIU.F',
  'external/iers2010/src/STEP2LON.F',
  'external/iers2010/src/ZERO_VEC8.F',
  'external/iers2010/src/ADMINT.F',
  'external/iers2010/src/ETUTC.F',
  'external/iers2010/src/EVAL.F',
  'external/iers2010/src/JULDAT.F',
  'external/iers2010/src/LEAP.F',
  'external/iers2010/src/MDAY.F',
  'external/iers2010/src/RECURS.F',
  'external/iers2010/src/SHELLS.F',
  'external/iers2010/src/SPLINE.F',
  'external/iers2010/src/TDFRPH.F',
  'external/iers2010/src/TOYMD.F',
  'external/iers2010/src/HARDISP_WRAP.F',
)

iers_bindings = custom_target(
  'iers2010-f2py-bindings',
  input: 'external/iers2010/bindings/iers2010.pyf',
  output: ['_iers2010module.c', '_iers2010-f2pywrappers.f'],
  command: [py, '-m', 'numpy.f2py', '@INPUT@', '--lower'],
)

py.extension_module(
  '_iers2010',
  [iers_fortran_sources, iers_bindings, f2py_support],
  dependencies: [py_dep, numpy_dep],
  subdir: 'llrops',
  install: true,
)

python_sources = run_command(
  py,
  [
    '-c',
    'import pathlib, sys; root = pathlib.Path(sys.argv[1]); print("\\n".join(str(path.relative_to(root)) for path in sorted(root.joinpath("llrops").rglob("*.py"))))',
    meson.project_source_root(),
  ],
  check: true,
).stdout().strip().split('\n')
py.install_sources(python_sources, pure: false, preserve_path: true)

install_data(
  iers_fortran_sources,
  install_dir: join_paths(
    py.get_install_dir(),
    'llrops',
    '_external',
    'iers2010',
    'src',
  ),
)
install_data(
  'external/iers2010/LICENSE',
  install_dir: join_paths(
    py.get_install_dir(),
    'llrops',
    '_external',
    'iers2010',
  ),
)
