project(
  'llrops',
  'c',
  version: '0.1.0',
  meson_version: '>=1.2.3',
  default_options: ['warning_level=2'],
)

add_languages('fortran', native: false)

py = import('python').find_installation(pure: false)
numpy_dep = dependency('numpy')

f2py_include = run_command(
  py,
  ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'],
  check: true,
).stdout().strip()
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

iers_signature = files('external/iers2010/bindings/iers2010.pyf')
iers_fortran_sources = files(
    'external/iers2010/src/FCUL_A.F',
    'external/iers2010/src/FCUL_ZD_HPA.F',
    'external/iers2010/src/ORTHO_EOP.F',
    'external/iers2010/src/CNMTX.F',
    'external/iers2010/src/PMSDNUT2.F',
    'external/iers2010/src/UTLIBR.F',
    'external/iers2010/src/FUNDARG.F',
)
iers_bindings = custom_target(
  '_iers2010module.c',
  input: iers_signature,
  output: ['_iers2010module.c', '_iers2010-f2pywrappers.f'],
  command: [py, '-m', 'numpy.f2py', '@INPUT@', '--lower'],
)

py.extension_module(
  '_iers2010',
  [iers_fortran_sources, iers_bindings, f2py_support],
  dependencies: numpy_dep,
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

iers_source_install_dir = join_paths(
  py.get_install_dir(),
  'llrops',
  '_external',
  'iers2010',
  'src',
)
install_data(
  iers_fortran_sources,
  install_dir: iers_source_install_dir,
)
