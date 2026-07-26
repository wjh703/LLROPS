# Changelog
All notable changes to the `IERS Conventions` will be documented in this file.
The first documented modifications will be for v1.2.0.

This file is based on [Keep a Changelog](http://keepachangelog.com/)
conventions, and is formatted in Markdown syntax.

The IERS Conventions uses incremental sequence increases to indicate the 
current version following the pattern vMajor.Update.Minor.

*Major* will be incremented when a new official IERS Conventions is released.
(The official IERS Conventions 2010 software release will be considered version 1.0.0)

*Update* will be incremented when substantial or significantly consequential updates
are introduced to the original Major version.

*Minor* will be incremented when minor updates or bug fixes are introduced to the
Update or Major version. 

### Modifications
- All dates will following the `DD-MM-YYYY` or `DAY MONTH YEAR` format.
- Changes to a given version are indicated by the heading level `##`. 
- Versions are list in reverse-chronological order (newest on top).
- Modifications are grouped according to the following categories:
  - *Added* for new content or software.
  - *Changed* for changes or updates in software and chapter text.
  - *Fixed* for any bug fixes in software or text.
  - *Removed* for content removed in this release.
  - *Deprecated* for content that will be removed in upcoming releases.
  - *Known Issues* for concepts, values, or software that are known to be
                   problematic or require updating.

---

## v1.3.0 (released 1 April 2019)
### Changed
- **Chapter 4** has been updated to be consistent with _ITRF2014_. 

## v1.2.0 (released 28 December 2018)
### Added
- Definition of _secular pole_ to the **Glossary**.

### Changed
- The Conventions has replaced the concept of the _mean pole_ with the 
_secular pole_ in **Chapters 6 & 7**.
- Definition of _mean pole_ in the **Glossary**.

### Removed
- `IERS_CMP_2015.F` from **Chapter 7**. Software has been removed as the
Conventions no longer supports the _mean pole_. 

### Fixed
- `CALC2JD.F` subroutine has been renamed to correspond to subroutine 
called in `DAT.F` (**Chapter 7**).


## v1.1.0 (released 17 November 2017)
### Changed
- The `W0` value in Table 1.1 of **Chapter 1** has been modified to correspond
  to the value adopted in the IAG 2015 Resolution (No. 1).


## v1.0.0 (released 2010)
### Added
- All software and text associated with the official IERS Conventions (2010).

