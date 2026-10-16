Rook-contiguity weights for Cao & Dowd's Proposition 99 panel
==============================================================

cd_rook_W.csv is a binary 51 x 51 adjacency matrix for the 50 U.S. states and
the District of Columbia, keyed by the postal codes of load_california_cd(51).
An entry is 1 when two units share a boundary segment (rook contiguity; a
shared corner does not count). Alaska and Hawaii have no neighbours.

The article docs/articles/cd-results.qmd uses it to fit the sar model on the
Cao-Dowd panel. It is not part of the package: load_california_cd() stays a
cd panel without spatial weights.

Source and rule: the same as the bundled 39-state California weights.
  - U.S. Census Bureau, TIGER/Line 2024 state shapefile
    https://www2.census.gov/geo/tiger/TIGER2024/STATE/tl_2024_us_state.zip
    SHA-256 of the files used:
      tl_2024_us_state.shp 5eca8051ec802d987cb9bf67e899980a35b4ab7abb20084def7b9bbad8fe50e7
      tl_2024_us_state.dbf 746b3338a34df7a872e9eef9305c837b1e7e079392d7faa7b45e7183eaf81104
      tl_2024_us_state.shx 690f7f029c3b30648635cb8d1ddb9756c510fda50a1eb98fc0d258b693f0815c
      tl_2024_us_state.prj 0b9041e921d9ebb43247d314608fe9e38a0b008ee793067fc1806199ea1fb9dd
  - rook contiguity, as R's spdep::poly2nb(queen = FALSE), computed with
    libpysal's Rook weights (geopandas 1.2.0, libpysal 4.15.0).

Built by tools/build_cd_contiguity.py, which first checks that the matrix
restricted to the 39 states of load_california() reproduces the bundled
california_W_matrix.csv and california_w_vector.csv exactly.

cd_rook_W.csv SHA-256:
  f1b70249c288a0b28da1ed09a4b0359139c6409be1dc9bfde55d733ce87ce70e
