# Custom window input context

DBSCAN needs the SQL partition boundaries and absolute input row for each custom
window callback. The DuckDB revision that this branch started from does not expose
them. `custom-window-row-context.patch` contains the companion DuckDB API change;
it does not alter the existing callback signatures.

The submodule includes this change for local testing. Before upstreaming Spatial,
submit the companion patch to DuckDB and replace the local submodule revision and
CI's `duckdb_version` with a published revision containing it. Do not publish an
extension compiled against this patch for an unpatched DuckDB release: the C++
window input layout has changed.

For a checkout of the original DuckDB revision, apply the patch explicitly:

```sh
git -C duckdb apply ../patches/duckdb/custom-window-row-context.patch
```

Keep Linux and macOS build directories and vcpkg executables separate. CMake
caches contain absolute paths and cannot be moved between those environments.
