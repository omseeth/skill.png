# T1 test fixture

`ontology/` holds the header and the first 60 rows of each of the eight
ontology CSVs the T1 tools read, with the dataset's directory layout, so the
vendored tools import and run in a test subprocess in well under a second.

Source: the T1 dataset by Capital One, https://huggingface.co/datasets/capitalone/T1,
revision `7a3e58355959426e3daeecc410c2b87139295140`, from
Chakraborty et al., "T1: A Tool-Oriented Conversational Dataset for Multi-Turn
Agentic Planning", NeurIPS 2025 Datasets and Benchmarks.

License: Creative Commons Attribution 4.0 International (CC BY 4.0),
https://creativecommons.org/licenses/by/4.0/. Changes: every file is cut to its
first 60 data rows; nothing else is altered.

Rebuild with, from the repository root:

```bash
SRC=$(python -c "from skillpng.benchmarks.t1.t1_build import ontology_root; print(ontology_root())")
for f in single-domain/t1_flight_data.csv single-domain/t1_hotel_data.csv \
         single-domain/t1_restaurant_data.csv single-domain/t1_attraction_data.csv \
         multi-domain/t1_hotel_attraction_data.csv multi-domain/t1_hotel_restaurant_data.csv \
         multi-domain/t1_restaurant_attraction_data.csv miscellaneous/t1_all_airports.csv; do
  head -n 61 "$SRC/ontology/$f" > "tests/fixtures/t1/ontology/$f"
done
```
