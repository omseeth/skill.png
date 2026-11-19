# Vendored upstream T1 code

A subset of the reference implementation of T1, the runtime pieces the T1
adapter needs: the tools and their schemas, the tool-config renderer, the
ontology constants the restaurant tools import, the evaluation metrics, and the
planner that renders upstream's prompt.

- Source: https://github.com/CapitalOne-Research/T1
- Commit: `22a16b7570a5c20b1dc05f22f78a82ab94488a00` (2025-10-16 20:39:29 +0000),
  HEAD of the default branch on 2026-09-26.
- Layout: `upstream/` is upstream's `src/t1/`, with the same relative paths.

## License

Upstream's `LICENSE` file is the Apache License 2.0 and is copied here
unchanged as `LICENSE`. Upstream's `pyproject.toml` declares
`license = {text = "MIT"}`, which contradicts it; the LICENSE file is what is
kept. The data the tools read (the ontology) is not vendored: it is CC-BY-4.0
and is downloaded from https://huggingface.co/datasets/capitalone/T1 at a
pinned revision.

## Files and the sha256 of each upstream original

| upstream path | sha256 |
|---|---|
| `LICENSE` | `43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1` |
| `src/t1/__init__.py` | `a32bfadf037b94de6eb056c2a37749ed80d33b29051c9c220c5ba044aaa21ca0` |
| `src/t1/tools/adjust_date.py` | `969ba900e6ad2900e53032a2cd78d2801d6ea98c36390d3eaa054cf6201c2aac` |
| `src/t1/tools/cache.py` | `62d4362d04aae1d79f9ead685fb6135efad9ab2f0df7f03d7412b3a172070b1c` |
| `src/t1/tools/filter_attractions.py` | `ceae14aeb62f5e0da1c74fcdf72d2f40cc1daf2da3ca30211a0c8cdc6d674b54` |
| `src/t1/tools/filter_flights.py` | `e88168f94fbdc115fb83fbffdeeca5d473fa9429b7410dbd42c35db223c3eed6` |
| `src/t1/tools/filter_hotels.py` | `6dcbc3158be805ce4e7d12ab447f38a99b7805b943ee083fcdca2fd9deb0c556` |
| `src/t1/tools/filter_restaurants.py` | `2b1ceb044769d91f10f8ef99b168b7d510efe1ff961b6c367b4ad8e5f0562a43` |
| `src/t1/tools/find_nearest.py` | `8ea5ccc47c9cc2b1d277117ef0e23c428b62e80639186f66def486540fbce8c0` |
| `src/t1/tools/search_attractions.py` | `4652509349d1a1118e9111a438569ade63f436c120a05ba568add6e70dca239b` |
| `src/t1/tools/search_flights.py` | `76000ec1d1edea9499fdbf13c1fd6afca007a785ed96d0bd7c593e693b4a8d9b` |
| `src/t1/tools/search_hotels.py` | `b9e2cf30c9c2f73036f9980389f139755bb827075b377ac1fd61bc56e66942aa` |
| `src/t1/tools/search_restaurants.py` | `e79bd1a1ffb0d63964f59eecd3058c8e67bdcbd84c6739fa1da968d35660f586` |
| `src/t1/tools/seek_information.py` | `872916818cd1e4a1a2bee79ee8d51d2f9c930723dbd3814db003e41e68bd1e23` |
| `src/t1/tools/sort_results.py` | `da213a09f62dba20754dfca7d8d0eb68e12d1406262e4e6eb5b56a3dbce54c14` |
| `src/t1/tools/schema/adjust_date.py` | `ea69787a0396667c689ff4194dc30f7c5d24d47f674b0fe98ba820f43b5d3247` |
| `src/t1/tools/schema/cache.py` | `51fa209ada36f473fe3bc36452d154dc0aee952c5fc0ed5d7c64e524b98fcec2` |
| `src/t1/tools/schema/filter_attractions.py` | `ffe04f850874ae05eb215dd4c0d28a88edb191800f6eeab8458f50df16a12aea` |
| `src/t1/tools/schema/filter_flights.py` | `1bf62c1954ecc4aef79bbc05481bc496116cbda706dad2a336e26dec780bd71c` |
| `src/t1/tools/schema/filter_hotels.py` | `41818f97a8adfbc078b6bbfaf51c753f51c2f274515965b84f1ff2ffbf8c68bf` |
| `src/t1/tools/schema/filter_restaurants.py` | `9c89399ac142fba902e306961280e5e86b145d20e5eed6ff1314483c4b649cd6` |
| `src/t1/tools/schema/find_nearest.py` | `37a484cde6326f5312eea9a6e0610d2c69ece9dcd63b473255c4695116e5b8da` |
| `src/t1/tools/schema/search_attractions.py` | `0dca41e11fe1975510c940d050994a851eb1e731639aaee17271553f36fbc131` |
| `src/t1/tools/schema/search_flights.py` | `5681bc57aeb744076d8e57b259d5257e4ef010d1152bb51c4a9f804bb458d602` |
| `src/t1/tools/schema/search_hotels.py` | `b97a3d01fab038ab114800544b5d1002feffa728b765911160837d28bc7c8b39` |
| `src/t1/tools/schema/search_output.py` | `be6ff3df490066a9c29e404a3fdb25dcac465e57af635a36fbd401212d298309` |
| `src/t1/tools/schema/search_restaurants.py` | `c892c43ce5733882456a59a2f633a1e84b10bed8aa262bc1297789768e551f75` |
| `src/t1/tools/schema/sort_results.py` | `81501333cf7ff09f9193f3492afa2222b5fd6c7d7fcd5ca25c1dae8f1a57c4a8` |
| `src/t1/tools/utils/get_tool_configurations.py` | `6284c8e9e40eadc0ebfa662e72e336bb64ecc124da1fa6d7ac5d4d295d98052a` |
| `src/t1/tools/utils/output_templates.py` | `7f5c311efd0d78949599d1754d11fa79ac36d5e36b9218e6a896d58485a1d08a` |
| `src/t1/data/ontology/attraction.py` | `a3500079045d2ba876553b13bdd70ba6f080653ca0480e667947d877f3c7f8e3` |
| `src/t1/data/ontology/city.py` | `2e24e15cae996d3fafd352dc9ae59c24dd3bdbbfcb1ea668d3e56cfc22d44a71` |
| `src/t1/data/ontology/flight.py` | `567bf0c0e46e0ef42157fd7e15d4c1e8b9cbd02ced9118cb0337a03166d383ce` |
| `src/t1/data/ontology/hotel.py` | `86de997fe45b8d57f435e60b79535eea432be1476035e5ce9cd07d2cc015ab34` |
| `src/t1/data/ontology/restaurant.py` | `04c9a2a9035951d2583c0d5df3279d27ceb05c69f56967ed56894a7df2a1496f` |
| `src/t1/evaluation/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `src/t1/evaluation/eval_metrics.py` | `76c2786a6ba12948ede810cd41772046223ce046a98437c112038518ef59683b` |
| `src/t1/planner/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `src/t1/planner/planner_code.py` | `84de976245c825601bf73cb0ee20f51b71a0ced940c24c308259bdc44feb4a27` |
| `src/t1/planner/planner_utils.py` | `fa17e4329b6ff0b383dc6836ec1964bbd2a3dc086b94bba5c21a72c3f7a49b5d` |

Not vendored: `tools/test_tools.py`, `t1.py`, `cli.py`, `utils.py`, `__main__.py`,
and everything outside `src/t1/`. `inference.py` and `evaluation/*.py` at the
repository root stay in the clone (`baselines/t1/code/`), where the parity
tests run them.

## Local modifications

Only these, and nothing inside a class body: `inspect.getsource` pastes
`SearchResultsOutputModel` into the prompt.

1. Every `from t1.` import is rewritten to `from skillpng.benchmarks.t1.upstream.`.
   All of them are module-level lines; there is no `import t1.` form.
2. Empty `__init__.py` files are added where upstream has namespace packages:
   `tools/`, `tools/schema/`, `tools/utils/`, `data/`, `data/ontology/`.
3. In `planner/planner_code.py`, the two lines `from google import genai` and
   `from transformers import AutoTokenizer` are deleted. Neither name is used
   by what is vendored.
4. Every file that 1 or 3 changed starts with this notice and a blank line, as
   Apache-2.0 section 4(b) requires of modified files:

   ```python
   # Modified by the Skill.png authors from T1 at commit 22a16b7 (Apache-2.0);
   # the changes are listed in skillpng/benchmarks/t1/upstream/VENDORED.md.
   ```

   Files that 1 and 3 leave alone carry no notice and stay byte-identical.

`tests/test_t1_protocol.py` checks, whenever the clone is present, that every
file here equals the clone's file after exactly these modifications.

## Reproduce

From the repository root, with the clone at the pinned commit:

```bash
git clone https://github.com/CapitalOne-Research/T1 baselines/t1/code
git -C baselines/t1/code checkout 22a16b7570a5c20b1dc05f22f78a82ab94488a00
SRC=baselines/t1/code/src/t1
DST=src/skillpng/benchmarks/t1/upstream
mkdir -p $DST/tools/schema $DST/tools/utils $DST/data/ontology $DST/evaluation $DST/planner
cp $SRC/__init__.py $DST/__init__.py
for f in $SRC/tools/*.py; do [ "$(basename "$f")" = test_tools.py ] || cp "$f" $DST/tools/; done
cp $SRC/tools/schema/*.py $DST/tools/schema/
cp $SRC/tools/utils/get_tool_configurations.py $SRC/tools/utils/output_templates.py $DST/tools/utils/
cp $SRC/data/ontology/*.py $DST/data/ontology/
cp $SRC/evaluation/__init__.py $SRC/evaluation/eval_metrics.py $DST/evaluation/
cp $SRC/planner/__init__.py $SRC/planner/planner_utils.py $SRC/planner/planner_code.py $DST/planner/
cp baselines/t1/code/LICENSE $DST/LICENSE
touch $DST/tools/__init__.py $DST/tools/schema/__init__.py $DST/tools/utils/__init__.py \
      $DST/data/__init__.py $DST/data/ontology/__init__.py
find $DST -name '*.py' -exec sed -i.bak -E 's/^(from|import) t1\./\1 skillpng.benchmarks.t1.upstream./' {} +
sed -i.bak -e '/^from google import genai$/d' -e '/^from transformers import AutoTokenizer$/d' \
    $DST/planner/planner_code.py
find $DST -name '*.bak' -delete
for f in $(cd $DST && find . -name '*.py'); do
  if [ -f "$SRC/$f" ] && ! cmp -s "$SRC/$f" "$DST/$f"; then
    { printf '%s\n' "# Modified by the Skill.png authors from T1 at commit 22a16b7 (Apache-2.0);" \
        "# the changes are listed in skillpng/benchmarks/t1/upstream/VENDORED.md." ""
      cat "$DST/$f"; } > "$DST/$f.new"
    mv "$DST/$f.new" "$DST/$f"
  fi
done
```

`sed -i.bak` works with both BSD and GNU sed.

## Lint and type checks

`upstream/` is excluded from ruff (`extend-exclude`) and from strict mypy
(`exclude`, plus `ignore_errors` for `skillpng.benchmarks.t1.upstream.*`, which
mypy still follows through the adapter's import of `eval_metrics`). `make fmt`
leaves it untouched.

## Import rule

Only `evaluation/eval_metrics.py`, pure functions, is imported in the main or
the pytest process. Seven tool modules read the 1.35 GB ontology at import and
keep it in module globals, and `planner_code` imports every tool and prints at
import, so both are imported only inside the executor's worker
(`t1_exec.py`) and inside `t1_build.py`.
