# Vendored upstream SkillOpt code

The parts of SkillOpt that `baselines/skillopt/train.py` runs: the trainer
and everything it imports, upstream's generic prompts, the SearchQA and
LiveMathematicianBench analyst prompts, and the default recipe. Upstream's
blank SearchQA skill (`skillopt/envs/searchqa/skills/initial.md`) is not
vendored; it is `baselines/skillopt/seeds/searchqa.md`, next to the other
benchmarks' seeds.

- Source: https://github.com/microsoft/SkillOpt
- Commit: `32deaeb50f92fc2f3a225bbf67d3eb3a604ce8b6` (2026-10-06 16:04:28 +0000).
- Layout: the same relative paths as upstream's repository root.
- Imports: unchanged. `train.py` and `adapter.py` put this directory first on
  `sys.path`, so `import skillopt` resolves here.

Vendored rather than installed: the package installs a top-level `scripts`
package (and `skillopt_sleep`, `skillopt_webui`) into site-packages, and it
does not ship `configs/_base_/default.yaml`, the default hyperparameters. The
one third-party import of the code here, `openai`, is the `skillopt` extra
in `pyproject.toml`.

## License

MIT; upstream's `LICENSE` is copied unchanged.

## Files and the sha256 of each upstream original

Every file is byte-identical to upstream's at the commit above.

| upstream path | sha256 |
|---|---|
| `LICENSE` | `9515a04280cf50ce154eb3ab8fa375c7cc5e6b45ea5cb23b7ff3830737020c15` |
| `configs/_base_/default.yaml` | `d008b75a8ab1c70abfd77f678c3f4b84e9c3f6bd3392de072fa8adab7387f222` |
| `skillopt/__init__.py` | `acde34f209004e93450198fbcc4cf5f161e5079c53e5e38492d934e42347995c` |
| `skillopt/config.py` | `7614bc3d3e40738a7d63b3ce679393471a3c909884f4f8da397e0b7848561cb3` |
| `skillopt/datasets/__init__.py` | `39a907e5207f3daa2477a4b3f9b01395fbadd43918f22ecdb8d4c132958467e7` |
| `skillopt/datasets/base.py` | `82a68f80271fadb4947f49b81a27cf03af08af61ed85d5326381a7955d0da48b` |
| `skillopt/engine/__init__.py` | `ef54fb8adc9731d16ec85321959437a41df1ba9c88114a424fa4da12df2a82db` |
| `skillopt/engine/trainer.py` | `1f135ebe8c90306c1d1801edc16becd5862e8d7ba5e2b6f0986b8001086b2560` |
| `skillopt/envs/__init__.py` | `a3e66f90175f614a239a4f1264569cefdc6f8c7c62f4204986119d9c8f605621` |
| `skillopt/envs/base.py` | `ea8c650e8bce460be047a03183cf17fbc49b436345cd1b1484093f763c6c64d7` |
| `skillopt/envs/livemathematicianbench/prompts/analyst_error.md` | `9cd5e3449b9013fb67b6dec930486bd8c23c7128f2746f97a1ca29b07b178150` |
| `skillopt/envs/livemathematicianbench/prompts/analyst_success.md` | `42619d41000d16b7e5194e5dd09ba67fcd65d1c5ae8b04b4097c38e45a02750d` |
| `skillopt/envs/searchqa/prompts/analyst_error.md` | `ac58184f03e2198244ace28a9339674256061a62433787aed358954e6a031a19` |
| `skillopt/envs/searchqa/prompts/analyst_success.md` | `0a169f33a141ed544b460d3f800f9dd2da28450d42b93556a3656cf8a45f0322` |
| `skillopt/evaluation/__init__.py` | `1d34dd85f77152bca307443ec0ad4609cdc0f8fb0d68638d1e1235d2e5bbc854` |
| `skillopt/evaluation/gate.py` | `3c2405f13288e97f53afa0bdfc0752ba6917c51edd1904a810deb4b5a1a81bef` |
| `skillopt/gradient/__init__.py` | `570b2e37e2ca7a82b76ee0a0054cd8ecd849a55552d83dab1f7029be099fc43d` |
| `skillopt/gradient/aggregate.py` | `332d1aa261e8a0969a95d327b67436c153bba74dde27f312c650eae3cff5cd39` |
| `skillopt/gradient/reflect.py` | `e58205c4b13ee7c565a214a01e8dcde6076c7cccbfa8240bd68d4c3481e1e1c5` |
| `skillopt/model/__init__.py` | `cc0c81b9ce4189c2060c7ef7cdf8207451f73d5e367130701b76ff61d553d0ce` |
| `skillopt/model/azure_openai.py` | `0a726c842e4fafad169e0b7d8552624c9b071cde96512d80e29db4670cc6a12a` |
| `skillopt/model/backend_config.py` | `623906322dc6495bb39881f455befe6f1463c0574d953882b7b1af535da44066` |
| `skillopt/model/claude_backend.py` | `a3aca64462c40376bb72abd7c7498a4645c57803afb4b912630c2f28c4174fc6` |
| `skillopt/model/claude_code_backend.py` | `e4ce7ee56cb85de3664079ea6e813240e8c681a72412cd7a5b181e3a090dae35` |
| `skillopt/model/codex_backend.py` | `9b55fae2ea02071d37b1cc23043f769c15ad33a2bfc055c464cf317d50d1d93a` |
| `skillopt/model/codex_harness.py` | `7b873d1893c423958c3dafbb674f137e9e29d3e25ece654521e06ac66db67e33` |
| `skillopt/model/common.py` | `436b54b3faa418a533477645e650d6cd08384645dbab0dc6b23199658ba21780` |
| `skillopt/model/copilot_backend.py` | `51aa26ae359905f81578dec4dd402f7e3c7e4e862f1fcbc167baf3f106fbc6af` |
| `skillopt/model/minimax_backend.py` | `9bcf6d792d25f27000b263a41cae49a97cbc4b74f0dd2addb6f31104fa6e84f9` |
| `skillopt/model/openai_compatible_backend.py` | `f262160f2851c6f74d7d95c14d354a0fea417e5e65e91ee65c03b53a8a6cfae8` |
| `skillopt/model/qwen_backend.py` | `1c7b0886457325dc8a20083c76f7a700ce83e300a1c7085e110d3482c85bba13` |
| `skillopt/model/router.py` | `3265614f352cd0dd665f39eb351278351b0f3c98f7dc3cde9a6b4b005015e5fb` |
| `skillopt/optimizer/__init__.py` | `5f18d2f2fbcd469ef7ff64705fbb6b883102b28cccee7353af21e8bb36b8749b` |
| `skillopt/optimizer/appendix.py` | `220c373fd1e0bb14f47b99a51ff7c712dfc9387c6107f4fae3f89f9102ca4f9c` |
| `skillopt/optimizer/clip.py` | `0c1060799884eb32e131000299ae78af094ee375d9c81c3dd1e1e2c78bd3c728` |
| `skillopt/optimizer/lr_autonomous.py` | `8ce459602b548c495825af9fba9621fcb5066b798876cdd0306b5d4ed48d2b68` |
| `skillopt/optimizer/meta_skill.py` | `a9b02bded86aca1575bf8647d85c55808fbe12162b5de69ca91c3ef2b7424b6d` |
| `skillopt/optimizer/rewrite.py` | `d5606e7859d9a36e34b3d870fd1d9f8172f3cdc97cfd2160610c3fed2de3f158` |
| `skillopt/optimizer/scheduler.py` | `791b81b6d7e3de974d6a9d4a9e69636bf2a442019a19c039dc7e2e0ff26b894a` |
| `skillopt/optimizer/select.py` | `e675ebfe4c79627c944e6aa90de6542f85deaf948054c1cfe1b8876ec8442dab` |
| `skillopt/optimizer/skill.py` | `1cb3ecd9da7ab5ea89ad86780f3e4536fcf5cda84455ea6ae8fcc9c86cc73d3f` |
| `skillopt/optimizer/skill_aware.py` | `ea37beb95056ac52a61167153a2e05b05692d55fbe2b3bcfb039516801f0b418` |
| `skillopt/optimizer/slow_update.py` | `2255036ffc558db997a2d3ed3b916ca5563db5406ac9144681657d847506af28` |
| `skillopt/optimizer/update_modes.py` | `37c15553afd820a94216caadfe80d29a9cbfca7c01dacee3b628e439830d8841` |
| `skillopt/prompts/__init__.py` | `50c017a18c35d990bf24d5efa4019aaa13bb3ef8db0e2b50dcab5c923a376455` |
| `skillopt/prompts/analyst_error.md` | `40eaf561c7fdbdfc1c7a1febd9f7c3540270ca6d5e7e0a8798c7c90b8d303200` |
| `skillopt/prompts/analyst_error_full_rewrite.md` | `a81389ca63e75f13d6aca4d2f01cd6825a364e9cca833f316de125014a2f7365` |
| `skillopt/prompts/analyst_error_rewrite.md` | `c393358b2be63d49bc8400944f17bcfb45f58aed5959cf6a46c2e1c1caefcd1b` |
| `skillopt/prompts/analyst_success.md` | `0ffc8f85e13af2b14e1b9f3307a44d30ecf60e9d803ccb9190fdd9785e43fabc` |
| `skillopt/prompts/analyst_success_full_rewrite.md` | `359a7481e348a89d51e5c7ee60a3d4d1d7a5ce7fdd8193d490c64f2667aeec55` |
| `skillopt/prompts/analyst_success_rewrite.md` | `758bfe107d15ba1b5318f7191ea9e965fb09982b12599d62fee9db0d823f22f0` |
| `skillopt/prompts/lr_autonomous.md` | `667c78c7873a28fae0e6f47c5c6ed556d76dfe337c74da986757c9e4fadb1f0e` |
| `skillopt/prompts/merge_failure.md` | `071313895da226840164ccda6e7f01d5f849ea128037d3f337cddab66ae799db` |
| `skillopt/prompts/merge_failure_full_rewrite.md` | `adef4b6e8f0b1d7cf992e4bd9a34f633375b122cc054477c305a228627ec79ae` |
| `skillopt/prompts/merge_failure_rewrite.md` | `e9cd2289d8515bca4561e52d3c8da14a2fe97d1b2e77d50f0d34f2fb257326c1` |
| `skillopt/prompts/merge_final.md` | `b2c7b452032fbbe3b223e877b766ea6202775d685c3b32e68956a402e775b84b` |
| `skillopt/prompts/merge_final_full_rewrite.md` | `21ee282d5720dd07dcea9ddd3cce2de893adeeed51518a0c241590130f1635bd` |
| `skillopt/prompts/merge_final_rewrite.md` | `6a79e19eebe2fbbe20f93a6b864c7c157388bbae168980154c0cdac452bbdfe4` |
| `skillopt/prompts/merge_success.md` | `6b816bcf5526d31164b779921f3c8c847541bf92fbe26c6a38b555eb3b753712` |
| `skillopt/prompts/merge_success_full_rewrite.md` | `8f1339cd57ed41af82f8ac7675891a9a58412b5c7ef4e6762a12ad8caad347ca` |
| `skillopt/prompts/merge_success_rewrite.md` | `2b293d9a16da199cade8bdade560b12553577ad66aacb0f9de3ae979aa150ab7` |
| `skillopt/prompts/meta_skill.md` | `b2bab7f7635cd2a94348de6adf15ceaa247ddabd0a2a8bb7c53eab5cfca7deba` |
| `skillopt/prompts/ranking.md` | `8bf37725b7b328d8d54af7a8fb173179bdd0f02b57399ed2cc7a5b5d25edea8a` |
| `skillopt/prompts/ranking_rewrite.md` | `2ff8c755188704352cb7dd5c57bbce66c506145a5a5cd212b8881bd09d608031` |
| `skillopt/prompts/rewrite_skill.md` | `2da412ee1d92ab32e59f8a33e2816a2c315edef5e2906d7628d8d093ae0dc68a` |
| `skillopt/prompts/slow_update.md` | `7f6f97d2283a370ad072a91ad61e152551f4f4eb4f519c9a1c5c60679b4782f2` |
| `skillopt/types.py` | `f8f425c44f352255dd36a5bd6438ae0aafbc671d5e84af349ccb32c9674fddfa` |
| `skillopt/utils/__init__.py` | `7a2f608332c095057fa82173f4841dee1d6f9f7a127b35a4770bf418319bc5f0` |
| `skillopt/utils/console.py` | `211f0f4370b28044f826f04df10ea84388879432471166add335fef66e5f8869` |
| `skillopt/utils/json_utils.py` | `77d06263ab9dd6537c52c80a7cd4614d2a0362ed933407725a032bb095738589` |
| `skillopt/utils/scoring.py` | `f5756974a6d0f62f45bc408cc18ea13376eb1ab6592a351e8722d53860fcb8e9` |

Not vendored: `scripts/`, `skillopt_sleep/`, `skillopt_webui/`, `plugins/`,
`data/`, `ckpt/`, the other configs, `skillopt/scheduler/` (imported by
nothing here), and every env under `skillopt/envs/` except the prompt and
skill files listed above. Their adapters, loaders and rollouts are replaced by
`baselines/skillopt/adapter.py`.

## Local modifications

None. `tests/test_skillopt.py` checks every file against the table above.

## Reproduce

From the repository root:

```bash
git clone https://github.com/microsoft/SkillOpt /tmp/skillopt
git -C /tmp/skillopt checkout 32deaeb50f92fc2f3a225bbf67d3eb3a604ce8b6
SRC=/tmp/skillopt
DST=baselines/skillopt/upstream
mkdir -p $DST/configs/_base_ $DST/skillopt/envs/searchqa/prompts \
         $DST/skillopt/envs/livemathematicianbench/prompts
cp $SRC/LICENSE $DST/
cp $SRC/configs/_base_/default.yaml $DST/configs/_base_/
cp $SRC/skillopt/__init__.py $SRC/skillopt/config.py $SRC/skillopt/types.py $DST/skillopt/
for d in datasets engine evaluation gradient model optimizer prompts utils; do
  mkdir -p $DST/skillopt/$d
  find $SRC/skillopt/$d -maxdepth 1 \( -name '*.py' -o -name '*.md' \) -exec cp {} $DST/skillopt/$d/ \;
done
cp $SRC/skillopt/envs/__init__.py $SRC/skillopt/envs/base.py $DST/skillopt/envs/
cp $SRC/skillopt/envs/searchqa/prompts/analyst_error.md $SRC/skillopt/envs/searchqa/prompts/analyst_success.md \
   $DST/skillopt/envs/searchqa/prompts/
cp $SRC/skillopt/envs/livemathematicianbench/prompts/analyst_error.md \
   $SRC/skillopt/envs/livemathematicianbench/prompts/analyst_success.md \
   $DST/skillopt/envs/livemathematicianbench/prompts/
```

## Lint and type checks

This directory is excluded from ruff (`extend-exclude`) and from strict mypy
(`exclude`, plus `ignore_errors` for `skillopt.*`, which mypy still follows
through the adapter's imports). `make fmt` leaves it untouched.
