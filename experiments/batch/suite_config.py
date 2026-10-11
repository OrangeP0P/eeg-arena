import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

from experiments.common.argparse_utils import argv_contains_option


DEFAULT_SUITE_PATH = Path(__file__).resolve().parents[1] / "configs" / "default_suite.json"

BUILTIN_DEFAULT_SUITE = {
    "models": [
        "atcnet",
        "bendr",
        "biot",
        "cbramod",
        "csbrain",
        "deepconvnet",
        "eegconformer",
        "eegnet",
        "eegnex",
        "msvtnet",
        "shallowconvnet",
        "eegpt",
        "femba",
        "labram",
        "lead",
        "luna",
        "neurogpt",
        "neurorvq",
        "reve",
    ],
    "datasets": [
        "physionet_eegmmidb:motor_imagery_left_right_both_fists_feet_4class",
        "bcic_iv_2b:motor_imagery_left_right_2class",
        "figshare_largemi:motor_imagery_left_right_2class",
        "figshare_shudb:motor_imagery_left_right_2class",
        "other_highgammadataset:motor_imagery_left_right_feet_tongue_4class",
    ],
    "common_args": {
        "epochs": 2,
        "batch_size": 32,
        "num_workers": 8,
        "data_ratio": 1.0,
        "seed": 0,
        "split_mode": "cs",
        "fold": 4,
        "channel_system": "10-20",
        "cache_dir": "./data/cache",
    },
    "model_overrides": {
        "atcnet": {
            "input_format": "raw",
        },
        "bendr": {
            "input_format": "raw",
        },
        "biot": {
            "input_format": "raw",
            "variant": "shhs-prest-18ch",
        },
        "cbramod": {
            "input_format": "patched",
            "patch_size": 200,
        },
        "csbrain": {
            "input_format": "patched",
            "patch_size": 200,
        },
        "deepconvnet": {
            "input_format": "raw",
        },
        "eegconformer": {
            "input_format": "raw",
        },
        "eegnet": {
            "input_format": "raw",
        },
        "eegnex": {
            "input_format": "raw",
        },
        "msvtnet": {
            "input_format": "raw",
        },
        "shallowconvnet": {
            "input_format": "raw",
        },
        "eegpt": {
            "input_format": "patched",
            "patch_size": 64,
            "patch_stride": 64,
            "desired_time_len": 1024,
        },
        "femba": {
            "input_format": "raw",
            "variant": "base",
            "batch_size": 2,
            "pipeline": "datasets/pipelines/femba_official.py",
            "pipeline_params": {
                "resample": 256,
                "lower_bound": 0.1,
                "upper_bound": 75,
                "notch": 60,
                "scale_uv": True,
                "normalize": True,
                "q_lower": -20.0,
                "q_upper": 20.0,
            },
        },
        "labram": {
            "input_format": "patched",
            "patch_size": 200,
        },
        "lead": {
            "input_format": "raw",
            "batch_size": 16,
        },
        "luna": {
            "input_format": "raw",
            "variant": "base",
            "batch_size": 2,
        },
        "neurogpt": {
            "input_format": "patched",
            "patch_size": 200,
        },
        "neurorvq": {
            "input_format": "patched",
            "patch_size": 200,
            "batch_size": 16,
        },
        "reve": {
            "input_format": "raw",
            "model_size": "base",
            "depth": 12,
            "variant": "base",
        },
    },
    "dataset_overrides": {},
    "task_overrides": {},
}

REQUIRED_COMMON_ARGS = [
    "epochs",
    "batch_size",
    "num_workers",
    "seed",
    "split_mode",
    "fold",
    "channel_system",
    "cache_dir",
]

REQUIRED_RUN_ARGS = list(REQUIRED_COMMON_ARGS)
SPLIT_PLAN_MODE_ORDER = ("cs", "ws", "loso", "cross_trial")
SPLIT_ARG_KEYS = {"split_mode", "fold", "held_out_subject"}

LEGACY_DESIGN_TIME_FLAGS = [
    "--models",
    "--datasets",
    "--epochs",
    "--batch-size",
    "--num-workers",
    "--data-ratio",
    "--data_ratio",
    "--seed",
    "--split-mode",
    "--held-out-subject",
    "--held_out_subject",
    "--fold",
    "--channel-system",
]


@dataclass
class ExplicitTaskSpec:
    name: str | None
    model: str
    dataset: str
    run_args: dict


@dataclass
class SuiteSpec:
    source: str
    config_path: str | None
    models: list[str]
    datasets: list[str]
    common_args: dict
    sweeps: dict
    split_plan: dict
    model_overrides: dict
    dataset_overrides: dict
    task_overrides: dict
    explicit_tasks: list[ExplicitTaskSpec]


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_default_suite():
    if DEFAULT_SUITE_PATH.exists():
        return _load_json(DEFAULT_SUITE_PATH)
    return deepcopy(BUILTIN_DEFAULT_SUITE)


_DEFAULT_SUITE = load_default_suite()
DEFAULT_MODEL_ORDER = list(_DEFAULT_SUITE["models"])
DEFAULT_DATASET_ORDER = list(_DEFAULT_SUITE["datasets"])
DEFAULT_MODEL_OVERRIDES = deepcopy(_DEFAULT_SUITE["model_overrides"])

PIPELINE_ARG_KEY = "pipeline"
PIPELINE_PARAMS_ARG_KEY = "pipeline_params"
EPOCH_PARAMS_ARG_KEY = "epoch_params"
STRUCTURED_ARG_KEYS = {PIPELINE_PARAMS_ARG_KEY, EPOCH_PARAMS_ARG_KEY}
DISALLOWED_SWEEP_KEYS = {PIPELINE_ARG_KEY, PIPELINE_PARAMS_ARG_KEY, EPOCH_PARAMS_ARG_KEY}


def _ensure_param_object(mapping_name, key, value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(
            f"'{mapping_name}.{key}' must be an object mapping parameter names to scalar/bool values."
        )

    normalized = {}
    for param_key, param_value in value.items():
        if not isinstance(param_key, str) or not param_key or param_key.startswith("-"):
            raise ValueError(
                f"Invalid parameter key '{param_key}' in '{mapping_name}.{key}'. "
                "Use parameter names without leading dashes."
            )
        if isinstance(param_value, bool):
            normalized[param_key] = param_value
            continue
        if isinstance(param_value, (str, int, float)):
            normalized[param_key] = param_value
            continue
        raise ValueError(
            f"Unsupported value for '{mapping_name}.{key}.{param_key}': {type(param_value).__name__}. "
            "Only scalar and boolean values are supported."
        )
    return normalized


def _ensure_arg_mapping(mapping_name, mapping, *, allowed_structured_keys=None, nullable_keys=None):
    if not isinstance(mapping, dict):
        raise ValueError(f"'{mapping_name}' must be an object mapping arg names to scalar/bool values.")

    allowed_structured_keys = set(allowed_structured_keys or ())
    nullable_keys = set(nullable_keys or ())
    normalized = {}
    for key, value in mapping.items():
        if not isinstance(key, str) or not key or key.startswith("-"):
            raise ValueError(f"Invalid arg key '{key}' in '{mapping_name}'. Use argparse dest names without leading dashes.")
        if key in STRUCTURED_ARG_KEYS and key not in allowed_structured_keys:
            raise ValueError(f"'{mapping_name}.{key}' is not allowed in this section.")
        if key in allowed_structured_keys:
            normalized[key] = _ensure_param_object(mapping_name, key, value)
            continue
        if value is None:
            if key in nullable_keys:
                normalized[key] = None
                continue
            raise ValueError(
                f"Unsupported value for '{mapping_name}.{key}': null. "
                "Only scalar and boolean values are supported in this section."
            )
        if isinstance(value, bool):
            normalized[key] = value
            continue
        if isinstance(value, (str, int, float)):
            normalized[key] = value
            continue
        raise ValueError(
            f"Unsupported value for '{mapping_name}.{key}': {type(value).__name__}. "
            "Only scalar and boolean values are supported in suite configs."
        )
    return normalized


def _ensure_string_list(name, values):
    if not isinstance(values, list) or not values:
        raise ValueError(f"'{name}' must be a non-empty list of strings.")

    normalized = []
    seen = set()
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError(f"'{name}' must contain only non-empty strings.")
        if value in seen:
            raise ValueError(f"'{name}' contains duplicate entry '{value}'.")
        normalized.append(value)
        seen.add(value)
    return normalized


def _ensure_scalar_list(name, values):
    if not isinstance(values, list) or not values:
        raise ValueError(f"'{name}' must be a non-empty list of scalar/bool values.")

    normalized = []
    seen = set()
    for value in values:
        if not isinstance(value, (bool, str, int, float)):
            raise ValueError(
                f"Unsupported value in '{name}': {type(value).__name__}. "
                "Only scalar and boolean values are supported in suite sweeps."
            )
        dedupe_key = (type(value).__name__, value)
        if dedupe_key in seen:
            raise ValueError(f"'{name}' contains duplicate value '{value}'.")
        normalized.append(value)
        seen.add(dedupe_key)
    return normalized


def _ensure_sweeps_mapping(mapping_name, mapping):
    if mapping is None:
        return {}
    if not isinstance(mapping, dict):
        raise ValueError(f"'{mapping_name}' must be an object mapping arg names to value lists.")

    normalized = {}
    for key, values in mapping.items():
        if not isinstance(key, str) or not key or key.startswith("-"):
            raise ValueError(
                f"Invalid sweep key '{key}' in '{mapping_name}'. Use argparse dest names without leading dashes."
            )
        if key in DISALLOWED_SWEEP_KEYS:
            raise ValueError(
                f"'{mapping_name}.{key}' is not supported. "
                "Configure pipeline and epoch parameters via common_args/overrides/tasks instead of sweeps."
            )
        normalized[key] = _ensure_scalar_list(f"{mapping_name}.{key}", values)
    return normalized


def _ensure_no_sweep_key_conflicts(source_name, sweep_keys, mapping_name, mapping):
    conflicts = sorted(set(sweep_keys) & set(mapping.keys()))
    if conflicts:
        raise ValueError(
            f"{source_name} uses sweep keys in '{mapping_name}' as well: {', '.join(conflicts)}. "
            "Sweep keys must only be declared in 'sweeps'."
        )


def _get_pinned_sweep_keys(spec: SuiteSpec, model: str, dataset: str):
    pinned = set()
    pinned.update(spec.model_overrides.get(model, {}).keys())
    pinned.update(spec.dataset_overrides.get(dataset, {}).keys())
    pinned.update(spec.task_overrides.get(model, {}).get(dataset, {}).keys())
    return pinned


def _ensure_no_split_plan_arg_conflicts(source_name, mapping_name, mapping):
    conflicts = sorted(set(mapping.keys()) & SPLIT_ARG_KEYS)
    if conflicts:
        raise ValueError(
            f"{source_name} uses split args in '{mapping_name}' while 'split_plan' is set: "
            + ", ".join(conflicts)
        )


def _ensure_split_plan(payload):
    if payload is None:
        return {}
    if not isinstance(payload, dict) or not payload:
        raise ValueError("'split_plan' must be a non-empty object when provided.")

    normalized = {}
    for split_mode in SPLIT_PLAN_MODE_ORDER:
        if split_mode not in payload:
            continue

        split_cfg = payload[split_mode]
        if not isinstance(split_cfg, dict) or not split_cfg:
            raise ValueError(f"'split_plan.{split_mode}' must be an object.")

        if split_mode in {"cs", "ws", "cross_trial"}:
            unknown_keys = sorted(set(split_cfg.keys()) - {"folds"})
            if unknown_keys:
                raise ValueError(
                    f"'split_plan.{split_mode}' contains unknown keys: {', '.join(unknown_keys)}."
                )
            folds = split_cfg.get("folds")
            if folds == "all":
                normalized[split_mode] = {"folds": [1]} if split_mode == "cross_trial" else {"folds": "all"}
                continue
            if not isinstance(folds, list) or not folds:
                raise ValueError(f"'split_plan.{split_mode}.folds' must be 'all' or a non-empty list.")

            normalized_folds = []
            seen = set()
            for fold in folds:
                if not isinstance(fold, int):
                    raise ValueError(f"'split_plan.{split_mode}.folds' must contain only integers.")
                if split_mode == "cross_trial":
                    if fold != 1:
                        raise ValueError(
                            f"'split_plan.{split_mode}.folds' only supports the single-fold value 1."
                        )
                elif fold < 1 or fold > 5:
                    raise ValueError(f"'split_plan.{split_mode}.folds' values must be between 1 and 5.")
                if fold in seen:
                    raise ValueError(f"'split_plan.{split_mode}.folds' contains duplicate value '{fold}'.")
                normalized_folds.append(fold)
                seen.add(fold)
            normalized[split_mode] = {"folds": normalized_folds}
            continue

        unknown_keys = sorted(set(split_cfg.keys()) - {"subjects"})
        if unknown_keys:
            raise ValueError(
                f"'split_plan.{split_mode}' contains unknown keys: {', '.join(unknown_keys)}."
            )
        subjects = split_cfg.get("subjects")
        if subjects == "all":
            normalized[split_mode] = {"subjects": "all"}
            continue
        normalized_subjects = _ensure_string_list(f"split_plan.{split_mode}.subjects", subjects)
        normalized[split_mode] = {"subjects": normalized_subjects}

    unknown_modes = sorted(set(payload.keys()) - set(SPLIT_PLAN_MODE_ORDER))
    if unknown_modes:
        raise ValueError(f"'split_plan' contains unsupported split modes: {', '.join(unknown_modes)}.")
    return normalized


def _unique_in_order(values):
    ordered = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return ordered


def _validate_explicit_tasks_config(payload, source_name):
    unknown_keys = sorted(set(payload.keys()) - {"tasks", "common_args"})
    if unknown_keys:
        raise ValueError(
            f"{source_name} contains unknown top-level keys: {', '.join(unknown_keys)}."
        )

    tasks_raw = payload.get("tasks")
    if not isinstance(tasks_raw, list) or not tasks_raw:
        raise ValueError(f"{source_name} 'tasks' must be a non-empty list.")

    common_args = _ensure_arg_mapping(
        "common_args",
        payload.get("common_args", {}),
        allowed_structured_keys={PIPELINE_PARAMS_ARG_KEY},
        nullable_keys={PIPELINE_ARG_KEY, PIPELINE_PARAMS_ARG_KEY},
    )

    disallowed_keys = [
        key
        for key in ("models", "datasets", "sweeps", "split_plan", "model_overrides", "dataset_overrides", "task_overrides")
        if key in payload
    ]
    if disallowed_keys:
        raise ValueError(
            f"{source_name} task mode cannot be combined with matrix keys: {', '.join(disallowed_keys)}."
        )

    explicit_tasks = []
    ordered_models = []
    ordered_datasets = []

    for index, task_payload in enumerate(tasks_raw, start=1):
        item_name = f"tasks[{index - 1}]"
        if not isinstance(task_payload, dict):
            raise ValueError(f"{source_name} {item_name} must be an object.")

        unknown_keys = sorted(set(task_payload.keys()) - {"name", "model", "dataset", "args"})
        if unknown_keys:
            raise ValueError(
                f"{source_name} {item_name} contains unknown keys: {', '.join(unknown_keys)}."
            )

        model = task_payload.get("model")
        dataset = task_payload.get("dataset")
        if not isinstance(model, str) or not model:
            raise ValueError(f"{source_name} {item_name}.model must be a non-empty string.")
        if not isinstance(dataset, str) or not dataset:
            raise ValueError(f"{source_name} {item_name}.dataset must be a non-empty string.")

        name = task_payload.get("name")
        if name is not None and (not isinstance(name, str) or not name):
            raise ValueError(f"{source_name} {item_name}.name must be a non-empty string when provided.")

        args = _ensure_arg_mapping(
            f"{item_name}.args",
            task_payload.get("args", {}),
            allowed_structured_keys={PIPELINE_PARAMS_ARG_KEY, EPOCH_PARAMS_ARG_KEY},
            nullable_keys={PIPELINE_ARG_KEY, PIPELINE_PARAMS_ARG_KEY, EPOCH_PARAMS_ARG_KEY},
        )
        run_args = {}
        run_args.update(common_args)
        run_args.update(args)

        missing_run_args = [key for key in REQUIRED_RUN_ARGS if key not in run_args]
        if missing_run_args:
            raise ValueError(
                f"{source_name} {item_name} is missing required run args after merging common_args and args: "
                + ", ".join(missing_run_args)
            )
        _validate_resolved_run_args(run_args, f"{source_name} {item_name}")

        explicit_tasks.append(
            ExplicitTaskSpec(
                name=name,
                model=model,
                dataset=dataset,
                run_args=run_args,
            )
        )
        ordered_models.append(model)
        ordered_datasets.append(dataset)

    return SuiteSpec(
        source="config",
        config_path=None,
        models=_unique_in_order(ordered_models),
        datasets=_unique_in_order(ordered_datasets),
        common_args=common_args,
        sweeps={},
        split_plan={},
        model_overrides={},
        dataset_overrides={},
        task_overrides={},
        explicit_tasks=explicit_tasks,
    )


def load_suite_config(path):
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Suite config not found: {config_path}")
    if not config_path.is_file():
        raise ValueError(f"Suite config path is not a file: {config_path}")
    payload = _load_json(config_path)
    spec = validate_suite_config(payload, source_name=str(config_path))
    spec.config_path = str(config_path)
    spec.source = "config"
    return spec


def validate_suite_config(payload, source_name="<suite_config>"):
    if not isinstance(payload, dict):
        raise ValueError(f"{source_name} must be a JSON object.")

    if "tasks" in payload:
        return _validate_explicit_tasks_config(payload, source_name)

    if "models" not in payload:
        raise ValueError(f"{source_name} is missing required key 'models'.")
    if "datasets" not in payload:
        raise ValueError(f"{source_name} is missing required key 'datasets'.")
    if "common_args" not in payload:
        raise ValueError(f"{source_name} is missing required key 'common_args'.")

    allowed_keys = {
        "models",
        "datasets",
        "common_args",
        "sweeps",
        "split_plan",
        "model_overrides",
        "dataset_overrides",
        "task_overrides",
    }
    unknown_keys = sorted(set(payload.keys()) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"{source_name} contains unknown top-level keys: {', '.join(unknown_keys)}."
        )

    models = _ensure_string_list("models", payload["models"])
    datasets = _ensure_string_list("datasets", payload["datasets"])
    common_args = _ensure_arg_mapping(
        "common_args",
        payload["common_args"],
        allowed_structured_keys={PIPELINE_PARAMS_ARG_KEY},
        nullable_keys={PIPELINE_ARG_KEY, PIPELINE_PARAMS_ARG_KEY},
    )
    sweeps = _ensure_sweeps_mapping("sweeps", payload.get("sweeps", {}))
    split_plan = _ensure_split_plan(payload.get("split_plan"))

    _ensure_no_sweep_key_conflicts(source_name, sweeps.keys(), "common_args", common_args)
    if split_plan:
        _ensure_no_split_plan_arg_conflicts(source_name, "common_args", common_args)
        split_conflicts = sorted({"split_mode", "fold", "held_out_subject"} & set(sweeps.keys()))
        if split_conflicts:
            raise ValueError(
                f"{source_name} cannot combine 'split_plan' with split sweep keys: "
                + ", ".join(split_conflicts)
            )

    required_run_args = list(REQUIRED_RUN_ARGS)
    if split_plan:
        required_run_args = [key for key in required_run_args if key not in {"split_mode", "fold"}]
    missing_run_args = [
        key for key in required_run_args if key not in common_args and key not in sweeps
    ]
    if missing_run_args:
        raise ValueError(
            f"{source_name} is missing required run args across common_args/sweeps: {', '.join(missing_run_args)}."
        )

    model_overrides_raw = payload.get("model_overrides", {})
    dataset_overrides_raw = payload.get("dataset_overrides", {})
    task_overrides_raw = payload.get("task_overrides", {})

    if not isinstance(model_overrides_raw, dict):
        raise ValueError(f"{source_name} 'model_overrides' must be an object.")
    if not isinstance(dataset_overrides_raw, dict):
        raise ValueError(f"{source_name} 'dataset_overrides' must be an object.")
    if not isinstance(task_overrides_raw, dict):
        raise ValueError(f"{source_name} 'task_overrides' must be an object.")

    model_overrides = {}
    for model_name, override in model_overrides_raw.items():
        if model_name not in models:
            raise ValueError(
                f"{source_name} model_overrides references undeclared model '{model_name}'."
            )
        normalized_override = _ensure_arg_mapping(
            f"model_overrides.{model_name}",
            override,
            allowed_structured_keys={PIPELINE_PARAMS_ARG_KEY},
            nullable_keys={PIPELINE_ARG_KEY, PIPELINE_PARAMS_ARG_KEY},
        )
        if split_plan:
            _ensure_no_split_plan_arg_conflicts(
                source_name,
                f"model_overrides.{model_name}",
                normalized_override,
            )
        model_overrides[model_name] = normalized_override

    dataset_overrides = {}
    for dataset_name, override in dataset_overrides_raw.items():
        if dataset_name not in datasets:
            raise ValueError(
                f"{source_name} dataset_overrides references undeclared dataset '{dataset_name}'."
            )
        normalized_override = _ensure_arg_mapping(
            f"dataset_overrides.{dataset_name}",
            override,
            allowed_structured_keys={PIPELINE_PARAMS_ARG_KEY, EPOCH_PARAMS_ARG_KEY},
            nullable_keys={PIPELINE_ARG_KEY, PIPELINE_PARAMS_ARG_KEY, EPOCH_PARAMS_ARG_KEY},
        )
        if split_plan:
            _ensure_no_split_plan_arg_conflicts(
                source_name,
                f"dataset_overrides.{dataset_name}",
                normalized_override,
            )
        dataset_overrides[dataset_name] = normalized_override

    task_overrides = {}
    for model_name, dataset_map in task_overrides_raw.items():
        if model_name not in models:
            raise ValueError(
                f"{source_name} task_overrides references undeclared model '{model_name}'."
            )
        if not isinstance(dataset_map, dict):
            raise ValueError(f"{source_name} task_overrides.{model_name} must be an object.")
        task_overrides[model_name] = {}
        for dataset_name, override in dataset_map.items():
            if dataset_name not in datasets:
                raise ValueError(
                    f"{source_name} task_overrides.{model_name} references undeclared dataset '{dataset_name}'."
                )
            normalized_override = _ensure_arg_mapping(
                f"task_overrides.{model_name}.{dataset_name}",
                override,
                allowed_structured_keys={PIPELINE_PARAMS_ARG_KEY, EPOCH_PARAMS_ARG_KEY},
                nullable_keys={PIPELINE_ARG_KEY, PIPELINE_PARAMS_ARG_KEY, EPOCH_PARAMS_ARG_KEY},
            )
            if split_plan:
                _ensure_no_split_plan_arg_conflicts(
                    source_name,
                    f"task_overrides.{model_name}.{dataset_name}",
                    normalized_override,
                )
            task_overrides[model_name][dataset_name] = normalized_override

    return SuiteSpec(
        source="config",
        config_path=None,
        models=models,
        datasets=datasets,
        common_args=common_args,
        sweeps=sweeps,
        split_plan=split_plan,
        model_overrides=model_overrides,
        dataset_overrides=dataset_overrides,
        task_overrides=task_overrides,
        explicit_tasks=[],
    )


def build_legacy_suite_spec(args):
    common_args = {key: getattr(args, key) for key in REQUIRED_COMMON_ARGS}
    common_args["data_ratio"] = getattr(args, "data_ratio", 1.0)
    if getattr(args, "held_out_subject", None) is not None:
        common_args["held_out_subject"] = getattr(args, "held_out_subject")
    selected_model_overrides = {
        model: deepcopy(DEFAULT_MODEL_OVERRIDES[model])
        for model in args.models
        if model in DEFAULT_MODEL_OVERRIDES
    }
    return SuiteSpec(
        source="legacy_cli",
        config_path=None,
        models=list(args.models),
        datasets=list(args.datasets),
        common_args=common_args,
        sweeps={},
        split_plan={},
        model_overrides=selected_model_overrides,
        dataset_overrides={},
        task_overrides={},
        explicit_tasks=[],
    )


def ensure_no_config_conflicts(parser, args, argv):
    if not getattr(args, "config", None):
        return

    conflicting_flags = [
        flag for flag in LEGACY_DESIGN_TIME_FLAGS
        if argv_contains_option(argv, flag)
    ]
    if conflicting_flags:
        parser.error(
            "--config cannot be combined with design-time scheduler flags: "
            + ", ".join(conflicting_flags)
        )


def resolve_suite_spec(args, parser=None, argv=None):
    argv = list(argv or [])
    if getattr(args, "config", None):
        if parser is not None:
            ensure_no_config_conflicts(parser, args, argv)
        return load_suite_config(args.config)
    return build_legacy_suite_spec(args)


def iter_sweep_arg_maps(spec: SuiteSpec, *, model: str | None = None, dataset: str | None = None):
    if not spec.sweeps:
        yield {}
        return

    pinned_sweep_keys = set()
    if model is not None and dataset is not None:
        pinned_sweep_keys = _get_pinned_sweep_keys(spec, model, dataset)

    sweep_keys = [key for key in spec.sweeps.keys() if key not in pinned_sweep_keys]
    if not sweep_keys:
        yield {}
        return
    sweep_value_lists = [spec.sweeps[key] for key in sweep_keys]
    for values in product(*sweep_value_lists):
        yield dict(zip(sweep_keys, values))


def _apply_arg_layer(resolved, layer):
    for key, value in (layer or {}).items():
        if key in STRUCTURED_ARG_KEYS and value is not None:
            resolved[key] = deepcopy(value)
        else:
            resolved[key] = value


def _validate_resolved_run_args(run_args, context_name, spec=None):
    if run_args.get(PIPELINE_ARG_KEY) is None and run_args.get(PIPELINE_PARAMS_ARG_KEY):
        raise ValueError(
            f"{context_name} sets '{PIPELINE_PARAMS_ARG_KEY}' without a '{PIPELINE_ARG_KEY}'."
        )
    if spec is not None and spec.split_plan:
        return  # split_mode/fold will be set by _expand_run_args_for_split_plan
    split_mode = run_args.get("split_mode")
    if split_mode not in {"cs", "ws", "loso", "cross_trial"}:
        raise ValueError(
            f"{context_name} sets unsupported split_mode '{split_mode}'. "
            "Use one of: cs, ws, loso, cross_trial."
        )
    if run_args.get("held_out_subject") is not None and run_args.get("split_mode") != "loso":
        raise ValueError(
            f"{context_name} sets 'held_out_subject' but split_mode is not 'loso'."
        )
    if split_mode == "cross_trial" and int(run_args.get("fold", 1)) != 1:
        raise ValueError(
            f"{context_name} sets split_mode='cross_trial' but fold is not 1."
        )
    data_ratio = run_args.get("data_ratio", 1.0)
    try:
        data_ratio = float(data_ratio)
    except (TypeError, ValueError):
        raise ValueError(
            f"{context_name} sets invalid data_ratio '{run_args.get('data_ratio')}'. "
            "Use a float in (0, 1]."
        )
    if not math.isfinite(data_ratio) or data_ratio <= 0.0 or data_ratio > 1.0:
        raise ValueError(
            f"{context_name} sets invalid data_ratio '{run_args.get('data_ratio')}'. "
            "Use a float in (0, 1]."
        )


def resolve_task_run_args(spec: SuiteSpec, model: str, dataset: str, sweep_args=None):
    resolved = {}
    _apply_arg_layer(resolved, spec.common_args)
    _apply_arg_layer(resolved, spec.model_overrides.get(model, {}))
    _apply_arg_layer(resolved, spec.dataset_overrides.get(dataset, {}))
    _apply_arg_layer(resolved, spec.task_overrides.get(model, {}).get(dataset, {}))
    _apply_arg_layer(resolved, sweep_args or {})
    if not spec.split_plan:
        _validate_resolved_run_args(resolved, f"{model}:{dataset}", spec=spec)
    return resolved


def suite_spec_to_dict(spec: SuiteSpec):
    return asdict(spec)


__all__ = [
    "DEFAULT_DATASET_ORDER",
    "DEFAULT_MODEL_ORDER",
    "DEFAULT_SUITE_PATH",
    "ExplicitTaskSpec",
    "LEGACY_DESIGN_TIME_FLAGS",
    "REQUIRED_COMMON_ARGS",
    "REQUIRED_RUN_ARGS",
    "SuiteSpec",
    "build_legacy_suite_spec",
    "ensure_no_config_conflicts",
    "iter_sweep_arg_maps",
    "load_default_suite",
    "load_suite_config",
    "resolve_suite_spec",
    "resolve_task_run_args",
    "suite_spec_to_dict",
    "validate_suite_config",
]
