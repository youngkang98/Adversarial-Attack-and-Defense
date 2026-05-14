# attacks/backdoor/

Backdoor / trojan attack implementations.

## Subdirectories

### `wanet/`
WaNet backdoor attack — uses smooth warping-field triggers that are invisible to human inspection.
Self-contained; has its own internal imports via `utils/` at repo root.
Original source: [WaNet paper](https://arxiv.org/abs/2102.10369).

### `backdoor_toolbox/`
Unified backdoor attack framework supporting multiple attack methods in a single codebase.
Original directory name was `backdoor-toolbox` (hyphen replaced with underscore for Python import compatibility).

> ⚠️ **Note:** Both `wanet/` and `backdoor_toolbox/` share some overlap in attack coverage.
> `backdoor_toolbox/` is the more unified framework; `wanet/` is the original standalone implementation.
