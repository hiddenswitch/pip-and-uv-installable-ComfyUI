from __future__ import annotations

from .types import TensorParallelConfig


def tensor_parallel_operations(base_operations, parallel: TensorParallelConfig):
    """Decorate Comfy operations with Megatron-style tensor-parallel linears."""

    class TensorParallelOperations(base_operations):
        tensor_parallel = parallel

        class ColumnParallelLinear(base_operations.Linear):
            def __init__(self, in_features, out_features, bias=True, *, sections=1,
                         device=None, dtype=None):
                if out_features % (sections * parallel.size):
                    raise ValueError(
                        f"Column-parallel output {out_features} must divide "
                        f"{sections} sections across {parallel.size} ranks"
                    )
                super().__init__(
                    in_features, out_features // parallel.size, bias=bias,
                    device=device, dtype=dtype,
                )
                self.tensor_parallel_sections = sections
                self.tensor_parallel_out_features = out_features

        class RowParallelLinear(base_operations.Linear):
            def __init__(self, in_features, out_features, bias=True, *,
                         device=None, dtype=None):
                if bias:
                    raise NotImplementedError("Row-parallel bias is not implemented")
                if in_features % parallel.size:
                    raise ValueError(
                        f"Row-parallel input {in_features} must divide {parallel.size} ranks"
                    )
                super().__init__(
                    in_features // parallel.size, out_features, bias=False,
                    device=device, dtype=dtype,
                )
                self.tensor_parallel_in_features = in_features

            def forward(self, *args, **kwargs):
                return parallel.operations.sum(super().forward(*args, **kwargs))

    TensorParallelOperations.__name__ = f"TensorParallel{base_operations.__name__}"
    return TensorParallelOperations
