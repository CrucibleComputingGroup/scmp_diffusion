"""Re-export shim — canonical MP API lives in ``scmp_kernels.mp``.

Local relative imports inside this package (``from .mp_config import ...``)
continue to work via this shim.
"""

from scmp_kernels.mp import (  # noqa: F401
    MPConfig,
    AdaptiveMPConfig,
    RangeMPConfig,
    RowAssignment,
    classify_rows_by_metric,
    adaptive_classify_rows,
    classify_groups_by_range,
    MPDistributionLogger,
    MetricProfiler,
)
