# SPDX-License-Identifier: Apache-2.0
"""Cover queued target reads/writeback for the entire actual model invocation."""

from functools import wraps


def use_receipt_targets(execute):
    @wraps(execute)
    def guarded(runner, scheduler_output, *args, **kwargs):
        # The runner resolves its actual connector, so this helper needs no
        # global transport or NPU imports. Scope exit records queued-use tails,
        # including on exceptions and the execute_model(None)/sample split.
        with runner.receipt_target_use_context(scheduler_output):
            return execute(runner, scheduler_output, *args, **kwargs)

    return guarded
