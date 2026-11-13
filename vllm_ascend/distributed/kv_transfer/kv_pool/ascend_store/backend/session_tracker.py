#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping


class LayerwiseSessionTracker:
    """Track backend sessions that may span chunked-prefill steps."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._request_load_entries: dict[str, dict[str, tuple[int, int]]] = {}
        self._pending_put_owners: dict[str, dict[str, tuple[int, int]]] = {}
        self._load_key_owners: dict[str, set[str]] = {}
        self._request_instances: dict[str, str] = {}
        self._private_snapshot_ends: dict[str, int] = {}
        self._retired_private_keys: set[str] = set()

    def _replace_block_entry(self, entries: dict[str, tuple[int, int]], key: str, block_index: tuple[int, int]) -> None:
        for previous_key, previous_index in list(entries.items()):
            if previous_index == block_index and previous_key != key:
                del entries[previous_key]
                if previous_key in self._private_snapshot_ends:
                    self._retired_private_keys.add(previous_key)
        entries[key] = block_index

    def bind_request_instance(self, req_id: str, instance: str) -> list[str]:
        """Detach old private versions when an external request ID is reused."""
        if not instance:
            raise ValueError("A private snapshot requires a request instance")
        with self._lock:
            if self._request_instances.get(req_id) == instance:
                return []
            released = self.release_terminal({req_id})
            self._request_instances[req_id] = instance
            return released

    def register_private_snapshot(self, key: str, end_token: int) -> None:
        if not any(marker in key for marker in ("@private_snapshot_v1:", "@private_full_v1:")) or end_token <= 0:
            raise ValueError("Invalid private snapshot identity or valid token range")
        with self._lock:
            previous = self._private_snapshot_ends.get(key)
            if previous is not None and previous != end_token:
                raise ValueError("A private snapshot key cannot change its valid token range")
            self._private_snapshot_ends[key] = end_token

    def private_snapshot_end(self, key: str) -> int | None:
        with self._lock:
            return self._private_snapshot_ends.get(key)

    def register_put_keys(self, req_id: str, entries: Iterable[tuple[str, int]], group_id: int = 0) -> None:
        """Remember which requests should consume a key after commit succeeds."""
        with self._lock:
            for key, block_index in entries:
                self._pending_put_owners.setdefault(key, {})[req_id] = (group_id, block_index)

    def commit_put_keys(self, keys: Iterable[str]) -> None:
        """Promote committed keys into each owner's future load set."""
        with self._lock:
            for key in keys:
                request_entries = self._pending_put_owners.pop(key, {})
                for req_id, block_index in request_entries.items():
                    entries = self._request_load_entries.setdefault(req_id, {})
                    self._replace_block_entry(entries, key, block_index)

    def revoke_put_keys(self, keys: Iterable[str]) -> None:
        with self._lock:
            for key in keys:
                self._pending_put_owners.pop(key, None)
                if key in self._private_snapshot_ends:
                    self._retired_private_keys.add(key)

    def prepare_load_entries(
        self,
        req_id: str,
        current_entries: Iterable[tuple[str, int]],
        group_id: int = 0,
        prefer_committed: bool = False,
        record_current: bool = True,
    ) -> list[tuple[str, int]]:
        """Merge current remote hits with keys committed by earlier chunks."""
        with self._lock:
            entries = self._request_load_entries.setdefault(req_id, {})
            if not record_current:
                entries = entries.copy()
            for key, block_index in current_entries:
                if (
                    prefer_committed
                    and key not in self._private_snapshot_ends
                    and (group_id, block_index) in entries.values()
                ):
                    continue
                self._replace_block_entry(entries, key, (group_id, block_index))
            return [(key, index) for key, (group, index) in entries.items() if group == group_id]

    def record_get_result(
        self,
        key: str,
        req_ids: Iterable[str],
        *,
        succeeded: bool,
    ) -> None:
        with self._lock:
            if succeeded:
                self._load_key_owners.setdefault(key, set()).update(req_ids)
            else:
                self._load_key_owners.pop(key, None)

    def release_failed_get_attempts(
        self,
        request_ids_by_key: Mapping[str, Iterable[str]],
    ) -> list[str]:
        """Release only owners involved in a failed get-start attempt."""
        with self._lock:
            keys_to_end: list[str] = []
            for key, req_ids in request_ids_by_key.items():
                owners = self._load_key_owners.get(key)
                if owners is None:
                    keys_to_end.append(key)
                    continue
                owners.difference_update(req_ids)
                if not owners:
                    keys_to_end.append(key)
                    del self._load_key_owners[key]
            return keys_to_end

    def _release_active_owners_locked(self, req_ids: set[str]) -> list[str]:
        keys_to_end: list[str] = []
        for key, owners in list(self._load_key_owners.items()):
            owners.difference_update(req_ids)
            if not owners:
                keys_to_end.append(key)
                del self._load_key_owners[key]
        return keys_to_end

    def release_for_retry(self, req_ids: set[str]) -> list[str]:
        """Release active sessions while retaining request entries for retry."""
        if not req_ids:
            return []
        with self._lock:
            return self._release_active_owners_locked(req_ids)

    def release_terminal(self, req_ids: set[str]) -> list[str]:
        """Release sessions and delete all state owned by terminal requests."""
        if not req_ids:
            return []
        with self._lock:
            keys_to_end = self._release_active_owners_locked(req_ids)
            for req_id in req_ids:
                entries = self._request_load_entries.pop(req_id, {})
                self._retired_private_keys.update(key for key in entries if key in self._private_snapshot_ends)
                self._request_instances.pop(req_id, None)
            for key, owners in list(self._pending_put_owners.items()):
                for req_id in req_ids:
                    owners.pop(req_id, None)
                if not owners:
                    del self._pending_put_owners[key]
                    if key in self._private_snapshot_ends:
                        self._retired_private_keys.add(key)
            return keys_to_end

    def collect_obsolete_private_sessions(self) -> list[str]:
        """Release retired versions only after no request can restore them."""
        with self._lock:
            referenced = {key for entries in self._request_load_entries.values() for key in entries}
            obsolete = self._retired_private_keys - referenced - self._pending_put_owners.keys()
            return [key for key in sorted(obsolete) if self._load_key_owners.pop(key, None) is not None]

    def private_cleanup_candidates(self) -> list[str]:
        with self._lock:
            referenced = {key for entries in self._request_load_entries.values() for key in entries}
            return sorted(
                self._retired_private_keys - referenced - self._pending_put_owners.keys() - self._load_key_owners.keys()
            )

    def acknowledge_private_cleanup(self, keys: Iterable[str]) -> None:
        with self._lock:
            for key in keys:
                if key in self._retired_private_keys:
                    self._retired_private_keys.remove(key)
                    self._private_snapshot_ends.pop(key, None)
