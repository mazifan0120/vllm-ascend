# SPDX-License-Identifier: Apache-2.0
"""Stable data-plane control values shared by sender, receiver and transport."""

GET_TARGET_REGIONS_MSG = b"get_target_regions_v1"
TARGET_REGIONS_REPLY = b"target_regions_v1"
TARGET_PEER_ERROR = b"target_peer_error_v1"
LAYER_RESULTS_MSG = b"layer_results_v1"
PROOF_ACK = b"ACK"
PROOF_ACK_TARGET_CLOSED = b"ACK_TARGET_CLOSED"
MAX_PROOF_BATCH_FRAMES = 64
MAX_PROOF_SEND_ATTEMPTS = 3
TARGET_PEER_TIMEOUT_SECONDS = 30.0

LAYER_RESULT_MSG = b"layer_result_msg"
RANK_TERMINAL_MSG = b"rank_terminal_msg"
RECEIPT_WATCHDOG_SECONDS = 300.0
