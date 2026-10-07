import concurrent.futures
from argparse import Namespace
from collections.abc import Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.distributed as dist

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.protocols.transports.mooncake import MooncakeTransport, RemoteShard
from miles.backends.training_utils.weight_update.protocols.utils.model_param_stager import ModelParamStager
from miles.backends.training_utils.weight_update.protocols.utils.model_replica import (
    ModelReplica,
    ModelReplicas,
    RolloutEngineRankConfig,
    assert_replica_matches_shard,
    pack_into_buffers,
    query_rollout_engine_rank_configs,
    query_runner_roles,
)
from miles.backends.training_utils.weight_update.protocols.utils.rollout_engine_rank_assignment import (
    assign_rollout_engine_ranks,
)
from miles.backends.training_utils.weight_update.protocols.utils.transfer_buffers import TransferBuffers
from miles.utils.distributed_utils import get_gloo_group

# one is loaded while the writes of the other are in flight
_NUM_TRANSFER_BUFFERS = 2


class _ReplicaTarget(NamedTuple):
    model_replica: ModelReplica
    remote_shards: list[RemoteShard]


@dataclass
class _RunnerSender:
    """One model runner (target or draft) of the rollout engine ranks a sender serves: its replicas, the HF tensors
    staged for it and its write targets. Each connect builds a new one around the replicas kept for the process."""

    model_replicas: ModelReplicas
    model_param_stager: ModelParamStager
    replica_targets: list[_ReplicaTarget]

    def holds_param_of(self, hf_name: str) -> bool:
        return (
            self.model_replicas.param_mapper.map(hf_name).sglang_name
            in self.model_replicas.transfer_buffer_param_layouts
        )


class UpdateWeightP2P(WeightTransferProtocol):
    """Writes weight updates straight into the rollout engines' GPU memory over Mooncake.

    Each sender loads the HF tensors of the updater's buckets with a model replica of each rollout engine rank it
    sends to, into transfer buffers in the bytes that rank's loader would write, and writes those bytes to the
    rank's published addresses. When the trainer holds MTP layers and the engines draft with them, the draft runner
    of each rank gets its own replicas and writes. The end of the base weights waits for every write and fails the
    update if any failed or is still running.
    """

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        if args.sglang_pp_size != 1:
            raise NotImplementedError("Rollout pipeline parallelism is not tested yet.")
        self.global_rank = dist.get_rank(group=get_gloo_group())
        self._pending_writes: list[tuple[RemoteShard, Future]] = []
        self._model_replicas_by_runner_role: dict[str, ModelReplicas] = {}
        self._transport: MooncakeTransport | None = None
        self._transfer_buffers: TransferBuffers | None = None
        # the target first: see send_bucket
        self._runner_senders: list[_RunnerSender] = []

    def after_base_weights(self) -> None:
        """Wait for every write of this update; fail the update if any write failed or is still running."""
        if not self.is_sender:
            return
        pending_writes, self._pending_writes = self._pending_writes, []
        _raise_if_any_write_failed(pending_writes, timeout=self.args.p2p_transfer_timeout)
        for runner_sender in self._runner_senders:
            runner_sender.model_param_stager.assert_all_done()

    def send_bucket(self, converted_named_tensors: list[tuple[str, torch.Tensor]]) -> None:
        """Loads the params this bucket completes into transfer buffers, a group at a time, and writes them to every
        rollout engine rank this sender serves.

        Each HF tensor goes to the first runner whose replica holds its param, the target before the draft: a param
        the draft shares with the target (embed, head) is the target's storage, written once through the target.
        The last runner also gets the tensors no runner holds; its stager reports them.
        """
        if not self.is_sender or not converted_named_tensors:
            return
        unclaimed_hf_tensors = converted_named_tensors
        for runner_sender in self._runner_senders[:-1]:
            claimed_hf_tensors = [
                (hf_name, tensor) for hf_name, tensor in unclaimed_hf_tensors if runner_sender.holds_param_of(hf_name)
            ]
            unclaimed_hf_tensors = [
                (hf_name, tensor)
                for hf_name, tensor in unclaimed_hf_tensors
                if not runner_sender.holds_param_of(hf_name)
            ]
            self._send_runner_bucket(runner_sender, claimed_hf_tensors)
        self._send_runner_bucket(self._runner_senders[-1], unclaimed_hf_tensors)

        converted_named_tensors.clear()

    def _send_runner_bucket(
        self, runner_sender: _RunnerSender, converted_named_tensors: list[tuple[str, torch.Tensor]]
    ) -> None:
        ready_hf_tensors_by_param_name = runner_sender.model_param_stager.get_transfer_ready_params(
            converted_named_tensors,
            param_mapper=runner_sender.model_replicas.param_mapper,
            params_dict=runner_sender.model_replicas.transfer_buffer_param_layouts,
        )
        for param_names in pack_into_buffers(
            ready_hf_tensors_by_param_name,
            runner_sender.model_replicas.transfer_buffer_param_layouts,
            self._transfer_buffers.buffer_nbytes,
        ):
            hf_tensors = [hf_tensor for name in param_names for hf_tensor in ready_hf_tensors_by_param_name[name]]
            for target in runner_sender.replica_targets:
                buffer = self._transfer_buffers.acquire()
                param_bytes_by_name = target.model_replica.load_into(buffer, param_names, hf_tensors)
                writes = self._transport.write(target.remote_shards, param_bytes_by_name)
                self._transfer_buffers.release_after(buffer, writes)
                self._pending_writes += zip(target.remote_shards, writes, strict=True)

    def connect(
        self,
        rollout_engines: Sequence[SGLangApiClient],
        engine_gpu_counts: Sequence[int] | None,
        engine_gpu_offsets: Sequence[int] | None,
        parallel_state: ParallelState,
        placement: WeightUpdatePlacement,
        selector: str,
    ) -> None:
        """Connects this trainer rank to the rollout engines handed over: assigns it rollout engine ranks over them
        and the iterator's placement, picks the model runners `selector` covers, queries their configs and Mooncake
        shards, and checks each model replica against the weights its ranks publish. Replicas, the transport and the
        transfer buffers carry over from earlier connects."""
        self.rollout_engines = rollout_engines
        assignments = assign_rollout_engine_ranks(parallel_state, placement, engine_gpu_counts)
        self.is_sender = bool(assignments)

        if self.is_sender:
            runner_roles = query_runner_roles(rollout_engines, assignments, selector)
            configs_by_runner_role = {
                runner_role: query_rollout_engine_rank_configs(rollout_engines, assignments, runner_role)
                for runner_role in runner_roles
            }
            if self._transport is None:
                self._transport = MooncakeTransport()
            remote_shards_by_runner_role = self._transport.connect(rollout_engines, assignments, runner_roles)
            self._runner_senders = [
                self._connect_runner(
                    runner_role, configs_by_runner_role[runner_role], remote_shards_by_runner_role[runner_role]
                )
                for runner_role in runner_roles
            ]
            if self._transfer_buffers is None:
                self._transfer_buffers = self._create_transfer_buffers()

    def _connect_runner(
        self,
        runner_role: str,
        configs_by_rollout_engine_rank: dict[int, RolloutEngineRankConfig],
        remote_shards_by_rollout_engine_rank: dict[int, list[RemoteShard]],
    ) -> _RunnerSender:
        if runner_role not in self._model_replicas_by_runner_role:
            self._model_replicas_by_runner_role[runner_role] = ModelReplicas(model_path=self.args.hf_checkpoint)
        model_replicas = self._model_replicas_by_runner_role[runner_role]
        replica_targets = []
        for rollout_engine_rank, remote_shards in remote_shards_by_rollout_engine_rank.items():
            model_replica = model_replicas.get_or_build(configs_by_rollout_engine_rank[rollout_engine_rank])
            for remote_shard in remote_shards:
                assert_replica_matches_shard(
                    model_replica,
                    remote_shard.published_nbytes_by_name,
                    published_by=(
                        f"the {runner_role} of rollout engine {remote_shard.rollout_engine_ind} "
                        f"rank {rollout_engine_rank}"
                    ),
                )
            replica_targets.append(_ReplicaTarget(model_replica, remote_shards))
        return _RunnerSender(model_replicas, ModelParamStager(), replica_targets)

    def _create_transfer_buffers(self) -> TransferBuffers:
        largest_param_nbytes = max(
            layout.occupied_nbytes
            for runner_sender in self._runner_senders
            for layout in runner_sender.model_replicas.transfer_buffer_param_layouts.values()
        )
        return TransferBuffers(
            _NUM_TRANSFER_BUFFERS,
            max(self.args.update_weight_buffer_size, largest_param_nbytes),
            device=self._transport.transfer_buffer_device,
            register_memory=self._transport.register_memory,
        )


def _raise_if_any_write_failed(pending_writes: list[tuple[RemoteShard, Future]], timeout: float) -> None:
    _, unfinished_writes = concurrent.futures.wait([write for _, write in pending_writes], timeout=timeout)
    failures = []
    for remote_shard, write in pending_writes:
        if write in unfinished_writes:
            reason = f"still running after {timeout}s"
        elif write.exception() is not None:
            reason = repr(write.exception())
        else:
            continue
        failures.append(
            f"the {remote_shard.runner_role} of rollout engine {remote_shard.rollout_engine_ind} rank "
            f"{remote_shard.rollout_engine_rank} (session {remote_shard.session_id}): {reason}"
        )
    if failures:
        raise RuntimeError(f"{len(failures)} of {len(pending_writes)} p2p writes failed:\n" + "\n".join(failures))
