# Copyright The PyTorch Lightning team.
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
import os

import torch

from pytorch_lightning import _logger as log
from pytorch_lightning.overrides import LightningDistributedModule
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.utilities import _SMDIST_AVAILABLE
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities.seed import seed_everything

if _SMDIST_AVAILABLE:
    import smdistributed.dataparallel.torch.distributed as dist
    from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel


class SMDDPPlugin(DDPPlugin):

    distributed_backend = "smddp"

    def setup(self, model):
        pass

    def pre_dispatch(self):
        # TODO: check if needed
        seed = os.environ.get("PL_GLOBAL_SEED")
        if seed is not None:
            seed_everything(int(seed))

        # determine which process we are and world size
        self.set_world_ranks()

        # set warning rank
        rank_zero_only.rank = self.global_rank

        # set up server using proc 0's ip address
        # try to init for 20 times at max in case ports are taken
        # where to store ip_table
        self.init_ddp_connection(self.global_rank, self.world_size)

        # TODO: we moved it to the trainer.fit after calling pre_dispatch
        #   ... need to double check that it is the correct place
        # self.trainer.call_setup_hook(self.model)

        # on world_size=0 let everyone know training is starting
        if self.is_global_zero and not dist.is_initialized():
            print("===" * 10, "Inside the loop")
            log.info("-" * 100)
            log.info(f"distributed_backend={self.distributed_backend}")
            log.info(f"All DDP processes registered. Starting ddp with {self.world_size} processes")
            log.info("-" * 100)

        # set the ranks and devices
        self.dist.rank = self.global_rank
        self.dist.device = self.root_device

        if self.sync_batchnorm:
            self.model = self.configure_sync_batchnorm(self.model)

        # move the model to the correct device
        self.model_to_device()

        self.configure_ddp()

        self.barrier()

    def init_ddp_connection(self, global_rank: int, world_size: int) -> None:

        os.environ["MASTER_ADDR"] = str(self.cluster_environment.master_address())
        os.environ["MASTER_PORT"] = str(self.cluster_environment.master_port())
        os.environ["WORLD_SIZE"] = str(self.cluster_environment.world_size())

        if not dist.is_initialized():
            log.info(f"initializing ddp: GLOBAL_RANK: {global_rank}, MEMBER: {global_rank + 1}/{world_size}")
            dist.init_process_group(self.torch_distributed_backend, rank=global_rank, world_size=world_size)

    def configure_ddp(self):
        self.pre_configure_ddp()
        print("=Device IDs=" * 5, self.determine_ddp_device_ids())
        print("=Local Device IDs=" * 5, dist.get_local_rank())
        self._model = DistributedDataParallel(
            LightningDistributedModule(self.model),
            device_ids=[dist.get_local_rank()],
            **self._ddp_kwargs,
        )
