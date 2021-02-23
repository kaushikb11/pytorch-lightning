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
from pytorch_lightning import _logger as log
from pytorch_lightning.overrides import LightningDistributedModule
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.utilities import _SMDIST_AVAILABLE

if _SMDIST_AVAILABLE:
    import smdistributed.dataparallel.torch.distributed as dist
    from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel


class SMDDPPlugin(DDPPlugin):

    def init_ddp_connection(self, global_rank: int, world_size: int) -> None:

        os.environ["MASTER_ADDR"] = str(self.cluster_environment.master_address())
        os.environ["MASTER_PORT"] = str(self.cluster_environment.master_port())
        os.environ["WORLD_SIZE"] = str(self.cluster_environment.world_size())

        if not dist.is_initialized():
            log.info(f"initializing ddp: GLOBAL_RANK: {global_rank}, MEMBER: {global_rank + 1}/{world_size}")
            dist.init_process_group(self.torch_distributed_backend, rank=global_rank, world_size=world_size)

    def configure_ddp(self):
        self.pre_configure_ddp()
        self._model = DistributedDataParallel(
            LightningDistributedModule(self.model),
            device_ids=self.determine_ddp_device_ids(),
            **self._ddp_kwargs,
        )
