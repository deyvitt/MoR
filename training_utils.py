#___________________________________________________________________________

# training_utils.py (sample)

import torch

import torch.nn as nn

from torch.optim import AdamW

from torch.optim.lr_scheduler import CosineAnnealingLR

class MoRTrainingManager:

    """Training manager for MoR transformers"""

    

    def init(

        self, 

        model: MoRTransformer, 

        base_lr: float = 1e-4,

        router_lr: float = 5e-4,

        aux_loss_weight: float = 0.01

    ):

        self.model = model

        self.aux_loss_weight = aux_loss_weight

        

        # Separate optimizers for base model and routers

        base_params = []

        router_params = []

        

        for name, param in model.named_parameters():

            if 'router' in name:

                router_params.append(param)

            else:

                base_params.append(param)

        

        self.base_optimizer = AdamW(base_params, lr=base_lr)

        self.router_optimizer = AdamW(router_params, lr=router_lr)

        

        self.scheduler = CosineAnnealingLR(self.base_optimizer, T_max=1000)

    

    def compute_auxiliary_loss(self) -> torch.Tensor:

        """Compute auxiliary loss to encourage balanced routing"""

        aux_loss = 0.0

        

        for layer in self.model.layers:

            if hasattr(layer, 'router'):

                stats = layer.router.get_routing_stats()

                recursion_rate = stats['recursion_rate']

                

                # Encourage moderate recursion rate (not too high, not too low)

                target_rate = 0.3

                aux_loss += (recursion_rate - target_rate) ** 2

        

        return torch.tensor(aux_loss, requires_grad=True)

    

    def training_step(

        self, 

        batch: torch.Tensor, 

        targets: torch.Tensor,

        criterion: nn.Module

    ) -> Dict[str, float]:

        """Single training step"""

        

        # Forward pass

        outputs = self.model(batch)

        main_loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

        

        # Auxiliary loss

        aux_loss = self.compute_auxiliary_loss()

        total_loss = main_loss + self.aux_loss_weight * aux_loss

        

        # Backward pass

        self.base_optimizer.zero_grad()

        self.router_optimizer.zero_grad()

        

        total_loss.backward()

        

        # Gradient clipping

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        

        self.base_optimizer.step()

        self.router_optimizer.step()

        self.scheduler.step()

        

        return {

            'main_loss': main_loss.item(),

            'aux_loss': aux_loss.item(),

            'total_loss': total_loss.item()

        }

#________________________________________________________________________________

