#___________________________________________________________________________________

# router.py (sample)

import torch

import torch.nn as nn

import torch.nn.functional as F

from typing import Callable, Optional, Dict, Any

import logging

class MoRRouter(nn.Module):

    """Mixture of Recursion Router for dynamic layer routing in transformers.

    Args:

        input_dim: Dimension of input hidden states

        hidden_dim: Hidden dimension of router network

        max_recursions: Maximum number of recursions allowed

        decision_threshold: Threshold for routing decisions

        use_attention: Whether to use attention-based routing"""

    

    def init(

        self, 

        input_dim: int, 

        hidden_dim: int = 256,

        max_recursions: int = 3,

        decision_threshold: float = 0.5,

        use_attention: bool = False

    ):

        super().__init__()

        self.input_dim = input_dim

        self.hidden_dim = hidden_dim

        self.max_recursions = max_recursions

        self.decision_threshold = decision_threshold

        self.recursion_count = 0

        

        if use_attention:

            self.gate = self._build_attention_gate()

        else:

            self.gate = self._build_mlp_gate()

        

        # Optional: Learnable recursion embeddings

        self.recursion_embeddings = nn.Embedding(max_recursions + 1, input_dim)

        

        # Metrics tracking

        self.register_buffer('total_decisions', torch.tensor(0))

        self.register_buffer('recurse_decisions', torch.tensor(0))

    

    def buildmlp_gate(self) -> nn.Module:

        """Build MLP-based routing gate"""

        return nn.Sequential(

            nn.Linear(self.input_dim, self.hidden_dim),

            nn.LayerNorm(self.hidden_dim),

            nn.ReLU(),

            nn.Dropout(0.1),

            nn.Linear(self.hidden_dim, self.hidden_dim // 2),

            nn.ReLU(),

            nn.Linear(self.hidden_dim // 2, 3),  # [recurse, forward, skip]

            nn.Softmax(dim=-1)

        )

    

    def buildattention_gate(self) -> nn.Module:

        """Build attention-based routing gate"""

        return nn.MultiheadAttention(

            embed_dim=self.input_dim,

            num_heads=8,

            dropout=0.1,

            batch_first=True

        )

    

    def get_routing_signal(self, hidden_states: torch.Tensor) -> torch.Tensor:

        """Extract routing signal from hidden states"""

        # Mean pooling across sequence length

        if hidden_states.dim() == 3:  # [batch, seq_len, hidden_dim]

            routing_signal = hidden_states.mean(dim=1)

        else:

            routing_signal = hidden_states

        

        # Add recursion depth information

        recursion_emb = self.recursion_embeddings(

            torch.tensor(self.recursion_count, device=hidden_states.device)

        )

        routing_signal = routing_signal + recursion_emb

        

        return routing_signal

    

    def forward(

        self, 

        hidden_states: torch.Tensor,

        layer_fn: Callable,

        next_layer_fn: Optional[Callable] = None,

        training: bool = True

    ) -> torch.Tensor:

        """

        Forward pass with routing decision

        

        Args:

            hidden_states: Input hidden states [batch, seq_len, hidden_dim]

            layer_fn: Function to apply current layer

            next_layer_fn: Function to apply next layer (if forwarding)

            training: Whether in training mode

            

        Returns:

            Processed hidden states

        """

        routing_signal = self.get_routing_signal(hidden_states)

        routing_probs = self.gate(routing_signal)  # [batch, 3]

        

        # Update metrics

        self.total_decisions += 1

        

        if training:

            # During training, use soft routing (Gumbel softmax)

            routing_decision = F.gumbel_softmax(

                torch.log(routing_probs + 1e-10), tau=1.0, hard=False

            )

        else:

            # During inference, use hard routing

            routing_decision = torch.zeros_like(routing_probs)

            max_idx = routing_probs.argmax(dim=-1)

            routing_decision.scatter_(1, max_idx.unsqueeze(1), 1.0)

        

        # Apply routing decision

        output = torch.zeros_like(hidden_states)

        

        for batch_idx in range(hidden_states.size(0)):

            decision_probs = routing_decision[batch_idx]

            

            if decision_probs[0] > 0.5 and self.recursion_count < self.max_recursions:

                # Recurse: Apply same layer again

                self.recursion_count += 1

                self.recurse_decisions += 1

                output[batch_idx] = self.forward(

                    hidden_states[batch_idx:batch_idx+1], 

                    layer_fn, 

                    next_layer_fn, 

                    training

                )[0]

                self.recursion_count -= 1

                

            elif decision_probs[1] > 0.5 and next_layer_fn is not None:

                # Forward: Apply next layer

                output[batch_idx] = next_layer_fn(

                    hidden_states[batch_idx:batch_idx+1]

                )[0]

                

            else:

                # Skip or default: Apply current layer once

                output[batch_idx] = layer_fn(

                    hidden_states[batch_idx:batch_idx+1]

                )[0]

        

        return output

    

    def reset_metrics(self):

        """Reset routing metrics"""

        self.total_decisions.zero_()

        self.recurse_decisions.zero_()

    

    def get_routing_stats(self) -> Dict[str, float]:

        """Get routing statistics"""

        if self.total_decisions > 0:

            recursion_rate = (self.recurse_decisions.float() / 

                            self.total_decisions.float()).item()

        else:

            recursion_rate = 0.0

        

        return {

            'total_decisions': self.total_decisions.item(),

            'recurse_decisions': self.recurse_decisions.item(),

            'recursion_rate': recursion_rate

        }

#___________________________________________________________________________

