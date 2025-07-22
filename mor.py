#____________________________________________________________________________

# transformer_with_mor.py (sample)
import torch
import torch.nn as nn
from typing import List, Optional
from router import MoRRouter

class MoRTransformerLayer(nn.Module):
    """Transformer layer enhanced with MoR routing"""
    def init(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1, use_router: bool = True, router_config: Optional[dict] = None):
        super().__init__()

        # Standard transformer components
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # MoR Router
        self.use_router = use_router
        if use_router:
            router_config = router_config or {}
            self.router = MoRRouter(input_dim=d_model, **router_config)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None, next_layer: Optional['MoRTransformerLayer'] = None) -> torch.Tensor:
        """Forward pass with optional MoR routing"""
        def current_layer_fn(x):
            """Apply current layer transformations"""
            # Self-attention
            x2 = self.self_attn(x, x, x, attn_mask=src_mask)[0]
            x = x + self.dropout1(x2)
            x = self.norm1(x)

            # Feedforward
            x2 = self.linear2(self.dropout(F.relu(self.linear1(x))))
            x = x + self.dropout2(x2)
            x = self.norm2(x)
            return x

        def next_layer_fn(x):
            """Apply next layer if available"""
            if next_layer is not None:
                return next_layer(x, src_mask)
            return x

        if self.use_router:
            return self.router(src, current_layer_fn, next_layer_fn, training=self.training)
        else:
            return current_layer_fn(src)

class MoRTransformer(nn.Module):
    """Complete transformer with MoR routing"""
    def init(self, vocab_size: int, d_model: int = 512, nhead: int = 8, num_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1, router_layers: Optional[List[int]] = None):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        # Determine which layers should have routers
        if router_layers is None:
            # Default: Add routers to first 8 layers
            router_layers = list(range(min(8, num_layers)))

        # Create transformer layers
        self.layers = nn.ModuleList([MoRTransformerLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, use_router=(i in router_layers),
                router_config={'hidden_dim': d_model // 2,
                    'max_recursions': 3 if i < 4 else 2
                }
            )
            for i in range(num_layers)
        ])
        self.output_projection = nn.Linear(d_model, vocab_size)
        

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None):
        """Forward pass through MoR transformer"""
        # Embedding and positional encoding
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Pass through layers with routing
        for i, layer in enumerate(self.layers):
            next_layer = self.layers[i + 1] if i + 1 < len(self.layers) else None
            x = layer(x, src_mask, next_layer)
        return self.output_projection(x)

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics from all layers"""
        stats = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'router'):
                stats[f'layer_{i}'] = layer.router.get_routing_stats()
        return stats

#____________________________________________________________________________
