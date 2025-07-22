# MoR
Modifying Transformers to have Mixture of Recursion
This is just an exploration of this new method to inference transformer models. This is still on early stage
and we will add more details as we go:

Usage Examples
The following codes are sample codes to use this MoR

#________________________________________________________________________

# Initialize model

model = MoRTransformer(

    vocab_size=50000,

    d_model=512,

    num_layers=12,

    router_layers=[0, 1, 2, 3, 5, 7]  # Add routers to specific layers

)

# Training setup

training_manager = MoRTrainingManager(model)

criterion = nn.CrossEntropyLoss()

# Training loop

for epoch in range(num_epochs):

    for batch, targets in dataloader:

        losses = training_manager.training_step(batch, targets, criterion)

        

        # Log routing statistics

        if step % 100 == 0:

            routing_stats = model.get_routing_stats()

            print(f"Routing stats: {routing_stats}")

#___________________________________________________________________

Advanced Configuration
The following is how you can implement more advanced configuration of this MoR system

#_____________________________________________________________________________________

# Custom router configuration for different layers

router_configs = {

    0: {'max_recursions': 4, 'hidden_dim': 256},  # Early layer: more recursions

    1: {'max_recursions': 3, 'hidden_dim': 256},

    2: {'max_recursions': 3, 'hidden_dim': 128},

    4: {'max_recursions': 2, 'hidden_dim': 128},  # Later layer: fewer recursions

}

# Initialize with custom configs

model = MoRTransformer(

    vocab_size=50000,

    d_model=512,

    num_layers=12,

    router_layers=list(router_configs.keys())

)

# Apply custom configurations

for layer_idx, config in router_configs.items():

    if hasattr(model.layers[layer_idx], 'router'):

        for key, value in config.items():

            setattr(model.layers[layer_idx].router, key, value)

#_____________________________________________________________________________


