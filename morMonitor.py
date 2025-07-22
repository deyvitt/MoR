#___________________________________________________________________

class MoRMonitor:

    """Monitor MoR model performance and routing behavior"""

    

    def init(self, model: MoRTransformer):

        self.model = model

        self.routing_history = []

    

    def log_routing_decision(self, layer_idx: int, decision: str, input_complexity: float):

        """Log routing decisions for analysis"""

        self.routing_history.append({

            'layer': layer_idx,

            'decision': decision,

            'complexity': input_complexity,

            'timestamp': time.time()

        })

    

    def analyze_routing_patterns(self):

        """Analyze routing patterns over time"""

        import pandas as pd

        

        df = pd.DataFrame(self.routing_history)

        

        # Analyze recursion patterns by complexity

        complexity_routing = df.groupby(['complexity', 'decision']).size().unstack(fill_value=0)

        

        # Plot routing decisions over layers

        layer_decisions = df.groupby(['layer', 'decision']).size().unstack(fill_value=0)

        

        return {

            'complexity_routing': complexity_routing,

            'layer_decisions': layer_decisions

        }

#_________________________________________________________________________

