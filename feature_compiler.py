import numpy as np

class MicrostructureFeatureCompiler:
    def __init__(self, price_levels=20, time_steps=50):
        self.price_levels = price_levels
        self.time_steps = time_steps
        
        # State tracking to calculate changes between tick frames (t-1 and t)
        self.prev_best_bid_p = 0.0
        self.prev_best_bid_v = 0.0
        self.prev_best_ask_p = 0.0
        self.prev_best_ask_v = 0.0
        
        # Rolling historical buffer matrix for the AI: Shape (Channels, Price_Levels, Time_Steps)
        # Channel 0: Normalized Bids, Channel 1: Normalized Asks, Channel 2: Rolling OFI
        self.feature_tensor = np.zeros((3, self.price_levels, self.time_steps), dtype=np.float32)

    def calculate_ofi(self, best_bid_p, best_bid_v, best_ask_p, best_ask_v):
        """
        Implements the formal Quantitative Equation for Order Flow Imbalance (OFI).
        Measures net supply/demand shifts across the immediate top of the book.
        """
        # Calculate Bid demand shifts
        if best_bid_p > self.prev_best_bid_p:
            delta_v_bid = best_bid_v
        elif best_bid_p < self.prev_best_bid_p:
            delta_v_bid = -self.prev_best_bid_v
        else:
            delta_v_bid = best_bid_v - self.prev_best_bid_v
            
        # Calculate Ask supply shifts
        if best_ask_p < self.prev_best_ask_p:
            delta_v_ask = best_ask_v
        elif best_ask_p > self.prev_best_ask_p:
            delta_v_ask = -self.prev_best_ask_v
        else:
            delta_v_ask = best_ask_v - self.prev_best_ask_v

        # Save current states for the next incoming evaluation frame
        self.prev_best_bid_p = best_bid_p
        self.prev_best_bid_v = best_bid_v
        self.prev_best_ask_p = best_ask_p
        self.prev_best_ask_v = best_ask_v

        # Net OFI is buying pressure minus selling pressure
        return delta_v_bid - delta_v_ask

    def compile_tensor_frame(self, current_bids, current_asks):
        """
        Transforms raw order book arrays into standardized, normalized tensors
        ready for PyTorch / ONNX execution.
        """
        if not current_bids or not current_asks:
            return None

        # Extract values into fixed-length numeric arrays
        bid_prices = np.array([x[0] for x in current_bids], dtype=np.float32)
        bid_volumes = np.array([x[1] for x in current_bids], dtype=np.float32)
        ask_prices = np.array([x[0] for x in current_asks], dtype=np.float32)
        ask_volumes = np.array([x[1] for x in current_asks], dtype=np.float32)

        # Compute immediate OFI metric using top-of-book levels
        ofi_value = self.calculate_ofi(bid_prices[0], bid_volumes[0], ask_prices[0], ask_volumes[0])

        # Step historical steps forward in time inside the circular buffer (rolling window)
        self.feature_tensor = np.roll(self.feature_tensor, shift=-1, axis=2)

        # Vectorized rolling Z-Score normalization to prevent predictive data leakage
        # Map raw volumes into stationary normal distributions
        mean_bid_v, std_bid_v = bid_volumes.mean(), (bid_volumes.std() + 1e-6)
        mean_ask_v, std_ask_v = ask_volumes.mean(), (ask_volumes.std() + 1e-6)
        
        normalized_bids = (bid_volumes - mean_bid_v) / std_bid_v
        normalized_asks = (ask_volumes - mean_ask_v) / std_ask_v

        # Insert fresh computations into the newest time slot (last index index)
        self.feature_tensor[0, :, -1] = normalized_bids
        self.feature_tensor[1, :, -1] = normalized_asks
        self.feature_tensor[2, :, -1] = ofi_value  # Broadcasts across levels for sequence representation

        return self.feature_tensor