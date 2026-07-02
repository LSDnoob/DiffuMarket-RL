import asyncio
import json
import time
import sys
import numpy as np

# Immutable explicit connection string defined at the module root level
BINANCE_WSS_ENDPOINT = "wss://fstream.binance.com/public/ws/btcusdt@depth"

import websockets
from feature_compiler import MicrostructureFeatureCompiler
from inference_worker import GPUInferenceWorker

class MarketDataEngine:
    def __init__(self, target_depth=20):
        self.target_depth = target_depth
        self.bids = {}
        self.asks = {}
        
        # Instantiate the modular mathematical matrix compiler
        self.compiler = MicrostructureFeatureCompiler(price_levels=20, time_steps=50)

        # Initialize the hardware-accelerated GPU inference session
        self.gpu_worker = GPUInferenceWorker(model_path="microstructure_cnn.onnx")

        # Storage parameters for latest calculated tracking vectors
        self.latest_signal = "NEUTRAL"
        self.latest_confidence = 0.0
        self.inference_latency_ms = 0.0
        
        self.message_count = 0
        self.last_metrics_dump = time.time()

    def update_book(self, side, price, quantity):
        book = self.bids if side == "bids" else self.asks
        if quantity == 0.0:
            book.pop(price, None)
        else:
            book[price] = quantity

    def get_top_depth(self):
        sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[:self.target_depth]
        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])[:self.target_depth]
        return sorted_bids, sorted_asks

    async def log_performance(self):
        """Monitors system parameters, listing calculations and GPU inference results."""
        signal_labels = {0: "🛑 SELL / DOWN", 1: "⏳ NEUTRAL", 2: "🚀 BUY / UP"}

        while True:
            await asyncio.sleep(1.0)
            now = time.time()
            elapsed = now - self.last_metrics_dump
            
            if elapsed >= 1.0:
                msg_per_sec = self.message_count / elapsed
                
                print(f"\n--- [Quant System Metrics Engine] ---")
                print(f"Data Throughput     : {msg_per_sec:.2f} ticks/sec")
                print(f"AI Prediction Signal: {self.latest_signal}")
                print(f"Model Confidence    : {self.latest_confidence * 100:.2f}%")
                print(f"GPU Core Latency    : {self.inference_latency_ms:.2f} ms")
                print(f"--------------------------------------")

                self.message_count = 0
                self.last_metrics_dump = now

    async def stream_market_data(self):
        signal_map = {0: "🛑 SELL / DOWN", 1: "⏳ NEUTRAL", 2: "🚀 BUY / UP"}
        print(f"Opening network socket layer connection to exchange...")
        
        # Force exact connection to the root endpoint string literal
        uri_target = str(BINANCE_WSS_ENDPOINT).strip()
        print(f"Opening network layer connection to: {uri_target}")
        
        async with websockets.connect(uri_target) as websocket:
            print("WebSocket handshake complete. Running pipeline calculations...")
            while True:
                try:
                    raw_payload = await websocket.recv()
                    data = json.loads(raw_payload)
                    self.message_count += 1
                    
                    if 'b' in data:
                        for bid in data['b']:
                            self.update_book("bids", float(bid[0]), float(bid[1]))
                    if 'a' in data:
                        for ask in data['a']:
                            self.update_book("asks", float(ask[0]), float(ask[1]))
                    
                    # Convert pricing dictionaries to sorted structures for the feature compiler
                    top_bids, top_asks = self.get_top_depth()
                    if len(top_bids) == self.target_depth and len(top_asks) == self.target_depth:
                        # 1. Compile mathematical tensor matrices on the CPU
                        live_tensor = self.compiler.compile_tensor_frame(top_bids, top_asks)

                        if live_tensor is not None:
                            # 2. Measure exactly how fast the GPU processes the math matrix
                            start_inf = time.perf_counter()
                            
                            class_id, conf = self.gpu_worker.predict_signal(live_tensor)
                            
                            end_inf = time.perf_counter()
                            
                            # Update system global metrics
                            self.latest_signal = signal_map[class_id]
                            self.latest_confidence = conf
                            self.inference_latency_ms = (end_inf - start_inf) * 1000

                except websockets.exceptions.ConnectionClosed:
                    print("\nConnection dropped. Re-establishing socket socket session...")
                    await asyncio.sleep(2)
                    break
                except Exception as e:
                    print(f"\nPipeline execution anomaly: {e}")

    async def start(self):
        await asyncio.gather(
            self.stream_market_data(),
            self.log_performance()
        )

if __name__ == "__main__":
    engine = MarketDataEngine(target_depth=20)
    try:
        asyncio.run(engine.start())
    except KeyboardInterrupt:
        print("\nPipeline terminated safely")
