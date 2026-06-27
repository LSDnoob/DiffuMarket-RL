import asyncio
import json
import time
import sys
import numpy as np

# Immutable explicit connection string defined at the module root level
BINANCE_WSS_ENDPOINT = "wss://fstream.binance.com/public/ws/btcusdt@depth"

import websockets
from feature_compiler import MicrostructureFeatureCompiler

class MarketDataEngine:
    def __init__(self, target_depth=20):
        self.target_depth = target_depth
        self.bids = {}
        self.asks = {}
        
        # Instantiate the modular mathematical matrix compiler
        self.compiler = MicrostructureFeatureCompiler(price_levels=20, time_steps=50)
        
        self.message_count = 0
        self.start_time = time.time()
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
        while True:
            await asyncio.sleep(1.0)
            now = time.time()
            elapsed = now - self.last_metrics_dump
            
            if elapsed >= 1.0:
                msg_per_sec = self.message_count / elapsed
                top_bids, top_asks = self.get_top_depth()
                
                best_bid = top_bids[0][0] if top_bids else 0.0
                best_ask = top_asks[0][0] if top_asks else 0.0
                spread = best_ask - best_bid
                
                print(f"\n--- [Metrics Engine] ---")
                print(f"Throughput    : {msg_per_sec:.2f} msg/sec")
                print(f"Total Parsed  : {self.message_count}")
                print(f"Best Bid/Ask  : {best_bid:.2f} / {best_ask:.2f} (Spread: {spread:.2f})")
                print(f"Book Levels   : Bids={len(self.bids)} | Asks={len(self.asks)}")
                print(f"------------------------")
                
                self.message_count = 0
                self.last_metrics_dump = now

    async def stream_market_data(self):
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
                        # Stream parsed structures into mathematical numpy vectors
                        live_tensor = self.compiler.compile_tensor_frame(top_bids, top_asks)
                        
                        if self.message_count % 5 == 0 and live_tensor is not None:
                            print(f"[Compiler State] Formed Input Tensor Image Matrix -> Shape: {live_tensor.shape}", end='\r')

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
        print("\nEngine safely terminated by user request.")
