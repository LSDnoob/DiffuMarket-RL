import numpy as np
import onnxruntime as ort

class GPUInferenceWorker:
    def __init__(self, model_path='microstructure_cnn.onnx'):
        self.model_path = model_path
        
        # Configure ONNX Runtime to target  NVIDIA GPU via CUDA
        # It fallbacks gracefully to CPU only if execution providers mismatch
        self.providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024, # Limit to 2GB VRAM
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider'
        ]

        print("Initializing GPU Inference Engine via ONNX Runtime...")
        self.session = ort.InferenceSession(self.model_path, providers=self.providers)
        
        # Get input and output names from the exported model metadata
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Log active hardware targeting
        active_provider = self.session.get_providers()[0]
        print(f"--- [Inference Engine Ready] ---")
        print(f"Active Compute Provider: {active_provider}")
        print(f"--------------------------------\n")
    
    def predict_signal(self, feature_tensor):
        """
        Receives a (3, 20, 50) numpy feature matrix, adds batch dimensions, 
        and runs highly parallelized forward matrix operations on the GPU.
        """
        if feature_tensor is None:
            return 1, 0.50 # Return Neutral signal if tensor isn't populated yet
        
        # Add the explicit Batch Dimension required by the CNN: Shape becomes (1, 3, 20, 50)
        input_data = np.expand_dims(feature_tensor, axis=0).astype(np.float32)
        
        # Execute the forward pass on hardware accelerator cores
        raw_outputs = self.session.run([self.output_name], {self.input_name: input_data})
        logits = raw_outputs[0][0] # Extract the prediction class vectors
        
        # Calculate Softmax probabilities on the logits vector manually for system speed
        exp_logits = np.exp(logits - np.max(logits)) # Subtract max to prevent numerical overflow
        probabilities = exp_logits / np.sum(exp_logits)
        
        # Extract highest probability index class
        predicted_class_id = int(np.argmax(probabilities))
        confidence_score = float(probabilities[predicted_class_id])
        
        # Map IDs to semantic directional trading signals
        # 0: Down / Sell Pressure | 1: Neutral | 2: Up / Buy Pressure
        return predicted_class_id, confidence_score