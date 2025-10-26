"""XAI (Explainable AI) hooks for extracting attention and activations.

Provides utilities to:
- Extract attention weights from transformer layers
- Capture intermediate activations
- Visualize attention patterns
- Compute attribution scores
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
from collections import defaultdict


class ActivationHook:
    """Hook to capture activations from a specific layer."""
    
    def __init__(self, module: nn.Module, name: str):
        """Initialize activation hook.
        
        Args:
            module: PyTorch module to hook
            name: Name identifier for this hook
        """
        self.name = name
        self.activations = None
        self.hook = module.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module: nn.Module, input: Tuple, output: torch.Tensor):
        """Hook function to capture output.
        
        Args:
            module: The module being hooked
            input: Input to the module
            output: Output from the module
        """
        if isinstance(output, tuple):
            self.activations = output[0].detach()
        else:
            self.activations = output.detach()
    
    def remove(self):
        """Remove the hook."""
        self.hook.remove()
    
    def get_activations(self) -> Optional[torch.Tensor]:
        """Get captured activations.
        
        Returns:
            Activations tensor or None
        """
        return self.activations


class AttentionHook:
    """Hook to capture attention weights from transformer layers."""
    
    def __init__(self, module: nn.Module, name: str):
        """Initialize attention hook.
        
        Args:
            module: Transformer layer module
            name: Name identifier for this hook
        """
        self.name = name
        self.attention_weights = None
        self.hook = None
        
        # Try to hook the self-attention layer
        if hasattr(module, 'self_attn'):
            self.hook = module.self_attn.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module: nn.Module, input: Tuple, output: Tuple):
        """Hook function to capture attention weights.
        
        Args:
            module: The attention module
            input: Input to the module
            output: Output tuple (output, attention_weights)
        """
        # MultiheadAttention returns (output, attention_weights) when need_weights=True
        if isinstance(output, tuple) and len(output) >= 2:
            self.attention_weights = output[1].detach()
        else:
            self.attention_weights = None
    
    def remove(self):
        """Remove the hook."""
        if self.hook is not None:
            self.hook.remove()
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get captured attention weights.
        
        Returns:
            Attention weights tensor or None
        """
        return self.attention_weights


class GradientHook:
    """Hook to capture gradients for attribution analysis."""
    
    def __init__(self, module: nn.Module, name: str):
        """Initialize gradient hook.
        
        Args:
            module: PyTorch module to hook
            name: Name identifier for this hook
        """
        self.name = name
        self.gradients = None
        self.hook = module.register_full_backward_hook(self.hook_fn)
    
    def hook_fn(self, module: nn.Module, grad_input: Tuple, grad_output: Tuple):
        """Hook function to capture gradients.
        
        Args:
            module: The module being hooked
            grad_input: Gradients with respect to inputs
            grad_output: Gradients with respect to outputs
        """
        if grad_output[0] is not None:
            self.gradients = grad_output[0].detach()
    
    def remove(self):
        """Remove the hook."""
        self.hook.remove()
    
    def get_gradients(self) -> Optional[torch.Tensor]:
        """Get captured gradients.
        
        Returns:
            Gradients tensor or None
        """
        return self.gradients


class XAIHooks:
    """Manager for XAI hooks on a model."""
    
    def __init__(self, model: nn.Module):
        """Initialize XAI hooks manager.
        
        Args:
            model: PyTorch model to instrument
        """
        self.model = model
        self.activation_hooks = {}
        self.attention_hooks = {}
        self.gradient_hooks = {}
    
    def register_activation_hook(self, module_name: str, module: nn.Module):
        """Register an activation hook on a module.
        
        Args:
            module_name: Name for this module
            module: PyTorch module to hook
        """
        hook = ActivationHook(module, module_name)
        self.activation_hooks[module_name] = hook
    
    def register_attention_hook(self, layer_name: str, layer: nn.Module):
        """Register an attention hook on a transformer layer.
        
        Args:
            layer_name: Name for this layer
            layer: Transformer layer to hook
        """
        hook = AttentionHook(layer, layer_name)
        self.attention_hooks[layer_name] = hook
    
    def register_gradient_hook(self, module_name: str, module: nn.Module):
        """Register a gradient hook on a module.
        
        Args:
            module_name: Name for this module
            module: PyTorch module to hook
        """
        hook = GradientHook(module, module_name)
        self.gradient_hooks[module_name] = hook
    
    def register_all_attention_hooks(self):
        """Register attention hooks on all transformer layers in the model."""
        def register_recursive(module: nn.Module, prefix: str = ''):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                # Check if this is a transformer layer
                if isinstance(child, nn.TransformerEncoderLayer):
                    self.register_attention_hook(full_name, child)
                
                # Recurse
                register_recursive(child, full_name)
        
        register_recursive(self.model)
    
    def register_all_activation_hooks(self, target_modules: List[str]):
        """Register activation hooks on specified module types.
        
        Args:
            target_modules: List of module type names (e.g., ['Linear', 'ReLU'])
        """
        def register_recursive(module: nn.Module, prefix: str = ''):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                # Check if this module type should be hooked
                if type(child).__name__ in target_modules:
                    self.register_activation_hook(full_name, child)
                
                # Recurse
                register_recursive(child, full_name)
        
        register_recursive(self.model)
    
    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Get all captured activations.
        
        Returns:
            Dictionary mapping module names to activation tensors
        """
        activations = {}
        for name, hook in self.activation_hooks.items():
            acts = hook.get_activations()
            if acts is not None:
                activations[name] = acts
        return activations
    
    def get_attention_weights(self) -> Dict[str, torch.Tensor]:
        """Get all captured attention weights.
        
        Returns:
            Dictionary mapping layer names to attention weight tensors
        """
        attention_weights = {}
        for name, hook in self.attention_hooks.items():
            weights = hook.get_attention_weights()
            if weights is not None:
                attention_weights[name] = weights
        return attention_weights
    
    def get_gradients(self) -> Dict[str, torch.Tensor]:
        """Get all captured gradients.
        
        Returns:
            Dictionary mapping module names to gradient tensors
        """
        gradients = {}
        for name, hook in self.gradient_hooks.items():
            grads = hook.get_gradients()
            if grads is not None:
                gradients[name] = grads
        return gradients
    
    def remove_all_hooks(self):
        """Remove all registered hooks."""
        for hook in self.activation_hooks.values():
            hook.remove()
        for hook in self.attention_hooks.values():
            hook.remove()
        for hook in self.gradient_hooks.values():
            hook.remove()
        
        self.activation_hooks.clear()
        self.attention_hooks.clear()
        self.gradient_hooks.clear()


class AttentionAnalyzer:
    """Analyzer for attention patterns."""
    
    @staticmethod
    def compute_attention_rollout(attention_weights: List[torch.Tensor]) -> torch.Tensor:
        """Compute attention rollout across layers.
        
        Attention rollout aggregates attention across layers to show
        which input tokens influence the output most.
        
        Args:
            attention_weights: List of attention weight tensors from each layer
                Each tensor shape: (batch, num_heads, seq_len, seq_len)
        
        Returns:
            Rolled out attention of shape (batch, seq_len, seq_len)
        """
        if not attention_weights:
            return None
        
        # Average over heads
        rolled_attention = attention_weights[0].mean(dim=1)  # (batch, seq_len, seq_len)
        
        # Add identity matrix (residual connections)
        eye = torch.eye(rolled_attention.size(1), device=rolled_attention.device)
        eye = eye.unsqueeze(0).expand_as(rolled_attention)
        rolled_attention = 0.5 * rolled_attention + 0.5 * eye
        
        # Multiply attention matrices across layers
        for attention in attention_weights[1:]:
            attention_avg = attention.mean(dim=1)
            attention_avg = 0.5 * attention_avg + 0.5 * eye
            rolled_attention = torch.matmul(attention_avg, rolled_attention)
        
        return rolled_attention
    
    @staticmethod
    def compute_attention_flow(attention_weights: List[torch.Tensor],
                              source_indices: List[int],
                              target_index: int) -> np.ndarray:
        """Compute attention flow from source tokens to target token.
        
        Args:
            attention_weights: List of attention weight tensors
            source_indices: List of source token indices
            target_index: Target token index
            
        Returns:
            Flow scores for each source token
        """
        rolled = AttentionAnalyzer.compute_attention_rollout(attention_weights)
        
        # Extract flows to target
        flows = rolled[0, target_index, source_indices].cpu().numpy()
        
        return flows
    
    @staticmethod
    def get_top_attended_tokens(attention_weights: torch.Tensor,
                               query_idx: int,
                               top_k: int = 5) -> Tuple[List[int], List[float]]:
        """Get top-k most attended tokens for a query token.
        
        Args:
            attention_weights: Attention tensor (batch, num_heads, seq_len, seq_len)
            query_idx: Index of query token
            top_k: Number of top tokens to return
            
        Returns:
            Tuple of (token_indices, attention_scores)
        """
        # Average over batch and heads
        avg_attention = attention_weights.mean(dim=(0, 1))  # (seq_len, seq_len)
        
        # Get attention scores for query token
        query_attention = avg_attention[query_idx]  # (seq_len,)
        
        # Get top-k
        top_scores, top_indices = torch.topk(query_attention, k=top_k)
        
        return top_indices.tolist(), top_scores.tolist()


class GradientAttribution:
    """Gradient-based attribution methods for XAI."""
    
    @staticmethod
    def integrated_gradients(model: nn.Module,
                            input_tensor: torch.Tensor,
                            target_class: int,
                            baseline: Optional[torch.Tensor] = None,
                            steps: int = 50) -> torch.Tensor:
        """Compute Integrated Gradients attribution.
        
        Args:
            model: PyTorch model
            input_tensor: Input tensor (1, input_dim)
            target_class: Target class index
            baseline: Baseline input (zeros if None)
            steps: Number of integration steps
            
        Returns:
            Attribution scores (1, input_dim)
        """
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, steps, device=input_tensor.device)
        interpolated_inputs = []
        for alpha in alphas:
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated_inputs.append(interpolated)
        
        interpolated_inputs = torch.cat(interpolated_inputs, dim=0)
        
        # Compute gradients
        interpolated_inputs.requires_grad_(True)
        outputs = model(interpolated_inputs)
        
        # Handle different output types
        if isinstance(outputs, dict):
            outputs = outputs['discard']
        elif isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # Get gradients
        target_outputs = outputs[:, target_class]
        gradients = torch.autograd.grad(target_outputs.sum(), interpolated_inputs)[0]
        
        # Average gradients and multiply by input difference
        avg_gradients = gradients.mean(dim=0, keepdim=True)
        attributions = avg_gradients * (input_tensor - baseline)
        
        return attributions
    
    @staticmethod
    def compute_saliency_map(model: nn.Module,
                            input_tensor: torch.Tensor,
                            target_class: int) -> torch.Tensor:
        """Compute saliency map (gradient magnitude).
        
        Args:
            model: PyTorch model
            input_tensor: Input tensor (1, input_dim)
            target_class: Target class index
            
        Returns:
            Saliency scores (1, input_dim)
        """
        input_tensor.requires_grad_(True)
        
        outputs = model(input_tensor)
        
        # Handle different output types
        if isinstance(outputs, dict):
            outputs = outputs['discard']
        elif isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # Compute gradient
        target_output = outputs[0, target_class]
        gradient = torch.autograd.grad(target_output, input_tensor)[0]
        
        # Saliency is absolute value of gradient
        saliency = gradient.abs()
        
        return saliency


