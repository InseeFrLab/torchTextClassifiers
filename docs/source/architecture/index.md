# Architecture

This section explains the design and architecture of torchTextClassifiers.

```{toctree}
:maxdepth: 2

overview
```

## Overview

torchTextClassifiers is built on a **modular, component-based pipeline** that balances simplicity for beginners with flexibility for advanced users.

The core pipeline consists of four main components:

1. **Tokenizer**: Converts text strings into numerical tokens
2. **Text Embedder**: Creates semantic embeddings from tokens (with optional attention)
3. **Categorical Handler**: Processes additional categorical features (optional)
4. **Classification Head**: Produces final class predictions

This design allows you to:

- Understand the clear data flow through the model
- Mix and match components for your specific needs
- Start simple and add complexity as required
- Use the high-level API or drop down to PyTorch for full control

## Quick Links

- {doc}`overview`: Complete architecture explanation with examples
- {doc}`../api/index`: API reference for all components

## Design Philosophy

The architecture follows these principles:

**Modularity**
: Each component (Tokenizer, Embedder, Categorical Handler, Classification Head) is independent and can be used separately or replaced with custom implementations

**Clear Data Flow**
: The pipeline shows exactly how data moves from text input through embeddings to predictions, making the model transparent and understandable

**Composability**
: Components can be mixed and matched to create custom architectures—use text-only, add categorical features, or build entirely custom combinations

**Flexibility**
: Start with the high-level `torchTextClassifiers` wrapper for simplicity, or compose components directly with PyTorch for maximum control

**Type Safety**
: Extensive use of type hints and dataclasses for better IDE support and fewer runtime errors

**Framework Integration**
: All components are standard `torch.nn.Module` objects with seamless PyTorch and PyTorch Lightning integration
