import torch
from model import MoeTransformer

def create_simple_model():
    """Create a simple MoE transformer model with default parameters."""
    model = MoeTransformer(
        depth=2,                 # Number of transformer blocks
        n_vocab=1000,            # Vocabulary size
        emd_dim=128,             # Embedding dimension
        num_q_heads=4,           # Number of query heads
        num_kv_heads=2,          # Number of key/value heads
        v_dim=32,                # Value dimension
        k_dim=32,                # Key dimension
        hidden_dim=256,          # Hidden dimension in feed-forward networks
        num_experts=4,           # Number of experts in MoE
        active_experts=2,        # Number of active experts per token
        expert_capacity=0.25,    # Capacity of each expert (as fraction of tokens)
        ff_bias=True,            # Use bias in feed-forward networks
        attn_bias=True,          # Use bias in attention
        attn_dropout=0.1,        # Attention dropout rate
        attn_resid_dropout=0.1,  # Residual dropout rate in attention
    )
    return model

def main():
    # Create model
    model = create_simple_model()
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create sample input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Set model to evaluation mode
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        logits, _ = model(input_ids)
        print(f"Output shape: {logits.shape}")
    
    # Example of generating one token at a time
    print("\nAutoregressive generation example:")
    input_seq = torch.tensor([[42, 100, 200]])  # Starting sequence
    kv_caches = {}
    
    model.eval()
    with torch.no_grad():
        for i in range(5):  # Generate 5 more tokens
            # Get predictions and updated KV cache
            logits, kv_caches = model(input_seq, kv_caches=kv_caches)
            
            # Get the last token's prediction (for the next token)
            next_token_logits = logits[:, -1, :]
            
            # Sample from the distribution (or just take the argmax)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            print(f"Generated token: {next_token.item()}")
            
            # Prepare for next iteration - use only the new token as input
            input_seq = next_token.unsqueeze(0)

if __name__ == "__main__":
    main() 