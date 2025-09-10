import torch
import torch.nn.functional as F
import torch.nn as nn
import math

class DecoderOnlyTransformer(nn.Module):
    
    def __init__(self, num_tokens, d_model, max_len, device, encoder_dim, num_heads=12, num_layers=6, dropout=0.1):
        
        super().__init__()

        self.device = device
        self.d_model = d_model                  # d_model = number of embedding values per token.
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        
        self.token_embedding = nn.Embedding(num_embeddings=num_tokens, 
                                            embedding_dim=d_model)     
        
        self.position_encoding = PositionEncoding(d_model=d_model, 
                                                  max_len=max_len, 
                                                  dropout=dropout,
                                                  device=device)
        
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model=d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(d_model)
        self.encoder_projection = nn.Linear(encoder_dim, d_model)
        self.output_projection = nn.Linear(in_features=d_model, out_features=num_tokens)
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:                             # p.dim() > 1: Weight matrices (2D tensors)
                                                        # p.dim() == 1: Bias vectors, embedding parameters, layer norm parameters
                nn.init.xavier_uniform_(p)
        
    def forward(self, token_ids, encoder_features, encoder_mask=None):

        batch_size, seq_len = token_ids.size(0), token_ids.size(1)
        
        # Token embeddings + positional encoding
        word_embeddings = self.token_embedding(token_ids.long())   
        word_embeddings = self.position_encoding(word_embeddings)
     

        encoder_features = self.encoder_projection(encoder_features)  

        # Create causal mask for self-attention
        causal_mask = self.create_causal_mask(seq_len).to(self.device)
        
        # Pass through decoder layers
        hidden_states = word_embeddings
        cross_attention_weights = []

        for layer in self.decoder_layers:
            hidden_states, layer_weights = layer(
                hidden_states, 
                encoder_features, 
                causal_mask=causal_mask,
                encoder_mask=encoder_mask
            )
            cross_attention_weights.append(layer_weights)
        
        # Final layer norm and output projection
        hidden_states = self.final_norm(hidden_states)
        logits = self.output_projection(hidden_states)
        cross_attention_weights = torch.stack(cross_attention_weights, dim=0)
        
        return logits, cross_attention_weights
    
    def create_causal_mask(self, seq_len):

        mask = torch.tril(torch.ones(seq_len, seq_len))
        mask = mask == 0 
        return mask
    
    def generate(self, encoder_features, start_token, end_token, max_len=50, temperature=0.5):

        self.eval()
        device = encoder_features.device
        batch_size = encoder_features.size(0)
        all_attention_weights = []
        
        # Initialize with start token
        tokens = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
        
        with torch.no_grad():
            for step in range(max_len):
                # Forward pass
                logits, cross_weights = self.forward(tokens, encoder_features)

                # We take the last layer and average across heads
                last_layer_attn = cross_weights[-1]  
                current_step_attn = last_layer_attn[:, :, step, :] 
                current_step_attn = current_step_attn.mean(dim=1) 
                all_attention_weights.append(current_step_attn)

                # Get next token probabilities
                next_token_logits = logits[:, -1, :] / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Sample next token (or use greedy decoding if temperature=0)
                if temperature == 0:
                    next_token = torch.argmax(probs, dim=-1)
                else:
                    next_token = torch.multinomial(probs, 1).squeeze(-1)
                
                # Append to sequence
                tokens = torch.cat([tokens, next_token.unsqueeze(-1)], dim=1)
                
                # Check for end token
                if (next_token == end_token).all():
                    break

        if all_attention_weights:
            alpha = torch.stack(all_attention_weights, dim=0)
        else:
            alpha = torch.zeros(1, batch_size, encoder_features.size(1), device=device)


        return tokens, alpha
    
    def caption(self, encoder_features, start_token, end_token):

        tokens, alpha = self.generate(encoder_features, start_token, end_token, temperature=0.0)
        sentence = tokens[0].tolist() 
        alpha = alpha[:, 0, :]  

        # Return first batch item as a list (original code expects this format)
        return sentence, alpha


class TransformerDecoderLayer(nn.Module):
    
    def __init__(self, d_model, num_heads, dropout=0.1):

        super().__init__()

        self.multi_head_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.cross_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
)
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, hidden_states, encoder_features, causal_mask=None, encoder_mask=None):
        
        attention, _ = self.multi_head_attention(query=hidden_states, 
                                              key=hidden_states, 
                                              value=hidden_states, 
                                              mask=causal_mask
        )

        hidden_states = self.norm1(hidden_states + self.dropout(attention))

        cross_attention, cross_weights = self.cross_attention(query=hidden_states,
                                               key=encoder_features,
                                               value=encoder_features,
                                               mask=encoder_mask
        )

        hidden_states = self.norm2(hidden_states + self.dropout(cross_attention))
        
        feedforward = self.ffn(hidden_states)
        hidden_states = self.norm3(hidden_states + feedforward)
        
        return hidden_states, cross_weights
    
        
class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model, num_heads):

        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        
        self.out_proj = nn.Linear(d_model, d_model)

        
    def forward(self, query, key, value, mask=None):

        batch_size, seq_len_q = query.size(0), query.size(1)
        seq_len_k = key.size(1)

        k = self.W_k(key)
        v = self.W_v(value)
        q = self.W_q(query)

        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)  

        if mask is not None:
            if mask.dim() == 2: 
                mask = mask.unsqueeze(0).unsqueeze(0)  
            elif mask.dim() == 2 and mask.size(1) == seq_len_k:  
                mask = mask.unsqueeze(1).unsqueeze(1) 
            
            scores = scores.masked_fill(mask, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, v)
        
        # Concatenate heads and put through final linear layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )
        
        
        output = self.out_proj(attention_output)
        avg_attention = attention_weights.mean(dim=1)

        return output, avg_attention
    

class PositionEncoding(nn.Module):
    
    def __init__(self, d_model, max_len, device, dropout=0.1):

        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        
        self.pos_encoding = torch.zeros(max_len, d_model, device = device)
        self.pos_encoding.requires_grad = False
        position = torch.arange(0, max_len, device=device) 
        position = position.float().unsqueeze(dim=1)
        
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        
        self.pos_encoding[:, 0::2] = torch.sin(position / (10000 ** (_2i / d_model)))    # even columns are sine
        self.pos_encoding[:, 1::2] = torch.cos(position / (10000 ** (_2i / d_model)))    # odd columns are cosine

    def forward(self, word_embeddings):
        
        batch_size, seq_len, d_model = word_embeddings.size(0), word_embeddings.size(1), word_embeddings.size(2)
        # print(f"size 0 : {self.pos_encoding.size(0)}, size 1 : {self.pos_encoding.size(1)}, size 2 : {self.pos_encoding.size(2)}")
        pos_enc = self.pos_encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        word_embeddings = word_embeddings + pos_enc
        return self.dropout(word_embeddings)