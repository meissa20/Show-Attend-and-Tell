import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, encoder_dim):
        super(Attention, self).__init__()
        self.U = nn.Linear(512, 512)            # instead of just using the raw hidden state, the model learns a new representation of it that might be more useful for alignment with image features.
        self.W = nn.Linear(encoder_dim, 512)
        self.v = nn.Linear(512, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(1)

    def forward(self, img_features, hidden_state):
        U_h = self.U(hidden_state).unsqueeze(1) # (B, 1, 512)
                                                # In attention, we want to compare the decoder’s hidden state with the encoder’s image features.
                                                # But the hidden state (from the RNN) and the image features (from CNN) may not be in the same “representation space.”
        W_s = self.W(img_features)              # So we apply learnable linear layers (U and W) to project both into a common 512-dim space before combining them.
        att = self.tanh(W_s + U_h)              # With tanh, the representation becomes bounded and smoother, which prevents exploding activations.
        e = self.v(att).squeeze(2)              # So (batch, num_pixels, 1) → (batch, num_pixels), Now you have one scalar score per pixel, per image, per batch.
        alpha = self.softmax(e)                 # how important is this pixel (feature location) for the current hidden state?
        context = (img_features * alpha.unsqueeze(2)).sum(1)  # (batch_size, encoder_dim) Each feature vector is scaled by its corresponding attention weight.
        return context, alpha
