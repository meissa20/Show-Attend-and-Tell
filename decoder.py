import torch
import torch.nn as nn
from attention import Attention


class Decoder(nn.Module):
    def __init__(self, vocabulary_size, encoder_dim, tf=False):
        super(Decoder, self).__init__()
        self.use_tf = tf                           # Teacher forcing used during training, model uses ground truth captions.

        self.vocabulary_size = vocabulary_size
        self.encoder_dim = encoder_dim

        self.init_h = nn.Linear(encoder_dim, 512)
        self.init_c = nn.Linear(encoder_dim, 512)
        self.tanh = nn.Tanh()

        self.f_beta = nn.Linear(512, encoder_dim)  # That linear layer lets the model learn over training when to pay attention to the image 
        self.sigmoid = nn.Sigmoid()                # and when to trust its internal language memory.

        self.deep_output = nn.Linear(512, vocabulary_size)
        self.dropout = nn.Dropout()

        self.attention = Attention(encoder_dim)
        self.embedding = nn.Embedding(vocabulary_size, 512)   # Converts each word in the vocab into a vector of size 512
        self.lstm = nn.LSTMCell(512 + encoder_dim, 512)

    def forward(self, img_features, captions):

        batch_size = img_features.size(0)

        h, c = self.get_init_lstm_state(img_features)
        max_timespan = max([len(caption) for caption in captions]) - 1

        prev_words = torch.zeros(batch_size, 1).long().cuda()
        if self.use_tf:
            embedding = self.embedding(captions) if self.training else self.embedding(prev_words)  
        else:
            embedding = self.embedding(prev_words)  # Teacher forcing OFF,  model uses its own predicted words.

        preds = torch.zeros(batch_size, max_timespan, self.vocabulary_size).cuda()
        alphas = torch.zeros(batch_size, max_timespan, img_features.size(1)).cuda()
        for t in range(max_timespan):
            context, alpha = self.attention(img_features, h)
            gate = self.sigmoid(self.f_beta(h)) # Sometimes the model might not want to rely too much on the image → maybe the word is something predictable from the sentence alone (like “a”, “the”).
            gated_context = gate * context      # Other times, the model really needs the image → like when deciding “dog” vs “cat”.
                                                # shape (B, encoder_dim)

            if self.use_tf and self.training:
                lstm_input = torch.cat((embedding[:, t], gated_context), dim=1)     # Just extract the vector of the word at t and conc with the gated context to feed it to LSTM
            else:
                embedding = embedding.squeeze(1) if embedding.dim() == 3 else embedding # shape of embedding(prev) (B, 1, 512)
                lstm_input = torch.cat((embedding, gated_context), dim=1)          

            h, c = self.lstm(lstm_input, (h, c))
            output = self.deep_output(self.dropout(h))  # (B, t, Vocab_size) 

            preds[:, t] = output  # store all predicted word scores over time.
            alphas[:, t] = alpha  # stores attention distribution (say 49 image regions)

            if not self.training or not self.use_tf:
                embedding = self.embedding(output.max(1)[1].reshape(batch_size, 1)) # .max(1) highest score in row, [1] for indicies
                                                                                    # convert the predicted word index back into its 512-d vector
        return preds, alphas

    def get_init_lstm_state(self, img_features):
        avg_features = img_features.mean(dim=1)     # Normally, an LSTM starts with some arbitrary hidden state and cell state  (like zeros).
                                                    # But in image captioning, we don’t want the LSTM to start blind. 
        c = self.init_c(avg_features)               # Instead, we let the CNN features of the image decide what the “starting memory” and “starting thought” of the LSTM should be.
        c = self.tanh(c)

        h = self.init_h(avg_features)
        h = self.tanh(h)

        return h, c

    def caption(self, img_features, beam_size):
        """
        We use beam search to construct the best sentences following a
        similar implementation as the author in
        https://github.com/kelvinxu/arctic-captions/blob/master/generate_caps.py
        """
        prev_words = torch.zeros(beam_size, 1).long()

        sentences = prev_words
        top_preds = torch.zeros(beam_size, 1)
        alphas = torch.ones(beam_size, 1, img_features.size(1))

        completed_sentences = []
        completed_sentences_alphas = []
        completed_sentences_preds = []

        step = 1
        h, c = self.get_init_lstm_state(img_features)

        while True:
            embedding = self.embedding(prev_words).squeeze(1)
            context, alpha = self.attention(img_features, h)
            gate = self.sigmoid(self.f_beta(h))
            gated_context = gate * context

            lstm_input = torch.cat((embedding, gated_context), dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            output = self.deep_output(h)
            output = top_preds.expand_as(output) + output

            if step == 1:
                top_preds, top_words = output[0].topk(beam_size, 0, True, True)
            else:
                top_preds, top_words = output.view(-1).topk(beam_size, 0, True, True)
            prev_word_idxs = top_words / output.size(1)
            next_word_idxs = top_words % output.size(1)

            sentences = torch.cat((sentences[prev_word_idxs], next_word_idxs.unsqueeze(1)), dim=1)
            alphas = torch.cat((alphas[prev_word_idxs], alpha[prev_word_idxs].unsqueeze(1)), dim=1)

            incomplete = [idx for idx, next_word in enumerate(next_word_idxs) if next_word != 1]
            complete = list(set(range(len(next_word_idxs))) - set(incomplete))

            if len(complete) > 0:
                completed_sentences.extend(sentences[complete].tolist())
                completed_sentences_alphas.extend(alphas[complete].tolist())
                completed_sentences_preds.extend(top_preds[complete])
            beam_size -= len(complete)

            if beam_size == 0:
                break
            sentences = sentences[incomplete]
            alphas = alphas[incomplete]
            h = h[prev_word_idxs[incomplete]]
            c = c[prev_word_idxs[incomplete]]
            img_features = img_features[prev_word_idxs[incomplete]]
            top_preds = top_preds[incomplete].unsqueeze(1)
            prev_words = next_word_idxs[incomplete].unsqueeze(1)

            if step > 50:
                break
            step += 1

        idx = completed_sentences_preds.index(max(completed_sentences_preds))
        sentence = completed_sentences[idx]
        alpha = completed_sentences_alphas[idx]
        return sentence, alpha
