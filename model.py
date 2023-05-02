import math
from collections import defaultdict
from tqdm.auto import tqdm
# Import PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.cuda.amp import autocast
from torch.nn import functional as F
# Import Huggingface
from transformers import PretrainedConfig, AutoModel, AutoTokenizer, BartForConditionalGeneration

class TransformerModel(nn.Module):
    def __init__(self, src_max_len: int = 150, trg_max_len: int = 150,
                 dropout: float = 0.3):
        super().__init__()

        """
        Initialize augmenter model

        Args:
            encoder_config (dictionary): encoder transformer's configuration
            d_latent (int): latent dimension size
            device (torch.device):
        Returns:
            log_prob (torch.Tensor): log probability of each word
            mean (torch.Tensor): mean of latent vector
            log_var (torch.Tensor): log variance of latent vector
            z (torch.Tensor): sampled latent vector
        """
        self.src_max_len = src_max_len
        self.trg_max_len = trg_max_len
        self.dropout = nn.Dropout(dropout)

        # Hyper-parameter setting
        self.model_config = PretrainedConfig.from_pretrained('facebook/bart-base')
        self.d_hidden = self.model_config.d_model
        self.d_embedding = int(self.d_hidden / 2)
        self.vocab_num = self.model_config.vocab_size

        # Encoder model setting
        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-base", forced_bos_token_id=0).model
        self.encoder_model = self.model.encoder

        # Latent model setting
        self.latent_encoder = nn.Linear(self.d_hidden, self.d_embedding)
        self.latent_decoder = nn.Linear(self.d_embedding, self.d_hidden)

        # Decoder model setting
        self.decoder_model = self.model.decoder
        self.decoder_linear = nn.Linear(self.d_hidden, self.d_embedding)
        self.decoder_norm = nn.LayerNorm(self.d_embedding, eps=1e-12)
        self.decoder_linear2 = nn.Linear(self.d_embedding, self.vocab_num)

        # Tokenizer Setting
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
        self.pad_idx = self.tokenizer.pad_token_id
        self.decoder_start_token_id = self.model_config.decoder_start_token_id
        self.bos_idx = self.tokenizer.bos_token_id
        self.eos_idx = self.tokenizer.eos_token_id

    def forward(self, src_input_ids, src_attention_mask, trg_input_ids, trg_attention_mask, non_pad_position):

        # Encoding
        encoder_out = self.encoder_model(input_ids=src_input_ids,
                                         attention_mask=src_attention_mask)
        encoder_out = encoder_out['last_hidden_state'] # (batch_size, seq_len, d_hidden)

        # Latent setting
        encoder_out = self.latent_encoder(encoder_out)
        encoder_out = self.latent_decoder(encoder_out)

        # Decoding
        # decoder_input_ids = shift_tokens_right(
        #     trg_input_ids, self.pad_idx, self.decoder_start_token_id
        # )

        decoder_outputs = self.decoder_model(
            input_ids=trg_input_ids,
            encoder_hidden_states=encoder_out,
            encoder_attention_mask=src_attention_mask
        )
        decoder_outputs = decoder_outputs['last_hidden_state'] # (batch_size, seq_len, d_hidden)
        if non_pad_position is not None:
            decoder_outputs = decoder_outputs[non_pad_position]

        decoder_outputs = self.dropout(F.gelu(self.decoder_linear(decoder_outputs)))
        decoder_outputs = self.decoder_linear2(self.decoder_norm(decoder_outputs))

        return decoder_outputs

    def generate(self, encoder_out, latent_out, attention_mask, beam_size, beam_alpha, repetition_penalty, device):
        # Input, output setting
        batch_size = encoder_out.size(0)
        src_seq_size = encoder_out.size(1)
        every_batch = torch.arange(0, beam_size * batch_size, beam_size, device=device)

        # Total Hidden States
        if self.encoder_out_cross_attention:
            if self.latent_out_mix_ratio == 0:
                encoder_hidden_states = encoder_out
            elif self.encoder_out_mix_ratio == 0:
                encoder_hidden_states = latent_out
            else:
                encoder_hidden_states = torch.add((self.encoder_out_mix_ratio * encoder_out), (self.latent_out_mix_ratio * latent_out.unsqueeze(1)))
        else:
            encoder_hidden_states = None
            src_key_padding_mask = None

        # Expanding
        if self.encoder_out_cross_attention:
            if self.latent_out_mix_ratio == 0:
                src_key_padding_mask = attention_mask.view(batch_size, 1, -1)
                src_key_padding_mask = src_key_padding_mask.repeat(1, beam_size, 1)
                src_key_padding_mask = src_key_padding_mask.view(-1, src_seq_size)
            else:
                src_key_padding_mask = None

            encoder_hidden_states = encoder_hidden_states.unsqueeze(1) # (batch_size, 1, seq_len, d_hidden)
            encoder_hidden_states = encoder_hidden_states.repeat(1, beam_size, 1, 1) # (batch_size, beam_size, seq_len, d_hidden)
            encoder_hidden_states = encoder_hidden_states.view(-1, src_seq_size, self.d_hidden) # (batch_size * beam_size, seq_len, d_hidden)

        # Scores save vector & decoding list setting
        scores_save = torch.zeros(beam_size * batch_size, 1).to(device) # (batch_size * beam_size, 1)
        top_k_scores = torch.zeros(beam_size * batch_size, 1).to(device) # (batch_size * beam_size, 1)
        complete_seqs = defaultdict(list)
        complete_ind = set()

        # Decoding start token setting
        seqs = torch.tensor([[self.decoder_start_token_id]], dtype=torch.long, device=device)
        seqs = seqs.repeat(beam_size * batch_size, 1).contiguous() # (batch_size * beam_size, 1)

        for step in range(self.src_max_len):
            # Decoding sentence
            decoder_outputs = self.decoder(
                input_ids=seqs,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=src_key_padding_mask
            )
            decoder_outputs = decoder_outputs['last_hidden_state']

            if self.encoder_out_to_augmenter:
                # Add encoder_out (batch_size, seq_len, d_hidden) for decoder_outputs (batch_size * k, seq_len, d_hidden)
                encoder_out_for_augmenter = encoder_out.sum(dim=1).unsqueeze(1) # (batch_size, 1, d_hidden)
                encoder_out_for_augmenter = encoder_out_for_augmenter.repeat(1, decoder_outputs.size(1), 1) # (batch_size, seq_len, d_hidden)
                encoder_out_for_augmenter = encoder_out_for_augmenter.unsqueeze(1) # (batch_size, 1, seq_len, d_hidden)
                encoder_out_for_augmenter = encoder_out_for_augmenter.repeat(1, beam_size, 1, 1) # (batch_size, k, seq_len, d_hidden)
                encoder_out_for_augmenter = encoder_out_for_augmenter.view(-1, decoder_outputs.size(1), self.d_hidden) # (batch_size * k, seq_len, d_hidden)

                decoder_outputs = torch.add(decoder_outputs, encoder_out_for_augmenter)

            # Score calculate
            scores = F.gelu(self.decoder_linear(decoder_outputs[:,-1])) # (batch_size * k, d_embedding)
            scores = self.decoder_augmenter(self.decoder_norm(scores)) # (batch_size * k, vocab_num)
            scores = F.log_softmax(scores, dim=1) # (batch_size * k, vocab_num)

            # Repetition Penalty
            if step >= 1 and repetition_penalty != 0:
                next_ix = next_word_inds.view(-1)
                for ix_ in range(len(next_ix)):
                    if scores[ix_][next_ix[ix_]] < 0:
                        scores[ix_][next_ix[ix_]] *= repetition_penalty
                    else:
                        scores[ix_][next_ix[ix_]] /= repetition_penalty

            # Add score
            scores = top_k_scores.expand_as(scores) + scores  # (batch_size * k, vocab_num)
            if step == 0:
                scores = scores[::beam_size] # (batch_size, vocab_num)
                scores[:, self.eos_idx] = float('-inf') # set eos token probability zero in first step
                top_k_scores, top_k_words = scores.topk(beam_size, 1, True, True)  # (batch_size, k) , (batch_size, k)
            else:
                top_k_scores, top_k_words = scores.view(batch_size, -1).topk(beam_size, 1, True, True)

            # Previous and Next word extract
            prev_word_inds = top_k_words // self.vocab_num # (batch_size * k, out_seq)
            next_word_inds = top_k_words % self.vocab_num # (batch_size * k, out_seq)
            top_k_scores = top_k_scores.view(batch_size * beam_size, -1) # (batch_size * k, out_seq)
            top_k_words = top_k_words.view(batch_size * beam_size, -1) # (batch_size * k, out_seq)
            seqs = seqs[prev_word_inds.view(-1) + every_batch.unsqueeze(1).repeat(1, beam_size).view(-1)] # (batch_size * k, out_seq)
            seqs = torch.cat([seqs, next_word_inds.view(beam_size * batch_size, -1)], dim=1) # (batch_size * k, out_seq + 1)

            # Find and Save Complete Sequences Score
            if self.eos_idx in next_word_inds:
                eos_ind = torch.where(next_word_inds.view(-1) == self.eos_idx)
                eos_ind = eos_ind[0].tolist()
                complete_ind_add = set(eos_ind) - complete_ind
                complete_ind_add = list(complete_ind_add)
                complete_ind.update(eos_ind)
                if len(complete_ind_add) > 0:
                    scores_save[complete_ind_add] = top_k_scores[complete_ind_add]
                    for ix in complete_ind_add:
                        complete_seqs[ix] = seqs[ix].tolist()

        # If eos token doesn't exist in sequence
        if 0 in scores_save:
            score_save_pos = torch.where(scores_save == 0)
            for ix in score_save_pos[0].tolist():
                complete_seqs[ix] = seqs[ix].tolist()
            scores_save[score_save_pos] = top_k_scores[score_save_pos]

        # Beam Length Normalization
        lp = torch.tensor([len(complete_seqs[i]) for i in range(batch_size * beam_size)], device=device)
        lp = (((lp + beam_size) ** beam_alpha) / ((beam_size + 1) ** beam_alpha)).unsqueeze(1)
        scores_save = scores_save / lp

        # Predicted and Label processing
        _, ind = scores_save.view(batch_size, beam_size, -1).max(1)
        ind_expand = ind.view(-1) + every_batch
        predicted = [complete_seqs[i] for i in ind_expand.tolist()]

        return predicted

    def generate_sample(self, encoder_out, latent_out, attention_mask, sampling_strategy, device, topk=5, topp=0.9, softmax_temp=3.0):
        # Input, output setting
        batch_size = encoder_out.size(0)
        src_seq_size = encoder_out.size(1)

        # Total Hidden States
        if self.encoder_out_cross_attention:
            src_key_padding_mask = attention_mask
            if self.latent_out_mix_ratio == 0:
                encoder_hidden_states = encoder_out
            elif self.encoder_out_mix_ratio == 0:
                encoder_hidden_states = latent_out
            else:
                encoder_hidden_states = torch.add((self.encoder_out_mix_ratio * encoder_out), (self.latent_out_mix_ratio * latent_out.unsqueeze(1)))
        else:
            encoder_hidden_states = None
            src_key_padding_mask = None

        # Decoding start token setting
        seqs = torch.tensor([[self.decoder_start_token_id]], dtype=torch.long, device=device)
        seqs = seqs.repeat(batch_size, 1).contiguous() # (batch_size, 1)

        for step in range(self.src_max_len):
            # Decoding sentence
            decoder_outputs = self.decoder(
                input_ids=seqs,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=src_key_padding_mask
            )
            decoder_outputs = decoder_outputs['last_hidden_state']

            if self.encoder_out_to_augmenter:
                encoder_out_for_augmenter = encoder_out.sum(dim=1).unsqueeze(1) # (batch_size, 1, d_hidden)
                encoder_out_for_augmenter = encoder_out_for_augmenter.repeat(1, decoder_outputs.size(1), 1) # (batch_size, seq_len, d_hidden)

                decoder_outputs = torch.add(decoder_outputs, encoder_out_for_augmenter)

            # Next word probability
            scores = F.gelu(self.decoder_linear(decoder_outputs[:,-1])) # (batch_size, d_embedding)
            scores = self.decoder_augmenter(self.decoder_norm(scores)) # (batch_size, vocab_num)
            
            # Avoid generating <pad> and <s> token and apply softmax temperature
            if step == 0:
                scores[:, self.pad_idx] = float('-inf')
                scores[:, self.bos_idx] = float('-inf')
            scores = scores / softmax_temp
            next_word_prob = F.softmax(scores, dim=1) # (batch_size, vocab_num)

            # Sampling through predetermined strategy
            if sampling_strategy == 'greedy':
                next_word = torch.argmax(next_word_prob, dim=-1)
            elif sampling_strategy == 'multinomial':
                next_word = torch.multinomial(next_word_prob, 1).squeeze(1) # (batch_size)
            elif sampling_strategy == 'topk':
                # Get topk token
                topk_prob, topk_idx = torch.topk(next_word_prob, topk, dim=-1) # (batch_size, topk)
                topk_prob = topk_prob / torch.sum(topk_prob.sum(dim=-1, keepdim=True)) # (batch_size, topk) - normalize probability to sum 1
                next_token_idx = torch.multinomial(topk_prob, 1).squeeze(1) # (batch_size) - sample token from topk token
                next_word = topk_idx[torch.arange(topk_idx.size(0)), next_token_idx] # (batch_size)
            elif sampling_strategy == 'topp':
                # Get topp token
                sorted_prob, sorted_idx = torch.sort(next_word_prob, descending=True, dim=-1)
                sorted_prob = sorted_prob.cumsum(dim=-1) # (batch_size, vocab_num)
                sorted_idx = sorted_idx[sorted_prob <= topp] # (batch_size, vocab_num)
                sorted_prob = sorted_prob[sorted_prob <= topp] # (batch_size, vocab_num)
                sorted_prob = sorted_prob / torch.sum(sorted_prob.sum(dim=-1, keepdim=True)) # (batch_size, vocab_num) - normalize probability to sum 1
                next_token_idx = torch.multinomial(sorted_prob, 1).squeeze(1) # (batch_size) - sample token from topp token
                next_word = sorted_idx[torch.arange(sorted_idx.size(0)), next_token_idx] # (batch_size)
            else:
                raise ValueError('Sampling strategy is not defined')

            # Concatenate generated token to sequence
            next_word = next_word.unsqueeze(1) # (batch_size, 1)
            seqs = torch.cat([seqs, next_word], dim=1) # (batch_size, seq_len + 1)

        # Postprocessing - remove generated tokens after <eos> token
        seqs = seqs.tolist()
        for i in range(batch_size):
            if self.eos_idx in seqs[i]:
                seqs[i] = seqs[i][:seqs[i].index(self.eos_idx) + 1] # remove generated tokens after <eos> token
            else:
                pass # do nothing if <eos> token is not generated

        return seqs

    def generate_topk(self, encoder_out, latent_out, attention_mask, device, topk=5, softmax_temp=5.0):
        # Input, output setting
        batch_size = encoder_out.size(0)
        src_seq_size = encoder_out.size(1)

        # Total Hidden States
        if self.encoder_out_cross_attention:
            if self.latent_out_mix_ratio == 0:
                encoder_hidden_states = encoder_out
            elif self.encoder_out_ratio == 0:
                encoder_hidden_states = latent_out
            else:
                encoder_hidden_states = torch.add((self.encoder_out_ratio * encoder_out), (self.latent_out_mix_ratio * latent_out.unsqueeze(1)))
        else:
            encoder_hidden_states = None
            src_key_padding_mask = None

        # Expanding
        if self.encoder_out_cross_attention:
            src_key_padding_mask = attention_mask.view(batch_size, 1, -1) # (batch_size, 1, src_seq_size)
            src_key_padding_mask = src_key_padding_mask.view(-1, src_seq_size)

            encoder_hidden_states = encoder_hidden_states.view(-1, batch_size, 1, self.d_hidden)
            encoder_hidden_states = encoder_hidden_states.view(src_seq_size, -1, self.d_hidden)

        # BART decoder start token
        seqs = torch.tensor([[self.decoder_start_token_id]], dtype=torch.long, device=device) # (1, 1)
        seqs = seqs.repeat(batch_size, 1) # (batch_size, 1) # <s> token
        # Generate sentence
        for step in range(self.src_max_len):
            decoder_outputs = self.decoder(
                input_ids=seqs,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attention_mask
            )
            decoder_outputs = decoder_outputs['last_hidden_state'] # (batch_size, seq_len, d_embedding)

            # Score calculate
            scores = F.gelu(self.decoder_linear(decoder_outputs)) # (batch_size, seq_len, d_embedding)
            scores = self.decoder_augmenter(self.decoder_norm(scores)) # (batch_size, seq_len, vocab_num)
            next_token_logits = scores[:, -1, :] # (batch_size, vocab_num)

            # Avoid generating <pad> and <s> token
            next_token_logits[:, self.pad_idx] = float('-inf')
            next_token_logits[:, self.bos_idx] = float('-inf')

            # Apply softmax temperature to logits
            next_token_logits = next_token_logits / softmax_temp
            next_token_prob = F.softmax(next_token_logits, dim=-1) # (batch_size, vocab_num)

            # Get topk token
            topk_prob, topk_idx = torch.topk(next_token_prob, topk, dim=-1) # (batch_size, topk)
            topk_prob = topk_prob / torch.sum(topk_prob.sum(dim=-1, keepdim=True)) # (batch_size, topk) - normalize probability to sum 1
            next_token_idx = torch.multinomial(topk_prob, 1).squeeze(1) # (batch_size) - sample token from topk token
            next_token = topk_idx[torch.arange(topk_idx.size(0)), next_token_idx] # (batch_size)

            # Concatenate generated token to sequence
            next_token = next_token.unsqueeze(1) # (batch_size, 1)
            seqs = torch.cat([seqs, next_token], dim=1) # (batch_size, seq_len + 1)

            # If <eos> token is generated, stop generating
            if self.eos_idx in next_token:
                break

        return seqs

def _prepare_bart_decoder_inputs(
    config, input_ids, decoder_input_ids=None, decoder_padding_mask=None, causal_mask_dtype=torch.float32
):
    """Prepare masks that ignore padding tokens in the decoder and a causal mask for the decoder if
    none are provided. This mimics the default behavior in fairseq. To override it pass in masks.
    Note: this is not called during generation
    """
    pad_token_id = config.pad_token_id
    if decoder_input_ids is None:
        decoder_input_ids = shift_tokens_right(input_ids, pad_token_id)
    bsz, tgt_len = decoder_input_ids.size()
    if decoder_padding_mask is None:
        decoder_padding_mask = make_padding_mask(decoder_input_ids, pad_token_id)
    else:
        decoder_padding_mask = invert_mask(decoder_padding_mask)
    causal_mask = torch.triu(fill_with_neg_inf(torch.zeros(tgt_len, tgt_len)), 1).to(
        dtype=causal_mask_dtype, device=decoder_input_ids.device
    )
    return decoder_input_ids, decoder_padding_mask, causal_mask

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

def make_padding_mask(input_ids, padding_idx=1):
    """True for pad tokens"""
    padding_mask = input_ids.eq(padding_idx)
    if not padding_mask.any():
        padding_mask = None
    return padding_mask

def invert_mask(attention_mask):
    assert attention_mask.dim() == 2
    return attention_mask.eq(0)

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a input_ids with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        pe = torch.zeros(max_len, d_model, dtype=torch.float)
        pe.require_grad = False

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
