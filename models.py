import torch
import torch.nn as nn
import numpy as np
import copy
import math
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer,
                          BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)
import logging

logger = logging.getLogger(__name__)

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
                 'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
                 'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer)}


def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e+6))


def build_or_load_gen_model(args):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
    print(tokenizer)
    if args.model_type == 'roberta':
        encoder = model_class.from_pretrained(args.model_name_or_path, config=config)

        # Original repo implementation Decoder:
        # decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        # decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        ## New implementation:
        # decoder_layer = DecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads, ffn_hidden=2048, drop_prob=0.1)
        # decoder = Decoder(decoder_layer, num_layers=6)
        # decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        ## Our decoder from scratch.
        decoder = TransformerDecoder(config.vocab_size, 6, config.hidden_size, config.num_attention_heads, d_ff=2048, dropout=0.1)
        
        ## Decoder model repo: ours
        # decoder = Decoder(config.vocab_size, 256, d_model=config.hidden_size, ffn_hidden=2048, n_head=config.num_attention_heads, n_layers=6, drop_prob=0.1, device='cpu')

        ## dec_voc_size, max_len, d_model, n_head, n_layers, drop_prob, device
        
        model = Seq2Seq(encoder=encoder, decoder=decoder, config=config,
                        beam_size=args.beam_size, max_length=args.max_target_length,
                        sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)
    else:
        model = model_class.from_pretrained(args.model_name_or_path)

    logger.info("Finish loading model [%s] from %s", get_model_size(model), args.model_name_or_path)

    if args.load_model_path is not None:
        logger.info("Reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    return config, model, tokenizer


# https://github.com/microsoft/CodeBERT/blob/master/CodeBERT/code2nl/model.py
class Seq2Seq(nn.Module):
    """
        Build Seqence-to-Sequence.

        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model.
        * `beam_size`- beam size for beam search.
        * `max_length`- max length of target for beam search.
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search.
    """

    def __init__(self, encoder, decoder, config, beam_size=None, max_length=None, sos_id=None, eos_id=None):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()

        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id

    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.encoder.embeddings.word_embeddings)

    def forward(self, source_ids=None, source_mask=None, target_ids=None, target_mask=None, args=None):
        outputs = self.encoder(source_ids, attention_mask=source_mask)
        encoder_output = outputs[0].permute([1, 0, 2]).contiguous()

        if target_ids is not None:
            attn_mask = -1e4 * (1 - self.bias[:target_ids.shape[1], :target_ids.shape[1]])
            tgt_embeddings = self.encoder.embeddings(target_ids).permute([1, 0, 2]).contiguous()


            # tgt_embeddings = tgt_embeddings.long()
            # encoder_output = encoder_output.long()
            # attn_mask = attn_mask.long()


            # print("type: tgt_embeddings", tgt_embeddings.dtype)
            # print("type: encoder op", encoder_output.dtype)
            # print("type: attn_mask", attn_mask.dtype)
            # print("type: source_mask", source_mask.dtype)


            # Original repo implementation Decoder:
            # out = self.decoder(tgt_embeddings, encoder_output, tgt_mask=attn_mask,
            #                    memory_key_padding_mask=~source_mask)
            
            ## Decoder from our Repo:
            # out = self.decoder(tgt_embeddings, encoder_output, attn_mask, ~source_mask)

            ## Our decoder from scratch.
            out = self.decoder(tgt_embeddings, encoder_output, source_mask)

            # memory_key_padding_mask=(1 - source_mask).bool())
            hidden_states = torch.tanh(self.dense(out)).permute([1, 0, 2]).contiguous()
            lm_logits = self.lm_head(hidden_states)
            # Shift so that tokens < n predict n
            active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                            shift_labels.view(-1)[active_loss])

            outputs = loss, loss * active_loss.sum(), active_loss.sum()
            return outputs
        else:
            # Predict
            preds = []
            zero = torch.cuda.LongTensor(1).fill_(0)
            for i in range(source_ids.shape[0]):
                context = encoder_output[:, i:i + 1]
                context_mask = source_mask[i:i + 1, :]
                beam = Beam(self.beam_size, self.sos_id, self.eos_id)
                input_ids = beam.getCurrentState()
                context = context.repeat(1, self.beam_size, 1)
                context_mask = context_mask.repeat(self.beam_size, 1)
                for _ in range(self.max_length):
                    if beam.done():
                        break
                    attn_mask = -1e4 * (1 - self.bias[:input_ids.shape[1], :input_ids.shape[1]])
                    tgt_embeddings = self.encoder.embeddings(input_ids).permute([1, 0, 2]).contiguous()

                    # Decoder from original repo:
                    # out = self.decoder(tgt_embeddings, context, tgt_mask=attn_mask,
                    #                    memory_key_padding_mask=~context_mask)


                    ## Our decoder from scratch.
                    out = self.decoder(tgt_embeddings, context, context_mask)

                    # tgt_embeddings = tgt_embeddings.type(torch.LongTensor)
                    # tgt_embeddings = tgt_embeddings.cuda()
                    
                    ## Decoder from our Repo:
                    # out = self.decoder(tgt_embeddings, context, attn_mask, ~context_mask)
                    
                    # memory_key_padding_mask=(1 - context_mask).bool())
                    out = torch.tanh(self.dense(out))
                    hidden_states = out.permute([1, 0, 2]).contiguous()[:, -1, :]
                    out = self.lsm(self.lm_head(hidden_states)).data
                    beam.advance(out)
                    input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                    input_ids = torch.cat((input_ids, beam.getCurrentState()), -1)
                hyp = beam.getHyp(beam.getFinal())
                pred = beam.buildTargetTokens(hyp)[:self.beam_size]
                pred = [torch.cat([x.view(-1) for x in p] + [zero] * (self.max_length - len(p))).view(1, -1) for p in
                        pred]
                preds.append(torch.cat(pred, 0).unsqueeze(0))

            preds = torch.cat(preds, 0)
            return preds


class Beam(object):
    def __init__(self, size, sos, eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                           .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished = []
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i))
            unfinished.sort(key=lambda a: -a[0])
            self.finished += unfinished[:self.size - len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps = []
        for _, timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j + 1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps

    def buildTargetTokens(self, preds):
        sentence = []
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok == self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence


##########################################################
## Our decoder model from scratch

# class DecoderLayer(nn.Module):
#     def __init__(self, d_model, nhead, ffn_hidden, drop_prob):
#         super(DecoderLayer, self).__init__()
#         self.self_attention = nn.MultiheadAttention(d_model, nhead)
#         self.norm1 = LayerNorm(d_model=d_model)
#         self.dropout1 = nn.Dropout(p=drop_prob)

#         self.enc_dec_attention = nn.MultiheadAttention(d_model, nhead)
#         self.norm2 = LayerNorm(d_model=d_model)
#         self.dropout2 = nn.Dropout(p=drop_prob)

#         self.ffn = PositionwiseFeedForward(d_model=d_model, ffn_hidden=ffn_hidden, drop_prob=drop_prob)
#         self.norm3 = LayerNorm(d_model=d_model)
#         self.dropout3 = nn.Dropout(p=drop_prob)

#     def forward(self, tgt, memory, tgt_mask, memory_key_padding_mask):
#         _x = tgt
#         x = self.self_attention(tgt, tgt, tgt, tgt_mask)
#         x = self.dropout1(x)
#         x = self.norm1(x + _x)

#         _x = x
#         x = self.enc_dec_attention(x, memory, memory, tgt_mask)
#         x = self.dropout2(x)
#         x = self.norm2(x + _x)

#         _x = x
#         x = self.ffn(x)
#         x = self.dropout3(x)

#         return x
            
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, num_layers, d_model, num_heads, d_ff, dropout):
        super().__init__()

        self.num_layers = num_layers
        self.d_model = d_model

        # create the positional encodings
        # self.positional_encodings = nn.Embedding(240, d_model)

        # create the multi-headed attention layer
        # self.multi_headed_attention = nn.MultiheadAttention(d_model, num_heads)

        # create the feedforward layer
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        # create the final linear layer
        self.output_linear = nn.Linear(d_model, d_model)

        # create the dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, encoder_output, mask):
        # add the positional encodings to the input
        # x = input + self.positional_encodings(input)
        x = input

        # pass the input through the N decoder layers
        for i in range(self.num_layers):
            x = self.dropout(x)

            # compute the multi-headed attention
            # x, attn = self.multi_headed_attention(x, x, x, mask)

            # pass the result through the feedforward layer
            x = self.feedforward(x)

        # apply the final linear layer and return the result
        return self.output_linear(x)


# class PositionwiseFeedForward(nn.Module):

#     def __init__(self, ffn_hidden, d_model, drop_prob=0.1):
#         super(PositionwiseFeedForward, self).__init__()
#         self.linear1 = nn.Linear(d_model, ffn_hidden)
#         self.linear2 = nn.Linear(ffn_hidden, d_model)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(p=drop_prob)

#     def forward(self, x):
#         x = self.linear1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.linear2(x)
#         return x


# class LayerNorm(nn.Module):
#     def __init__(self, d_model, eps=1e-12):
#         super(LayerNorm, self).__init__()
#         self.gamma = nn.Parameter(torch.ones(d_model))
#         self.beta = nn.Parameter(torch.zeros(d_model))
#         self.eps = eps

#     def forward(self, x):
#         mean = x.mean(-1, keepdim=True)
#         std = x.std(-1, keepdim=True)
#         # '-1' means last dimension. 

#         out = (x - mean) / (std + self.eps)
#         out = self.gamma * out + self.beta
#         return out


