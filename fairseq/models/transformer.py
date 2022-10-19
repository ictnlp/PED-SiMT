# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from os import device_encoding
import pdb
import random
from selectors import EpollSelector
from warnings import resetwarnings
from numpy import dtype
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from fairseq.modules.capsule_network import (CapsuleLayer)
from fairseq.modules.capsule import (ContextualCapsuleLayer, MultiInputPositionwiseFeedForward, Generator, WordPredictor, MultiTargetNMTCriterion)
from fairseq.modules.capsule import (convert_to_past_labels, convert_to_future_labels)
from torch.nn.parameter import Parameter
DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model('transformer')
class TransformerModel(FairseqEncoderDecoderModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    @classmethod
    def hub_models(cls):
        # fmt: off
        return {
            'transformer.wmt14.en-fr': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2',
            'transformer.wmt16.en-de': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2',
            'transformer.wmt18.en-de': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt18.en-de.ensemble.tar.gz',
            'transformer.wmt19.en-de': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.ensemble.tar.gz',
            'transformer.wmt19.en-ru': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.ensemble.tar.gz',
            'transformer.wmt19.de-en': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.ensemble.tar.gz',
            'transformer.wmt19.ru-en': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.ensemble.tar.gz',
            'transformer.wmt19.en-de.single_model': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.single_model.tar.gz',
            'transformer.wmt19.en-ru.single_model': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.single_model.tar.gz',
            'transformer.wmt19.de-en.single_model': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.single_model.tar.gz',
            'transformer.wmt19.ru-en.single_model': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.single_model.tar.gz',
        }
        # fmt: on

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens, encoder_embed_tokens)
        return TransformerModel(encoder, decoder)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens, encoder_embed_tokens):
        return TransformerDecoder(args, tgt_dict, embed_tokens, encoder_embde_tokens=encoder_embed_tokens)


class TransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))

        self.dropout = args.dropout

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])

        self.layers.extend([
            TransformerEncoderLayer(args)
            for i in range(args.encoder_layers)
        ])

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def forward(self, src_tokens, src_lengths):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions

        x = self.embed_scale * self.embed_tokens(src_tokens)

        if self.embed_positions is not None:
            x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        self_atten_tion = self.buffered_future_mask(x)
        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask, self_attention_mask = self_atten_tion)

        if self.layer_norm:
            x = self.layer_norm(x)

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device or self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]


    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)
        for i in range(len(self.layers)):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(state_dict, "{}.layers.{}".format(name, i))

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens,encoder_embde_tokens=None, no_encoder_attn=False):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))

        self.dropout = args.dropout

        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim

        self.output_embed_dim = args.decoder_output_dim

        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None

        # Capsule Network parameters
        self.pre_capsule_layer_norm = nn.LayerNorm(embed_dim)
        self.final_capsule_layer = ContextualCapsuleLayer(
            num_out_caps=4, num_in_caps=None,
            dim_in_caps=embed_dim,
            dim_out_caps=256,
            dim_context=embed_dim,
            num_iterations=3,
            share_route_weights_for_in_caps=True
        )
        dim_per_part = self.final_capsule_layer.dim_out_caps * (self.final_capsule_layer.num_out_caps // 2)
        self.out_and_cap_ffn = MultiInputPositionwiseFeedForward(
            size=embed_dim, hidden_size=1024, dropout=self.dropout,
            inp_sizes=[dim_per_part, dim_per_part]
        )
        self.out_layer_norm = nn.LayerNorm(embed_dim)
        self.past_generator = Generator(n_words=embed_tokens.weight.size(0),
                                   hidden_size=embed_dim,
                                   padding_idx=embed_tokens.padding_idx)
        self.past_generator.proj.weight = embed_tokens.weight
        self.future_generator = Generator(n_words=encoder_embde_tokens.weight.size(0),
                                   hidden_size=encoder_embde_tokens.weight.size(1),
                                   padding_idx=encoder_embde_tokens.padding_idx)
        self.future_generator.proj.weight = encoder_embde_tokens.weight
        self.wp_past = WordPredictor(generator=self.past_generator,
                                         input_size=dim_per_part,
                                         d_word_vec=embed_dim)
        self.wp_future = WordPredictor(generator=self.future_generator,
                                           input_size=dim_per_part*2,
                                           d_word_vec=encoder_embde_tokens.weight.size(1))
        self.wp_past_loss = MultiTargetNMTCriterion(label_smoothing=args.label_smoothing)
        self.wp_future_loss = MultiTargetNMTCriterion(label_smoothing=args.label_smoothing)
        
        self.de_past = Linear(embed_dim, dim_per_part, bias=False)
        self.de_future =  Linear(embed_dim, dim_per_part, bias=False)
        self.en_future = Linear(embed_dim, dim_per_part, bias=False)
        
        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, embed_dim, padding_idx,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(args, no_encoder_attn)
            for _ in range(args.decoder_layers)
        ])

        self.adaptive_softmax = None

        self.project_out_dim = Linear(embed_dim, self.output_embed_dim, bias=False) \
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights else None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), self.output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)

        if args.decoder_normalize_before and not getattr(args, 'no_decoder_final_norm', False):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None
    def randAnum(self, start, end):
        num = random.uniform(0,1)
        if num < 0.33:
            return 0
        elif num < 0.66:
            return 1
        else:
            return 2
    def forward(self, prev_output_tokens, encoder_out=None, tgt_tokens=None, src_tokens=None, incremental_state=None, new_times=5, reorder=None,step=None,isadd=True, **unused):
        """
        Args:

            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        waitK = new_times
        if step is None:
            tgt_len = tgt_tokens.size(1)
            src_len = src_tokens.size(1)
            zeroNum = 0
            cross_random_mask = None
            till_src_num = new_times
            i = 0
            while i < tgt_len:
                tmp_num = self.randAnum(0, 2)
                if tmp_num == 0:
                    zeroNum += 1
                else:
                    zeroNum = 0
                if zeroNum >= 2:
                    continue
                till_src_num += tmp_num
                i += 1
                step_masked = torch.triu(tgt_tokens.new_ones(1, src_len, device = tgt_tokens.device), till_src_num).bool()
                if cross_random_mask is None:
                    cross_random_mask = step_masked
                else:
                    cross_random_mask = torch.cat((cross_random_mask, step_masked), dim=0)
            new_times = cross_random_mask
            
        '''
        tgt_len = tgt_tokens.size(1)
        src_len = src_tokens.size(1)
        num1 = 1
        cross_random_mask = None
        till_src_num = new_times
        for i in range(tgt_len):
            tmp_num = random.randint(0,2)
            if (tmp_num==num1 and tmp_num == 0):
                tmp_num =random.randint(1,2)
            till_src_num += tmp_num
            step_masked = torch.triu(tgt_tokens.new_ones(1, src_len, device = tgt_tokens.device), till_src_num).bool()
            if cross_random_mask is None:
                cross_random_mask = step_masked
            else:
                cross_random_mask = torch.cat((cross_random_mask, step_masked), dim=0)
            num1 = tmp_num
        '''
        x, extra = self.extract_features(prev_output_tokens, encoder_out, incremental_state, new_times=new_times, reorder=reorder,step=step,isadd=isadd)

        # Integrate capsule outputs into target hidden states
        capsule_query = self.pre_capsule_layer_norm(x)
        capsules, probs = self.final_capsule_layer.forward_sequence(
            encoder_out['encoder_out'].transpose(0,1),
            encoder_out['encoder_padding_mask'],
            new_times,
            capsule_query,
            cache=None
        )
        capsules = capsules.view(capsules.size(0), capsules.size(1), -1)
        (past_caps, future_caps) = torch.chunk(capsules, 2, -1)
        if step is None:
            f_l, l_l = self.cal_capsule_loss(torch.transpose(encoder_out['encoder_out'], dim0=0, dim1=1), x, capsules, waitK, tgt_tokens=tgt_tokens, encoder_padding_mask=encoder_out['encoder_padding_mask'], cross_mask=new_times)
        output = self.out_and_cap_ffn(x, past_caps, future_caps)
        x = self.out_layer_norm(output)
        if step is None:
            first_loss, last_loss = self.cal_wp_loss(past_caps=past_caps, future_caps=torch.cat((past_caps,future_caps),dim=-1), prev_lables=prev_output_tokens, next_labels=tgt_tokens, new_times=waitK, src_tokens=src_tokens,cross_mask=new_times)
        
        x = self.output_layer(x)

        if step is not None:
            probs = probs.contiguous().view(probs.size(0),probs.size(1),probs.size(2),2,-1).sum(dim=-1).view(probs.size(0),probs.size(1),probs.size(2),-1)
            return [x, extra], probs[:,-1,:,:]
        else:
            return (x, extra), first_loss+last_loss, f_l.sum(dim=-1) + l_l.sum(dim=-1)


    def cal_wp_loss(self, past_caps=None, future_caps=None,prev_lables=None, next_labels=None, new_times=None, src_tokens=None, cross_mask=None):
        # Token-Level Loss
        logprobs_past = self.wp_past(past_caps)
        logprobs_future = self.wp_future(future_caps)
        past_labels, past_scores = convert_to_past_labels(next_labels,past_caps.dtype)
        future_labels, future_scores = convert_to_future_labels(next_labels,new_times,src_tokens,past_caps.dtype, cross_mask=cross_mask)
        params_wploss_past = dict(
            inputs=logprobs_past,
            labels=past_labels,
            target_scores=past_scores,
            update=True)
        params_wploss_future = dict(
            inputs=logprobs_future,
            labels=future_labels,
            target_scores=future_scores,
            update=True)
        first_loss = self.wp_past_loss(inputs=logprobs_past,labels=past_labels,target_scores=past_scores,update=True)
        second_loss = self.wp_future_loss(inputs=logprobs_future,labels=future_labels,target_scores=future_scores,update=True)
        return first_loss, second_loss
        
    def cal_capsule_loss(self, encoder_out, decoder_out, capsule_out, new_times, tgt_tokens=None, encoder_padding_mask=None, cross_mask=None):
        # segment-level loss
        batch_tgt_mask = torch.eq(tgt_tokens, 1)  
        batch_size = torch.ones(batch_tgt_mask.size(0), batch_tgt_mask.size(1), device=encoder_out.device).masked_fill(batch_tgt_mask,0)
        batch_size = batch_size.sum(dim=-1).type_as(encoder_out)

        tmp_decoder_outs = torch.cumsum(decoder_out, dim=1)
        tmp_zeros = torch.zeros(1,1,tmp_decoder_outs.size(-1),device=encoder_out.device).expand(tmp_decoder_outs.size(0),1,tmp_decoder_outs.size(-1)).type_as(encoder_out)
        tmp_decoder_outs = torch.cat((tmp_zeros, tmp_decoder_outs[:,:-1,:]),dim=1)
        tmp_decoder_outs = tmp_decoder_outs.masked_fill(batch_tgt_mask.unsqueeze(-1), 0)

        tmp_capsule_outs = capsule_out.reshape(capsule_out.size(0), capsule_out.size(1), 2, -1)
        past_capsule = tmp_capsule_outs[:,:,0,:].squeeze(-2)
        future_capsule = tmp_capsule_outs[:,:,1,:].squeeze(-2)

        tmp_decoder_outs_rev = decoder_out.masked_fill(batch_tgt_mask.unsqueeze(-1), 0)   # 将不存在的目标端掩饰掉
        tmp_decoder_outs_rev = torch.flip(torch.cumsum(torch.flip(tmp_decoder_outs_rev, [1]), dim=1), [1])

        if encoder_padding_mask is not None:
            tmp_encoder_outs_rev = encoder_out.masked_fill(encoder_padding_mask.unsqueeze(-1), 0)
        else:
            tmp_encoder_outs_rev = encoder_out

        time_series = torch.arange(decoder_out.size(1), device=encoder_out.device).type_as(encoder_out)
        time_series = time_series.reshape(1, -1, 1) + 6e-8

        tmp_decoder_outs = (tmp_decoder_outs / time_series).detach()
        first_loss = self.cal_normal_loss(past_capsule.masked_fill(batch_tgt_mask.unsqueeze(-1),0) - self.de_past(tmp_decoder_outs))
        first_loss = first_loss / batch_size.detach()

        time_series = torch.ones(batch_tgt_mask.size(),device=decoder_out.device).masked_fill(batch_tgt_mask, 0).type_as(encoder_out)
        time_series = torch.flip(torch.cumsum(torch.flip(time_series, [1]), dim=1), [1]) + 6e-8
        tmp_decoder_outs_rev = (tmp_decoder_outs_rev / time_series.unsqueeze(-1)).detach()
        
        srclen = encoder_out.size(1)
        tgtlen = decoder_out.size(1)
        if srclen <= new_times:
            return first_loss, self.cal_normal_loss(future_capsule.masked_fill(batch_tgt_mask.unsqueeze(-1),0) - self.de_future(tmp_decoder_outs_rev)) / batch_size.detach()
        else:
            tmp_encoder_outs_rev = torch.bmm(cross_mask.float().expand(tmp_encoder_outs_rev.size(0), cross_mask.size(0),cross_mask.size(1)), tmp_encoder_outs_rev)
            time_series = torch.zeros(cross_mask.size(), device=tmp_encoder_outs_rev.device)
            time_series = time_series.masked_fill(cross_mask, 1.0).unsqueeze(0).expand(tmp_encoder_outs_rev.size(0), time_series.size(0), time_series.size(1))
            if(encoder_padding_mask is not None):
                encoder_padding_mask = (1 - encoder_padding_mask.float()).unsqueeze(1).expand_as(time_series)
                time_series = time_series * encoder_padding_mask
            time_series = torch.sum(time_series, dim=-1) + 6e-8
            tmp_encoder_outs_rev = tmp_encoder_outs_rev.masked_fill(batch_tgt_mask.unsqueeze(-1), 0)
            tmp_encoder_outs_rev = (tmp_encoder_outs_rev / time_series.unsqueeze(-1)).detach()
            last_loss = self.cal_normal_loss(future_capsule.masked_fill(batch_tgt_mask.unsqueeze(-1),0) - self.de_future(tmp_decoder_outs_rev) + self.en_future(tmp_encoder_outs_rev))
            return first_loss, last_loss / batch_size.detach()

    def cal_normal_loss(self, input):
        squared_norm = (input.float() ** 2).sum(dim=-1)
        squared_norm = squared_norm.sum(dim=-1)
        return squared_norm
    def extract_features(self, prev_output_tokens, encoder_out=None, incremental_state=None, new_times=5, reorder=None, step=None,isadd=True, **unused):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]
        attn_weights = []
        calNumber = 0
        # decoder layers
        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
                new_times=new_times,
                reorder=reorder,
                step=step,
                isadd=isadd
            )
            inner_states.append(x)

            calNumber += 1
            if attn is not None and calNumber==6:
                attn_weights = attn

        if attn is None:
            attn_weights = None
            
        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {'attn': attn_weights, 'inner_states': inner_states}

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                return F.linear(features, self.embed_tokens.weight)
            else:
                return F.linear(features, self.embed_out)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())
        
    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device or self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)

        for i in range(len(self.layers)):
            # update layer norms
            layer_norm_map = {
                '0': 'self_attn_layer_norm',
                '1': 'encoder_attn_layer_norm',
                '2': 'final_layer_norm'
            }
            for old, new in layer_norm_map.items():
                for m in ('weight', 'bias'):
                    k = '{}.layers.{}.layer_norms.{}.{}'.format(name, i, old, m)
                    if k in state_dict:
                        state_dict['{}.layers.{}.{}.{}'.format(name, i, new, m)] = state_dict[k]
                        del state_dict[k]

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True, dtype=None):
    if dtype is not None:
        m = nn.Linear(in_features, out_features, bias, dtype=dtype)
    else:
        m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


@register_model_architecture('transformer', 'transformer')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.adaptive_input = getattr(args, 'adaptive_input', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)


@register_model_architecture('transformer', 'transformer_iwslt_de_en')
def transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    base_architecture(args)


@register_model_architecture('transformer', 'transformer_wmt_en_de')
def transformer_wmt_en_de(args):
    base_architecture(args)


# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture('transformer', 'transformer_vaswani_wmt_en_de_big')
def transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.dropout = getattr(args, 'dropout', 0.3)
    base_architecture(args)


@register_model_architecture('transformer', 'transformer_vaswani_wmt_en_fr_big')
def transformer_vaswani_wmt_en_fr_big(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    transformer_vaswani_wmt_en_de_big(args)


@register_model_architecture('transformer', 'transformer_wmt_en_de_big')
def transformer_wmt_en_de_big(args):
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    transformer_vaswani_wmt_en_de_big(args)


# default parameters used in tensor2tensor implementation
@register_model_architecture('transformer', 'transformer_wmt_en_de_big_t2t')
def transformer_wmt_en_de_big_t2t(args):
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', True)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.1)
    transformer_vaswani_wmt_en_de_big(args)