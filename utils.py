import tempfile

import torch.nn as nn
from espnet2.tasks.s2t import S2TTask
from espnet2.text.token_id_converter import TokenIDConverter

import sentencepiece as spm
import sentencepiece.sentencepiece_model_pb2 as spm_protobuf


def beam_search_add_new_tokens(beam_search, model, tokens):
    beam_search.token_list += tokens
    beam_search.n_vocab += len(tokens)
    beam_search.scorers["decoder"] = model.decoder
    beam_search.scorers["scorefilter"].vocab_size += len(tokens)


def converter_tokenizer_add_new_tokens(s2t_train_args, tokens):
    preprocessor = S2TTask.build_preprocess_fn(s2t_train_args, train=False)
    preprocessor_add_new_tokens(preprocessor, tokens)
    return preprocessor.token_id_converter, preprocessor.tokenizer


def preprocessor_add_new_tokens(preprocessor, tokens):
    # Update tokenizer
    mp = spm_protobuf.ModelProto()
    mp.ParseFromString(open(preprocessor.tokenizer.model, "rb").read())
    for token in tokens:
        mp.pieces.append(mp.SentencePiece(piece=token, score=0.0, type=4))

    with tempfile.NamedTemporaryFile() as f:
        f.write(mp.SerializeToString())
        preprocessor.tokenizer.sp = spm.SentencePieceProcessor(model_file=f.name)

    # Update token list
    token_list = preprocessor.token_id_converter.token_list + tokens
    preprocessor.token_id_converter = TokenIDConverter(token_list)


def model_add_new_tokens(model, new_tokens, initialize=None):
    model.decoder.embed[0] = _extend_embedding(model.decoder.embed[0], len(new_tokens), initialize)
    model.decoder.output_layer = _extend_linear(model.decoder.output_layer, len(new_tokens), initialize)
    model.ctc.ctc_lo = _extend_linear(model.ctc.ctc_lo, len(new_tokens), initialize)

    model.vocab_size += len(new_tokens)
    model.criterion_att.size += len(new_tokens)
    model.token_list += new_tokens
    model.error_calculator.char_list += new_tokens


def _extend_linear(layer, extend_size, initialize=None):
    new_layer = nn.Linear(
        in_features=layer.in_features,
        out_features=layer.out_features + extend_size,
        bias=layer.bias is not None,
    )

    if layer.bias is not None:
        new_layer.bias.data[:layer.out_features] = layer.bias.data
        if initialize is not None:
            new_layer.bias.data[layer.out_features:] = layer.bias.data[initialize]

    new_layer.weight.data[:layer.out_features] = layer.weight.data
    if initialize is not None:
        new_layer.weight.data[layer.out_features:] = layer.weight.data[initialize]

    return new_layer


def _extend_embedding(embedding, extend_size, initialize=None):
    new_embedding = nn.Embedding(
        embedding.num_embeddings + extend_size,
        embedding.embedding_dim,
    )

    new_embedding.weight.data[:embedding.num_embeddings, :] = embedding.weight.data
    if initialize is not None:
        new_embedding.weight.data[embedding.num_embeddings:, :] = embedding.weight.data[initialize]

    return new_embedding
