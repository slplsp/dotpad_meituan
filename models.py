import torch
from torch import nn
import torchvision
import random
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """
    Encoder
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        # Pretrained ImageNet ResNet-101
        # Remove linear and pool layers
        resnet = torchvision.models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune(fine_tune=True)

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: boolean
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class DecoderWithRNN(nn.Module):
    def __init__(self, cfg, encoder_dim=14 * 14 * 2048):
        """
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithRNN, self).__init__()

        self.encoder_dim = encoder_dim
        self.decoder_dim = cfg['decoder_dim']
        self.embed_dim = cfg['embed_dim']
        self.vocab_size = cfg['vocab_size']
        self.dropout = cfg['dropout']
        self.device = cfg['device']

        ############################################################################
        # To Do: define some layers for decoder with RNN
        # self.embedding : Embedding layer
        # self.decode_step : decoding LSTMCell, using nn.LSTMCell
        # self.init : linear layer to find initial input of LSTMCell
        # self.bn : Batch Normalization for encoder's output
        # self.fc : linear layer to transform hidden state to scores over vocabulary
        # other layers you may need
        # Your Code Here!
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.decode_step = nn.LSTMCell(input_size=self.decoder_dim, hidden_size=self.decoder_dim)
        self.init = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.bn = nn.BatchNorm1d(self.decoder_dim)
        self.fc = nn.Linear(self.decoder_dim, self.vocab_size)
        ############################################################################

        # initialize some layers with the uniform distribution
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.
        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.
        [32,14,14,2048]  [32, 52]  [32, 1]
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_out = encoder_out.reshape(batch_size, -1)
        vocab_size = self.vocab_size

        # Sort input data by decreasing lengths;
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()
        # Create tensors to hold word predicion scores
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(self.device)

        # Initialize LSTM state
        init_input = self.bn(self.init(encoder_out))
        h, c = self.decode_step(init_input)  # (batch_size_t, decoder_dim)

        ############################################################################
        # To Do: Implement the main decode step for forward pass 
        # Hint: Decode words one by one
        # Teacher forcing is used.
        # At each time-step, generate a new word in the decoder with the previous word embedding
        # Your Code Here!
        # decode_lengths 已经是降序排序后的状态
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            h, c = self.decode_step(embeddings[:batch_size_t, t, :], (h[:batch_size_t], c[:batch_size_t]))
            preds = self.fc(self.dropout_layer(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds

        ############################################################################
        return predictions, encoded_captions, decode_lengths, sort_ind

    def one_step(self, embeddings, h, c):
        ############################################################################
        # To Do: Implement the one time decode step for forward pass 
        # this function can be used for test decode with beam search
        # return predicted scores over vocabs: preds
        # return hidden state and cell state: h, c
        # Your Code Here!

        h, c = self.decode_step(embeddings, (h, c))
        preds = self.fc(self.dropout_layer(h))
        ############################################################################
        return preds, h, c


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        #################################################################
        # To Do: you need to define some layers for attention module
        # Hint: Firstly, define linear layers to transform encoded tensor
        # and decoder's output tensor to attention dim; Secondly, define
        # attention linear layer to calculate values to be softmax-ed; 
        # Your Code Here!
        self.encode_hidden = nn.Linear(encoder_dim, attention_dim)
        self.decode_hidden = nn.Linear(decoder_dim, attention_dim)
        self.atten_layer = nn.Linear(attention_dim, 1)

        #################################################################

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward pass.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        #################################################################
        # To Do: Implement the forward pass for attention module
        # Hint: follow the equation 
        # "e = f_att(encoder_out, decoder_hidden)"
        # "alpha = softmax(e)"
        # "z = alpha * encoder_out"
        # Your Code Here!

        encoder_atten = self.encode_hidden(encoder_out)
        docoder_atten = self.decode_hidden(decoder_hidden)
        # 按照注意力权重计算公式
        e = self.atten_layer(F.relu(encoder_atten + docoder_atten.unsqueeze(1))).squeeze(2)
        alpha = F.softmax(e, dim=1)
        # 获得注意力输出
        z = (alpha.unsqueeze(2) * encoder_out).sum(dim=1)

        #################################################################
        return z, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, cfg, encoder_dim=2048):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.decoder_dim = cfg['decoder_dim']
        self.attention_dim = cfg['attention_dim']
        self.embed_dim = cfg['embed_dim']
        self.vocab_size = cfg['vocab_size']
        self.dropout = cfg['dropout']
        self.device = cfg['device']

        ############################################################################
        # To Do: define some layers for decoder with attention
        # self.attention : Attention layer
        # self.embedding : Embedding layer
        # self.decode_step : decoding LSTMCell, using nn.LSTMCell
        # self.init_h : linear layer to find initial hidden state of LSTMCell
        # self.init_c : linear layer to find initial cell state of LSTMCell
        # self.beta : linear layer to create a sigmoid-activated gate
        # self.fc : linear layer to transform hidden state to scores over vocabulary
        # other layers you may need
        # Your Code Here!

        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.attention = Attention(self.encoder_dim, self.decoder_dim, self.attention_dim)
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.decode_step = nn.LSTMCell(input_size=self.decoder_dim + self.encoder_dim, hidden_size=self.decoder_dim)
        self.init_h = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.init_c = nn.Linear(self.encoder_dim, self.decoder_dim)
        # self.beta = nn.Sequential(
        #     nn.Linear(self.embed_dim, self.encoder_dim),
        #     nn.Sigmoid())
        self.beta = nn.Linear(self.embed_dim, self.encoder_dim)
        self.fc = nn.Linear(self.decoder_dim, self.vocab_size)

        ############################################################################

        # initialize some layers with the uniform distribution
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.
        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths;
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(self.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(self.device)

        # Initialize LSTM state
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)

        ############################################################################
        # To Do: Implement the main decode step for forward pass 
        # Hint: Decode words one by one
        # Teacher forcing is used.
        # At each time-step, decode by attention-weighing the encoder's output based 
        # on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        # Your Code Here!
        for len_ in range(1, max(decode_lengths) + 1):
            stop_index = len(decode_lengths)
            for i_index, i_len in enumerate(decode_lengths):
                # 如果长度小于当前长度，记录截止的索引
                if i_len < len_:
                    stop_index = i_index
                    break
            # Teacher forcing
            if random.random() < 0.7:
                preds, alpha, h, c = self.one_step(embeddings[:stop_index, len_ - 1, :], encoder_out[:stop_index],
                                                   h[:stop_index], c[:stop_index])
            else:
                preds, alpha, h, c = self.one_step(h[:stop_index], encoder_out[:stop_index], h[:stop_index],
                                                   c[:stop_index])
            predictions[:stop_index, len_ - 1, :] = preds
            alphas[:stop_index, len_ - 1, :] = alpha

        ############################################################################
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind

    def one_step(self, embeddings, encoder_out, h, c):
        ############################################################################
        # To Do: Implement the one time decode step for forward pass
        # this function can be used for test decode with beam search
        # return predicted scores over vocabs: preds
        # return attention weight: alpha
        # return hidden state and cell state: h, c
        # Your Code Here!

        # z注意力输出，alpha注意力权重矩阵
        z, alpha = self.attention(encoder_out, h)
        beta = F.sigmoid(self.beta(h))
        z = beta * z
        # 拼接注意力输出和解码器隐藏状态
        h, c = self.decode_step(torch.cat([embeddings, z], dim=1), (h, c))
        preds = self.fc(self.dropout_layer(h))

        ############################################################################
        return preds, alpha, h, c


