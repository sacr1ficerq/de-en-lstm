import torch
import torch.nn as nn
import torch.nn.functional as F

from time import sleep

import numpy as np

unk_idx, pad_idx, bos_idx, eos_idx = 0, 1, 2, 3

from dataset2 import bucket_iterator

class LSTM_3(nn.Module):
    def __init__(self, 
                 src_vocab_size=None, 
                 trg_vocab_size=None, 
                 embedding_dim=None, 
                 hidden_size=None, 
                 num_layers=None,
                 enc_dropout=None,
                 dec_dropout=None, 
                 emb_dropout=None,
                 attention_dropout=None, 
                 config=None,
                 pad_idx=pad_idx,
                 weights_filename=None):
        super().__init__()

        self.train_loss = []
        self.val_loss = []

        if config:
            if not src_vocab_size:
                src_vocab_size = config['src_vocab_size']
            if not trg_vocab_size:
                trg_vocab_size = config['trg_vocab_size']
            if not embedding_dim:
                embedding_dim = config['embedding_dim']
            if not hidden_size:
                hidden_size = config['hidden_size']
            if not num_layers:
                num_layers = config.get('num_layers', 2)
            if not enc_dropout:
                enc_dropout = config.get('dropout_enc', 0.1)
            if not dec_dropout:
                dec_dropout = config.get('dropout_dec', 0.1)
            if not emb_dropout:
                emb_dropout = config.get('dropout_emb', 0.1)
            if not attention_dropout:
                attention_dropout = config.get('dropout_attention', 0.1)
            if not weights_filename:
                weights_filename = config.get('weights', None)

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.src_vocab_size = src_vocab_size

        self.trg_vocab_size = trg_vocab_size

        self.src_embedding = nn.Embedding(src_vocab_size, embedding_dim, padding_idx=pad_idx)
        self.trg_embedding = nn.Embedding(trg_vocab_size, embedding_dim, padding_idx=pad_idx)

        self.emb_dropout = nn.Dropout(emb_dropout, inplace=False)

        self.enc_dropout = nn.Dropout(enc_dropout, inplace=False)
        self.dec_dropout = nn.Dropout(dec_dropout, inplace=False)

        self.attention_dropout = nn.Dropout(attention_dropout, inplace=False)

        self.encoder = nn.LSTM(embedding_dim, 
                               hidden_size, 
                               batch_first=True, 
                               bidirectional=True, 
                               num_layers=num_layers, 
                               dropout=enc_dropout)
        encoder_output_size = hidden_size * 2
        self.encoder_output_proj = nn.Linear(encoder_output_size, hidden_size)

        self.decoder = nn.LSTM(embedding_dim, 
                               hidden_size, 
                               batch_first=True,
                               num_layers=num_layers, 
                               dropout=dec_dropout)
        
        self.encoder_hidden_proj = nn.ModuleList([
            nn.Linear(hidden_size * 2, hidden_size) for _ in range(num_layers)
        ])
        self.encoder_cell_proj = nn.ModuleList([
            nn.Linear(hidden_size * 2, hidden_size) for _ in range(num_layers)
        ])

        self.fc = nn.Linear(hidden_size * 2, trg_vocab_size)

        self.emb_layer_norm = nn.LayerNorm(embedding_dim)
        self.encoder_output_norm = nn.LayerNorm(hidden_size)
        self.decoder_output_norm = nn.LayerNorm(hidden_size)

        self.proj_layer_norm = nn.LayerNorm(hidden_size)      # not added to training/inference yet
        # self.attention_norm = nn.LayerNorm(hidden_size)       # not added to training/inference yet

        # xavier
        for name, param in self.named_parameters():
            if "weight" in name and "embedding" not in name:
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)
        
        if weights_filename != None:
            self.load(weights_filename)

    def _project_hidden(self, state, proj_layers):
        batch_size = state.size(1)
        state = state.view(self.num_layers, 2, batch_size, self.hidden_size)
        state = torch.cat([state[:, 0], state[:, 1]], dim=2)  # (num_layers, batch, 2*hidden_size)
        return torch.stack([self.proj_layer_norm(proj_layers[i](state[i])) for i in range(self.num_layers)], dim=0)

    def _encode(self, src_embedded):
        encoder_outputs, (hidden, cell) = self.encoder(src_embedded)  # hidden/cell: (1, batch_size, hidden_size)
        encoder_outputs = self.encoder_output_proj(encoder_outputs)
        encoder_outputs = self.encoder_output_norm(encoder_outputs)

        hidden = self._project_hidden(hidden, self.encoder_hidden_proj)
        cell = self._project_hidden(cell, self.encoder_cell_proj)

        return encoder_outputs, (hidden, cell)

    def _decode(self, trg_embedded, hidden, cell):
        decoder_outputs, (hidden, cell) = self.decoder(trg_embedded, (hidden, cell))  # (batch_size, trg_len, hidden_size)
        decoder_outputs = self.decoder_output_norm(decoder_outputs)
        return decoder_outputs, (hidden, cell)

    def forward_no_tf(self, src_seq, trg_seq):
        src_embedded = self.src_embedding(src_seq) # (batch_size, src_len, embedding_dim)
        src_embedded = self.emb_layer_norm(src_embedded)
        src_embedded = self.emb_dropout(src_embedded)

        encoder_outputs, (hidden, cell) = self._encode(src_embedded)
        encoder_outputs = self.enc_dropout(encoder_outputs)

        trg_embedded = self.trg_embedding(trg_seq) # (batch_size, trg_len, embedding_dim)
        trg_embedded = self.emb_layer_norm(trg_embedded)
        trg_embedded = self.emb_dropout(trg_embedded)

        decoder_outputs, _ = self._decode(trg_embedded, hidden, cell)  # (batch_size, trg_len, hidden_size)
        decoder_outputs = self.dec_dropout(decoder_outputs)

        # attention
        energy = torch.bmm(decoder_outputs, encoder_outputs.transpose(1, 2))

        mask = (src_seq != pad_idx).unsqueeze(1)  # (batch_size, 1, src_len)
        energy = energy.masked_fill(mask == 0, -1e10)

        attention = F.softmax(energy, dim=-1)
        attention = self.attention_dropout(attention)

        context = torch.bmm(attention, encoder_outputs)
        combined = torch.cat([decoder_outputs, context], dim=2)

        logits = self.fc(combined)
        return logits

    def inference(self, src_seq, max_len=None, device='cuda'):
        if max_len == None:
            max_len = int(src_seq.size(1) * 1.5 + 1)
        self.eval()

        batch_size = src_seq.size(0)
        trg_seq = torch.tensor([[bos_idx]] * batch_size, dtype=torch.long).to(device)  # (batch_size, 1)
        mask = (src_seq != pad_idx).unsqueeze(1)  # (batch_size, 1, src_len)

        with torch.no_grad():
            # encoder forward
            src_embedded = self.src_embedding(src_seq)
            src_embedded = self.emb_layer_norm(src_embedded)

            encoder_outputs, (hidden, cell) = self.encoder(src_embedded) # (B, max_len, H)
            # print('Encoder norm:', torch.norm(encoder_outputs[0, :20, :], 2))

            encoder_outputs = self.encoder_output_proj(encoder_outputs)
            encoder_outputs = self.encoder_output_norm(encoder_outputs)

            # encoder_outputs = encoder_outputs.contiguous()  # ensure contiguous
            encoder_outputs_t = encoder_outputs.transpose(1, 2) # (B, max_len, 2 * H) 
            
            # print(cell.size(), hidden.size())
            # project encoder hidden states for decoder
            hidden = self._project_hidden(hidden, self.encoder_hidden_proj) # (layers, B, H)
            cell = self._project_hidden(cell, self.encoder_cell_proj)
            # print(cell.size(), hidden.size())
            # print('hidden norm:', torch.norm(hidden[:, 0, :], 2))

            for i in range(max_len):
                # get last token (batch_size, 1)
                current_trg = trg_seq[:, -1].unsqueeze(1)
                # print('trg: ', *current_trg[0])
                trg_embedded = self.trg_embedding(current_trg)  # (batch_size, 1, emb_dim)
                trg_embedded = self.emb_layer_norm(trg_embedded)

                decoder_output, (hidden, cell) = self.decoder(trg_embedded, (hidden, cell))
                decoder_output = self.decoder_output_norm(decoder_output)
                
                # print('Decoder norm:', torch.norm(decoder_output[0, :, :], 2))

                # energy = torch.bmm(decoder_output, encoder_outputs.transpose(1, 2).contiguous())
                energy = torch.bmm(decoder_output, encoder_outputs_t)
                # print('Energy norm:', torch.norm(energy[0, :, :], 2))
                energy = energy.masked_fill(mask == 0, -1e10)
                attention = F.softmax(energy, dim=-1)

                context = torch.bmm(attention, encoder_outputs)
                combined = torch.cat([decoder_output, context], dim=2)
                
                logits = self.fc(combined)  # (batch_size, 1, trg_vocab_size)
                logits[:, :, unk_idx] = -1e10
                
                # print(*torch.topk(logits, 3, dim=-1).values[0], *torch.topk(logits, 3, dim=-1).indices[0])
                next_token = logits.argmax(dim=-1) # greedy
                # print(next_token[0])
                
                
                trg_seq = torch.cat([trg_seq, next_token], dim=1)

                if (next_token == eos_idx).all():
                    break

        return trg_seq

    def forward(self, src_seq, trg_seq, device='cuda', teacher_forcing=1.0):
        if teacher_forcing == 1.0: return self.forward_no_tf(src_seq, trg_seq)
        src_embedded = self.emb_dropout(self.src_embedding(src_seq)) # (batch_size, src_len, embedding_dim)
        src_embedded = self.emb_layer_norm(src_embedded)
        src_embedded = self.emb_dropout(src_embedded)

        encoder_outputs, (hidden, cell) = self.encoder(src_embedded)  # hidden/cell: (1, batch_size, hidden_size)
        encoder_outputs = self.encoder_output_proj(encoder_outputs)
        encoder_outputs = self.enc_dropout(encoder_outputs)

        encoder_outputs_t = encoder_outputs.transpose(1, 2)
        mask = (src_seq != pad_idx).unsqueeze(1)  # (batch_size, 1, src_len)

        hidden = self._project_hidden(hidden, self.encoder_hidden_proj)
        cell = self._project_hidden(cell, self.encoder_cell_proj)

        batch_size = src_seq.size(0)
        trg_len = trg_seq.size(1)
        trg_vocab_size = self.fc.out_features

        # tensor to store decoder outputs
        logits = torch.zeros(batch_size, trg_len, trg_vocab_size, device=device)
        decoder_input = trg_seq[:, 0].unsqueeze(1)  # [<sos>] (batch_size, 1)

        tf_mask = (torch.rand(batch_size, trg_len - 1, device=device) < teacher_forcing).long()

        # autoregressive decoding with teacher forcing
        for t in range(1, trg_len):  # skip <sos>
            trg_embedded = self.trg_embedding(decoder_input)  # (batch_size, 1, emb_dim)
            # trg_embedded = self.emb_dropout(self.trg_embedding(decoder_input))  # (batch_size, 1, emb_dim)
            
            decoder_output, (hidden, cell) = self.decoder(trg_embedded, (hidden, cell))

            # decoder_output = self.dec_dropout(decoder_output)

            energy = torch.bmm(decoder_output, encoder_outputs_t)  # (batch_size, 1, src_len)
            energy = energy.masked_fill(mask == 0, -1e10)
            
            attention = F.softmax(energy, dim=-1)
            # attention = self.attention_dropout(attention)
            context = torch.bmm(attention, encoder_outputs)  # (batch_size, 1, hidden_dim)

            combined = torch.cat([decoder_output, context], dim=2)  # (batch_size, 1, hidden_dim * 2)
            # step_logits = self.fc(combined)  # (batch_size, 1, trg_vocab_size)
            # logits[:, t] = step_logits.squeeze(1)
            logits[:, t:t+1] = self.fc(combined)  # (batch_size, trg_vocab_size)
            # print(logits.size(), logits[:, t].size())
                
            next_token_gt = trg_seq[:, t].unsqueeze(1)  # Ground truth (batch_size, 1)
            # next_token_pred = step_logits.argmax(-1)  # Predicted token (batch_size, 1)
            next_token_pred = logits[:, t].argmax(-1).unsqueeze(1)  # Predicted token (batch_size, 1)
            # print(next_token_pred.size())
            decoder_input = torch.where(
                tf_mask[:, t - 1].unsqueeze(1).bool(),
                next_token_gt,
                next_token_pred,
            )

        return logits  # (batch_size, trg_len, trg_vocab_size)

    def inference_beam(self, src_seq, max_len=None, beam_width=5, remove_unk=True, 
                       device='cuda', verbose=0, vocab_trg=None, vocab_src=None, border=0.0):
        unk_idx, pad_idx, bos_idx, eos_idx, num_idx = 0, 1, 2, 3, 4
        

        self.eval()
        batch_size = src_seq.size(0)
        k = beam_width


        src_lens = (src_seq != pad_idx).sum(dim=1).repeat(k).view(batch_size, k) * 1.2 + 2
        if max_len == None:
            max_len = int((src_seq != pad_idx).sum(dim=1).max(dim=-1)[0].item() * 1.2 + 2)
            # max_len = 72

        if verbose >= 1: print(f'k: {k}\tB: {batch_size}\tmax_len: {max_len}')

        # encoder forward
        with torch.no_grad():
            src_embedded = self.src_embedding(src_seq)  # (batch_size, seq_len, emb_dim)
            encoder_outputs, (hidden, cell) = self._encode(src_embedded)
            encoder_outputs = encoder_outputs.contiguous()

            # expand encoder output for beams
            encoder_outputs = encoder_outputs.unsqueeze(1).repeat(1, k, 1, 1)  # (batch_size, k, seq_len, hidden_dim)
            encoder_outputs = encoder_outputs.view(batch_size * k, -1, encoder_outputs.size(-1))  # (batch_size*k, seq_len, hidden_dim)

            hidden = hidden.unsqueeze(2).repeat(1, 1, k, 1)  # (num_layers, batch_size, k, hidden_dim)
            hidden = hidden.view(hidden.size(0), -1, hidden.size(-1))  # (num_layers, batch_size*k, hidden_dim)
            cell = cell.unsqueeze(2).repeat(1, 1, k, 1).view(hidden.size(0), -1, hidden.size(-1))

            # init beams
            beam_scores = torch.zeros((batch_size, k), dtype=torch.float, device=device)  # (batch_size, k)
            beam_scores[:, 1:] = -1e10  # force beam 0 to be top 1

            beam_tokens = torch.full((batch_size, k, 1), bos_idx, dtype=torch.long, device=device)  # (batch_size, k, 1)
            token_scores = torch.full((batch_size, k, 1), 0, dtype=torch.long, device=device)  # (batch_size, k, 1)

            finished = torch.zeros((batch_size, k), dtype=torch.bool, device=device)

            num_layers = hidden.size(0)
            vocab_size = self.trg_vocab_size
            hidden_dim = hidden.size(-1)

            for step in range(max_len):
                if verbose > 1: print(f'\nstep:{step}')
                flat_hidden = hidden  # (num_layers, batch_size*k, hidden_dim)
                flat_cell = cell
                flat_tokens = beam_tokens.view(batch_size * k, -1)  # (batch_size*k, seq_len)  
                flat_token_scores = token_scores.view(batch_size * k, -1)  # (batch_size*k, seq_len)  

                def pretty_print(row, width=5):
                    print("|".join(f"{str(item)[:6]:<{width}}" for item in row))

                def display_src(line):
                    pretty_print(map(vocab_src.decode_idx, line))      

                def display_trg(line):
                    pretty_print(list(map(vocab_trg.decode_idx, line)))

                def get_array(tnsr):
                    return np.round(tnsr.detach().cpu().numpy(), 1)

                def display_probs(tnsr):
                    log_probs = get_array(tnsr)
                    ps = np.round(np.exp(log_probs)*100, 2)
                    pretty_print(ps)

                def display_beams(beams, scores, log_probs, const=100):
                    for i in range(batch_size):
                        display_src(src_seq[i])
                        print(f'index in batch: {i}')
                        probs = np.exp(get_array(scores[i])) * const
                        pretty_print(probs)
                        print()
                        for j, line in enumerate(get_array(beams[i])):
                            print(f'\tbeam probability: {probs[j]*100:0.2f}')
                            display_trg(line)
                            display_probs(log_probs[i, j])
                        print()


                # decoder
                current_trg = flat_tokens[:, -1].unsqueeze(1)  # (batch_size*k, 1)
                trg_embedded = self.trg_embedding(current_trg)  # (batch_size*k, 1, emb_dim)
                # decoder_output, (new_hidden, new_cell) = self.decoder(trg_embedded, (flat_hidden, flat_cell))
                decoder_output, (new_hidden, new_cell) = self._decode(trg_embedded, flat_hidden, flat_cell)

                current_token_scores = flat_token_scores[:, -1].unsqueeze(1) # (batch_size*k, 1)

                mask = (src_seq != pad_idx).unsqueeze(1)  # (batch_size, 1, src_len)
                mask = mask.repeat(k, 1, 1)
                mask.size()

                # attention + logits
                energy = torch.bmm(decoder_output, encoder_outputs.transpose(1, 2))  # (batch_size*k, 1, seq_len)
                energy = energy.masked_fill(mask == 0, -1e10)

                attention = F.softmax(energy, dim=-1)
                context = torch.bmm(attention, encoder_outputs)  # (batch_size*k, 1, hidden_dim)
                combined = torch.cat([decoder_output, context], dim=2)
                logits = self.fc(combined).squeeze(1)  # (batch_size*k, vocab_size)
                if remove_unk: logits[:, unk_idx] = -1e10  # Block <UNK>

                # scores (log probs)
                log_probs = F.log_softmax(logits, dim=-1)  # (batch_size*k, vocab_size)

                # print(finished.view(-1))
                log_probs[finished.view(-1)] = - 1e+10
                
                log_probs[finished.view(-1), pad_idx] = 0

                for i in range(batch_size * k):
                    # print(flat_tokens[i].cpu().numpy())
                    log_probs[i][flat_tokens[i]] -= np.log(2.5)
                    
                    
                l = src_lens.view(-1) - step + 2

                # print(log_probs[:, eos_idx].size())

                l = l.masked_fill((l > 0), 0)
                # print(l)
                log_probs[:, eos_idx] += l * np.log(1.2)

                next_scores = log_probs + beam_scores.view(-1, 1)  # (batch_size*k, vocab_size)

                # reshape to (batch_size, k * vocab_size)
                next_scores = next_scores.view(batch_size, k * vocab_size)

                # topk candidates for each batch 
                next_scores, next_tokens = torch.topk(next_scores, k, dim=1)  # log_probs: (batch_size, k), indices: (batch_size, k)

                beam_indices = next_tokens // vocab_size  # what beam is token from
                token_indices = next_tokens % vocab_size  # what token

                if verbose > 1:
                    print('beam tokens:')
                    display_beams(beam_tokens, beam_scores, token_scores)

                # new scores
                beam_scores = next_scores

                # new token scores
                token_scores = torch.cat([
                    token_scores[torch.arange(batch_size).unsqueeze(1), beam_indices],  # (batch_size, k) -> (batch_size, k, seq_len)
                    next_scores.unsqueeze(-1) - current_token_scores.view(batch_size, k, -1)
                ], dim=-1)

                # new beam tokens
                beam_tokens = torch.cat([
                    beam_tokens[torch.arange(batch_size).unsqueeze(1), beam_indices],  # (batch_size, k) -> (batch_size, k, seq_len)
                    token_indices.unsqueeze(-1)
                ], dim=-1)

                # new hidden states
                hidden = new_hidden.view(num_layers, batch_size, k, -1)  # (num_layers, batch_size, k, hidden_dim)
                hidden = hidden.view(num_layers, -1, hidden_dim)  # (num_layers, batch_size*k, hidden_dim)

                cell = new_cell.view(num_layers, batch_size, k, -1)  # (num_layers, batch_size, k, hidden_dim)
                cell = cell.view(num_layers, -1, hidden_dim)

                # early stopping
                new_finished = (token_indices == eos_idx)
                finished = finished.gather(1, beam_indices) | new_finished
                if finished.all():
                    break

            beam_lengths = (beam_tokens[torch.arange(batch_size)] != pad_idx).sum(dim=-1)
            penalty = (src_lens - beam_lengths) / beam_lengths
            # print(penalty)

            # Select best beam with highest normalized score
            # print(beam_lengths)
            final_scores = beam_scores - penalty
            best_indices = final_scores.argmax(dim=1)
            trg_seq = beam_tokens[torch.arange(batch_size), best_indices]
            trg_seq_scores = token_scores[torch.arange(batch_size), best_indices]

            if verbose > 0:
                print('beam tokens:')
                display_beams(beam_tokens, beam_scores, token_scores)

            if verbose>0:
                for i in range(batch_size):
                    print('result:')
                    display_trg(trg_seq[i])
                    # print('token scores:')
                    display_probs(trg_seq_scores[i])


            if border > 0.0:
                trg_seq = trg_seq.masked_fill(trg_seq_scores < np.log(border), pad_idx) # 1d

            return trg_seq

    def translate(self, src, vocab_src, vocab_trg, beams=0, verbose=0, border=0, device='cuda'):
        src = src.split()
        encoded = vocab_src.encode(src)
        encoded = torch.tensor(encoded, dtype=torch.long, device=device)
        batched = encoded.unsqueeze(0)
        if beams == 0:
            prediction = self.inference(batched, device=device).squeeze(0)
        else:
            prediction = self.inference_beam(batched, beam_width=beams, verbose=verbose, vocab_src=vocab_src, vocab_trg=vocab_trg, border=border, device=device).squeeze(0)
        # print('prediction:', prediction)
        trg = vocab_trg.decode(prediction, src=src, vocab_src=vocab_src)
        return " ".join(trg)      

    def demonstrate(self, dataset, vocab_src, vocab_trg, examples=10, device='cuda', wait=3, verbose=0, beam_width=5):
        from submission import bleu  
        n = 0
        for i, (src, trg) in enumerate(bucket_iterator(dataset, batch_size=1)):
            if i >= examples:
                break
            src = src[0]
            trg = trg[0]
            s = " ".join(vocab_src.decode(src))
            r = " ".join(vocab_trg.decode(trg))
            c1 = self.translate(s, 
                                device=device,
                                vocab_src=vocab_src, 
                                vocab_trg=vocab_trg)
            c2 = self.translate(s, 
                                device=device, 
                                vocab_src=vocab_src, 
                                vocab_trg=vocab_trg, 
                                verbose=verbose, 
                                beams=beam_width)
                # print(list(src[i]).index(eos_idx), list(trg[i]).index(eos_idx), list(predictions[i]).index(eos_idx), list(predictions_beam[i]).index(eos_idx))

            b, pres, bp = bleu(ref=r.split(), c=c1.split(), verbose=0)
            b_beam, pres_beam, bp_beam = bleu(ref=r.split(), c=c2.split(), verbose=0)
            if b > 20:
                pass
                # continue
            print("src:\t\t", s)
            print("trg:\t\t", r)
            print("pred:\t\t", c1)
            print("pred-beam:\t", c2)

            print(f'bleu:\t\t{b:0.2f}')
            print(f'bleu beam:\t{b_beam:0.2f}')
            print()
            torch.cuda.ipc_collect()
            sleep(wait)

    def save(self, filename, folder):
        np.save(folder + 'train.npy', self.train_loss)
        np.save(folder + 'val.npy', self.val_loss)
        torch.save(self.state_dict(), folder + filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, weights_only=True))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)