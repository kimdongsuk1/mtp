def multi_token_forward_backward_motif(self,
                                 hidden_states: torch.FloatTensor,
                                 input_embeds: torch.FloatTensor,
                                 outputs: MotifModelOutputWithPast,
                                 labels: torch.LongTensor,
                                 position_ids: Optional[torch.LongTensor],
                                 output_attentions: Optional[bool],
                                 use_cache: Optional[bool],
                                 cache_position: Optional[torch.LongTensor],
                                 return_dict: Optional[bool]) -> CausalLMOutputWithPast:
    """
    Performs multi-token forward and backward pass for a motif-based causal language model.
    
    Main Contributions:
    1. The hidden normalization is applied after the transformer layer forward pass, differing from DeepSeek MoE Multi Token Prediction.
    2. Instead of concatenation and projection (DeepSeek MoE MTP), this approach employs tensor rolling and elementwise addition.
    
    Args:
        hidden_states (torch.FloatTensor): The input hidden states of shape [batch_size, seq_len, hidden_dim].
        input_embeds (torch.FloatTensor): Input token embeddings of shape [batch_size, seq_len, hidden_dim].
        outputs (MotifModelOutputWithPast): Model output containing past key values, causal mask, and position embeddings.
        labels (torch.LongTensor): Ground truth labels for computing loss.
        position_ids (Optional[torch.LongTensor]): Token position IDs.
        output_attentions (Optional[bool]): Whether to return attention outputs.
        use_cache (Optional[bool]): Whether to use cache for past key values.
        cache_position (Optional[torch.LongTensor]): Cached token positions.
        return_dict (Optional[bool]): Whether to return results as a dictionary.
    
    Returns:
        CausalLMOutputWithPast: Model output containing loss, logits, past key values, and optional hidden states/attentions.
    """

    past_key_values = outputs.past_key_values
    causal_mask = outputs.causal_mask
    position_embeddings = outputs.position_embeddings

    final_loss = None

    for token_idx in range(self.multi_token_heads):
        if token_idx > 0:
            layer = self.tokenwise_last_layers[token_idx - 1] ## your transformer layer
            hidden_norm = self.last_hidden_norms[token_idx - 1] ## your rmsnorm layer
            embed_norm = self.embed_norms[token_idx - 1] ## your rmsnorm layer

            input_embeds[..., 0, :] = 0
            input_embeds = torch.roll(input_embeds, shifts=1, dims=1)
            embed_normed = embed_norm(input_embeds)

            hidden_states += embed_normed ## hidden states is the final output of transformer decoder block layers

            layer_outputs = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            last_hidden_states = layer_outputs[0]

            if hasattr(self, "num_logits_to_keep") and self.num_logits_to_keep > 0:
                assert labels is None
                last_hidden_states = last_hidden_states[:, -self.num_logits_to_keep:, :]

            hidden_normed = hidden_norm(last_hidden_states)
            tokenwise_logits = self.lm_head(hidden_normed)

            if labels is None:
                return {"loss": None, "logits": tokenwise_logits}

            shift_n = token_idx + 1
            shift_logits = tokenwise_logits[..., :-shift_n, :].contiguous()
            shift_labels = labels[..., shift_n:].contiguous()

            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)

            ntp_loss = loss_fct(shift_logits, shift_labels)
            hidden_states = hidden_normed
          
        else:
            tokenwise_logits = self.lm_head(hidden_states)
            shift_n = token_idx + 1
            shift_logits = tokenwise_logits[..., :-shift_n, :].contiguous()
            shift_labels = labels[..., shift_n:].contiguous()

            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)

            ntp_loss = loss_fct(shift_logits, shift_labels)
        if final_loss is None:
            final_loss = ntp_loss
        else:
            final_loss += ntp_loss
       
    final_loss /= self.multi_token_heads

    return CausalLMOutputWithPast(
        loss=final_loss,
        logits=tokenwise_logits, 
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
