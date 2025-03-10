import torch
import torch.distributed as dist

def get_model_param_stats(model, ref_model):
    model_params = torch.cat([p.view(-1) for p in model.parameters() if p.requires_grad])
    ref_model_params = torch.cat([p.view(-1) for p in ref_model.parameters()])
    assert model_params.shape == ref_model_params.shape, f"{model_params.shape=} {ref_model_params.shape=}"
    return {
        "params_with_grads_mean": model_params.mean().item(),
        "params_with_grads_std": model_params.std().item(),
        "distance_to_ref": torch.nn.functional.mse_loss(model_params, ref_model_params),
    }

def careful_repeat(tensor, num_repeats):
    assert isinstance(tensor, torch.Tensor)
    batch_size = tensor.shape[0]
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(1).repeat(1, num_repeats).reshape(batch_size*num_repeats, *tensor.shape[1:])
    elif tensor.ndim == 2:
        tensor = tensor.unsqueeze(1).repeat(1, num_repeats, 1).reshape(batch_size*num_repeats, *tensor.shape[1:])
    else:
        raise ValueError(f"Invalid ndim: {tensor.ndim}")
    return tensor

def careful_repeat_dict(data, num_repeats):
    assert isinstance(data, dict)
    for k, v in data.items():
        data[k] = careful_repeat(v, num_repeats)
    return data

def append_eos_token(inputs, eos_token_id, pad_token_id):
    B, S = inputs["input_ids"].shape
    device = inputs["input_ids"].device
    dtype = inputs["input_ids"].dtype
    # Append padding tokens
    inputs["input_ids"] = torch.cat([inputs["input_ids"], torch.full((B, 1), pad_token_id, dtype=dtype, device=device)], dim=1)
    # Find the lengths of each sequence (sum of attention mask along dim=1)
    lengths = inputs["attention_mask"].sum(dim=1)
    # Use torch.arange to construct index positions for each sequence
    batch_indices = torch.arange(B, device=device)
    # Assign the eos_token_id at the correct position (lengths indicate where padding starts)
    inputs["input_ids"][batch_indices, lengths] = eos_token_id
    return inputs

def step(model, next_input, attention_mask, position_ids, past_key_values, as_full_distribution=False):
    batch_size = next_input.shape[0]
    next_seq_length = next_input.shape[1]
    prev_seq_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
    assert position_ids.shape == (batch_size, next_seq_length), f"{position_ids.shape=}, {next_seq_length=}"
    assert attention_mask.shape == (batch_size, prev_seq_length+next_seq_length), f"{attention_mask.shape=}, {prev_seq_length=}, {next_seq_length=}"
    if as_full_distribution:
        # next_input is a distribution over the vocabulary (batch_size, next_seq_length, vocab_size)
        vocab_size = model.module.config.vocab_size if dist.is_initialized() else model.config.vocab_size
        assert next_input.shape == (batch_size, next_seq_length, vocab_size)
        all_embeds = model.module.model.embed_tokens.weight if dist.is_initialized() else model.model.embed_tokens.weight
        hidden_dim = all_embeds.shape[1]
        inputs_embeds = torch.matmul(next_input, all_embeds)
        assert inputs_embeds.shape == (batch_size, next_seq_length, hidden_dim)
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values)
    else:
        assert next_input.shape == (batch_size, next_seq_length)
        outputs = model(input_ids=next_input, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values)
    logits = outputs.logits
    past_key_values = outputs.past_key_values
    return logits, past_key_values

def get_generations(
        model,
        ref_model,
        tokenizer,
        questions_inputs,
        answer_inputs,
        cot_length,
        temperature=1.0,
        as_full_distribution=False,
        teacher_forcing=0.5,
        generations_per_prompt=1,
        normalization_type="grpo",
        kl_type="normal",
        reward_type="answer_generated",
):
    metrics = {}
    model = model.module if dist.is_initialized() else model
    device = model.device
    batch_size = questions_inputs["input_ids"].shape[0]
    # add eot to answer
    answer_inputs = append_eos_token(answer_inputs, model.config.eos_token_id, model.config.pad_token_id)
    # answer prompts
    answer_prompts = torch.tensor(tokenizer.encode(">>Answer:"), dtype=torch.int, device=device).unsqueeze(0).repeat(batch_size, 1)
    # shapes
    vocab_size = model.config.vocab_size
    question_length = questions_inputs["input_ids"].shape[1]
    answer_length = answer_inputs["input_ids"].shape[1]
    answer_prompt_length = answer_prompts.shape[1]
    if not as_full_distribution:
        # repeat
        questions_inputs = careful_repeat_dict(questions_inputs, generations_per_prompt)
        answer_inputs = careful_repeat_dict(answer_inputs, generations_per_prompt)
        answer_prompts = careful_repeat(answer_prompts, generations_per_prompt)
        # generate cots
        with torch.no_grad():
            q_cot_ids = model.generate(
                    input_ids=questions_inputs["input_ids"],
                    attention_mask=questions_inputs["attention_mask"],
                    max_new_tokens=cot_length,
                    temperature=temperature,
                    top_k=None,
                    do_sample=True,
                    eos_token_id=None,
                )
        # add answer prompts
        q_cot_prompt_ids = torch.cat((q_cot_ids, answer_prompts), dim=1)
        q_cot_prompt_mask = torch.ones_like(q_cot_prompt_ids)
        q_cot_prompt_mask[:, :question_length] = questions_inputs["attention_mask"]
        if reward_type == "answer_generated":
            # generate answer
            with torch.no_grad():
                q_cot_prompt_ans_ids = model.generate(
                    input_ids=q_cot_prompt_ids,
                    attention_mask=q_cot_prompt_mask,
                    max_new_tokens=answer_length,
                    temperature=temperature,
                    top_k=None,
                    do_sample=True,
                    eos_token_id=None,
                )
            answer = q_cot_prompt_ans_ids[:, -answer_length:]
        elif reward_type == "answer_prob":
            # append answer
            answer = answer_inputs["input_ids"]
            q_cot_prompt_ans_ids = torch.cat((q_cot_prompt_ids, answer), dim=1)
        else:
            raise ValueError(f"{reward_type=}")
        # forward pass
        q_cot_prompt_ans_mask = torch.cat((q_cot_prompt_mask, torch.ones_like(answer_inputs["input_ids"])), dim=1)
        all_logits = model(
            input_ids=q_cot_prompt_ans_ids[:, :-1],
            attention_mask=q_cot_prompt_ans_mask[:, :-1],
            ).logits
        # ref forward pass
        with torch.no_grad():
            ref_all_logits = ref_model(
                input_ids=q_cot_prompt_ans_ids[:, :-1],
                attention_mask=q_cot_prompt_ans_mask[:, :-1],
                ).logits
        # collect logits and generations
        cot_generations = q_cot_ids[:, question_length:question_length+cot_length]
        cot_logits = all_logits[:, question_length-1:question_length+cot_length-1]
        ans_logits = all_logits[:, -answer_length:]
    
    ############################################################################################################

    else: # as_full_distribution
        assert generations_per_prompt == 1, f"{generations_per_prompt=}"
        assert normalization_type == "none", f"{normalization_type=}"
        # make attention mask and position ids
        attention_mask = torch.ones((batch_size, question_length+cot_length+answer_prompt_length+answer_length), device=device)
        attention_mask[:, :question_length] = questions_inputs["attention_mask"]
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1)
        # question forward pass
        outputs = model(
            input_ids=questions_inputs["input_ids"][:, :-1],
            attention_mask=attention_mask[:, :question_length-1],
            position_ids=position_ids[:, :question_length-1],
        )
        past_key_values = outputs.past_key_values
        all_logits = outputs.logits
        next_input = questions_inputs["input_ids"][:, -1:]
        # full dist cot generation
        for t in range(cot_length+1):
            logits, past_key_values = step(model,
                next_input=next_input,
                attention_mask=attention_mask[:, :question_length+t],
                position_ids=position_ids[:, question_length+t-1:question_length+t],
                past_key_values=past_key_values,
                as_full_distribution=False if t == 0 else True,
                )
        next_input = torch.nn.functional.softmax(logits / temperature, dim=-1)
        all_logits = torch.cat((all_logits, logits), dim=1)
        # answer prompt forward pass
        logits, past_key_values = step(model,
            next_input=answer_prompts,
            attention_mask=attention_mask[:, :question_length+cot_length+answer_prompt_length],
            position_ids=position_ids[:, :question_length+cot_length+answer_prompt_length],
            past_key_values=past_key_values,
            as_full_distribution=False,
            )
        all_logits = torch.cat((all_logits, logits), dim=1)
        # supervised answer generation
        prev_seq_length = question_length+cot_length+answer_prompt_length
        logits = logits[:, -1:]
        correct_answer_one_hots = torch.nn.functional.one_hot(answer_inputs["input_ids"], num_classes=vocab_size).float()
        for i in range(answer_length-1):
            next_input = torch.nn.functional.softmax(logits / temperature, dim=-1)
            next_input = teacher_forcing * correct_answer_one_hots[:, i:i+1] + (1.0 - teacher_forcing) * next_input
            logits, past_key_values = step(model,
                next_input=next_input,
                attention_mask=attention_mask[:, :prev_seq_length+i+1],
                position_ids=position_ids[:, prev_seq_length+i:prev_seq_length+i+1],
                past_key_values=past_key_values,
                as_full_distribution=True,
                )
            all_logits = torch.cat((all_logits, logits), dim=1)
        # collect logits
        cot_logits = all_logits[:, question_length-1:question_length+cot_length-1]
        cot_dist = torch.distributions.Categorical(logits=cot_logits / temperature)
        cot_generations = cot_dist.sample()
        ans_logits = all_logits[:, -answer_length:]
        # set answer
        if reward_type == "answer_generated":
            # answer = sampled from answer_logits
            ans_dist = torch.distributions.Categorical(logits=ans_logits / temperature)
            answer = ans_dist.sample()
        elif reward_type == "answer_prob":
            # answer = actual answer
            answer = answer_inputs["input_ids"]
        else:
            raise ValueError(f"{reward_type=}")
        # ref forward pass
        with torch.no_grad():
            input_ids = torch.cat((
                torch.nn.functional.one_hot(questions_inputs["input_ids"], num_classes=vocab_size).float(),
                torch.nn.functional.softmax(cot_logits[:, :-1] / temperature, dim=-1),
                answer_prompts,
                (
                    teacher_forcing * correct_answer_one_hots[:, :-1]
                    + (1.0 - teacher_forcing) * torch.nn.functional.softmax(ans_logits[:, :-1] / temperature, dim=-1)
                ),
            ), dim=1)
            ref_all_logits = step(ref_model,
                next_input=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                as_full_distribution=True,
                ).logits
            ref_cot_logits = ref_all_logits[:, question_length-1:question_length+cot_length-1]
            ref_ans_logits = ref_all_logits[:, -answer_length:]

    ############################################################################################################

    # calculate logps
    all_cot_logps = torch.nn.functional.log_softmax(cot_logits, dim=-1)
    all_ans_logps = torch.nn.functional.log_softmax(ans_logits, dim=-1)
    per_token_cot_logps = torch.gather(all_cot_logps, 2, cot_generations.unsqueeze(-1)).squeeze(-1)
    per_token_ans_logps = torch.gather(all_ans_logps, 2, answer.unsqueeze(-1)).squeeze(-1)
    cot_logp = per_token_cot_logps.sum(dim=-1)
    ans_logp = (per_token_ans_logps * answer_inputs["attention_mask"]).sum(dim=-1)
    # calculate reward
    with torch.no_grad():
        if reward_type == "answer_generated":
            generated_answer = torch.where(answer_inputs["attention_mask"], answer, torch.tensor(-1, dtype=torch.int, device=device))
            correct_answer = torch.where(answer_inputs["attention_mask"], answer_inputs["input_ids"], torch.tensor(-1, dtype=torch.int, device=device))
            reward = (generated_answer == correct_answer).all(dim=-1).float()
        elif reward_type == "answer_prob":
            ans_logp_length_normalized = ans_logp / answer_inputs["attention_mask"].sum(dim=-1)
            reward = ans_logp_length_normalized.exp()
        else:
            raise ValueError(f"{reward_type=}")
        # normalize reward
        if generations_per_prompt > 1:
            reward = reward.reshape(batch_size, generations_per_prompt)
            if normalization_type == "grpo":
                normalized_reward = (reward - reward.mean(1, keepdim=True)) / (reward.std(1, keepdim=True) + 1e-6)
            elif normalization_type == "rloo":
                group_sum = reward.sum(1, keepdim=True)
                normalized_reward = (group_sum - reward) / (generations_per_prompt - 1)
            elif normalization_type == "none":
                normalized_reward = reward
            else:
                raise ValueError(f"{normalization_type=}")
            reward = reward.reshape(-1)
            normalized_reward = normalized_reward.reshape(-1)
        else:
            assert normalization_type == "none", f"{normalization_type=}"
            normalized_reward = reward

    # kl
    if kl_type == "normal":
        ref_all_cot_logps = torch.nn.functional.log_softmax(ref_cot_logits, dim=-1)
        ref_all_ans_logps = torch.nn.functional.log_softmax(ref_ans_logits, dim=-1)
        ref_per_token_cot_logps = torch.gather(ref_all_cot_logps, 2, cot_generations.unsqueeze(-1)).squeeze(-1)
        ref_per_token_ans_logps = torch.gather(ref_all_ans_logps, 2, answer.unsqueeze(-1)).squeeze(-1)
        cot_kl = (torch.exp(ref_per_token_cot_logps - per_token_cot_logps) - (ref_per_token_cot_logps - per_token_cot_logps) - 1)
        ans_kl = (torch.exp(ref_per_token_ans_logps - per_token_ans_logps) - (ref_per_token_ans_logps - per_token_ans_logps) - 1)
        ans_kl = ans_kl * answer_inputs["attention_mask"]
        kl = cot_kl.sum(dim=-1) + ans_kl.sum(dim=-1)
    elif kl_type == "full":
        all_logps = torch.nn.functional.log_softmax(all_logits, dim=-1)
        ref_all_logps = torch.nn.functional.log_softmax(ref_all_logits, dim=-1)
        kl = (all_logps.exp() * (all_logps - ref_all_logps)).sum(-1).mean(-1)
    else:
        raise ValueError(f"{kl_type=}")
    
    # entropy
    pd = torch.nn.functional.softmax(all_logits, dim=-1)
    entropy = torch.logsumexp(all_logits, dim=-1) - torch.sum(pd * all_logits, dim=-1)

    # metrics

    # return
    assert cot_logp.requires_grad == True, f"{cot_logp.requires_grad=}"
    assert ans_logp.requires_grad == True, f"{ans_logp.requires_grad=}"
    assert normalized_reward.requires_grad == False, f"{normalized_reward.requires_grad=}"
    assert kl.requires_grad == True, f"{kl.requires_grad=}"
    assert entropy.requires_grad == True, f"{entropy.requires_grad=}"
    x = (cot_logp, ans_logp, normalized_reward, kl, entropy)
    return x, cot_generations, reward, metrics











    

    



    