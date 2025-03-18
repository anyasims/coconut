import torch
import torch.distributed as dist
import gc

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
    inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.full((B, 1), 0, dtype=dtype, device=device)], dim=1)
    # Find the lengths of each sequence (sum of attention mask along dim=1)
    lengths = inputs["attention_mask"].sum(dim=1)
    # Use torch.arange to construct index positions for each sequence
    batch_indices = torch.arange(B, device=device)
    # Assign the eos_token_id at the correct position (lengths indicate where padding starts)
    inputs["input_ids"][batch_indices, lengths] = eos_token_id
    inputs["attention_mask"][batch_indices, lengths] = 1
    return inputs

def step(model, next_input, attention_mask, position_ids, past_key_values, as_full_distribution=False):
    batch_size = next_input.shape[0]
    next_seq_length = next_input.shape[1]
    prev_seq_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
    assert position_ids.shape == (batch_size, next_seq_length), f"{position_ids.shape=}, {next_input.shape=}"
    assert attention_mask.shape == (batch_size, prev_seq_length+next_seq_length), f"{attention_mask.shape=}, {prev_seq_length=}, {next_seq_length=}"
    if as_full_distribution:
        # next_input is a distribution over the vocabulary (batch_size, next_seq_length, vocab_size)
        vocab_size = model.config.vocab_size
        assert next_input.shape == (batch_size, next_seq_length, vocab_size)
        all_embeds = model.model.embed_tokens.weight
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

def decode(tokenizer, cot, ans_gen, ans_argmax, ans_correct):
    decoded_cot = tokenizer.batch_decode(cot, skip_special_tokens=False)
    decoded_ans_gen = tokenizer.batch_decode(ans_gen, skip_special_tokens=False)
    decoded_ans_argmax = tokenizer.batch_decode(ans_argmax, skip_special_tokens=False)
    decoded_ans_correct = tokenizer.batch_decode(ans_correct, skip_special_tokens=False)
    decoded_generations = [(a, b, c, d) for a, b, c, d in zip(decoded_cot, decoded_ans_gen, decoded_ans_argmax, decoded_ans_correct)]
    return decoded_generations

def generate_not_as_full_dist(
        model,
        ref_model,
        questions_inputs,
        answer_inputs,
        answer_prompts,
        cot_length,
        temperature=1.0,
        generations_per_prompt=1,
        reward_type="answer_generated",
):
    # shapes
    batch_size = questions_inputs["input_ids"].shape[0]
    batch_size = batch_size * generations_per_prompt
    vocab_size = model.config.vocab_size
    question_length = questions_inputs["input_ids"].shape[1]
    ans_length = answer_inputs["input_ids"].shape[1]
    answer_prompt_length = answer_prompts.shape[1]
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
        generated_cot = q_cot_ids[:, question_length:question_length+cot_length]
    # add answer prompts
    q_cot_prompt_ids = torch.cat((q_cot_ids, answer_prompts), dim=1)
    q_cot_prompt_mask = torch.ones_like(q_cot_prompt_ids)
    q_cot_prompt_mask[:, :question_length] = questions_inputs["attention_mask"]
    assert q_cot_prompt_ids.shape == (batch_size, question_length+cot_length+answer_prompt_length), f"{q_cot_prompt_ids.shape=}, {(batch_size, question_length, cot_length, answer_prompt_length)=}"
    if reward_type == "answer_generated":
        # generate answer
        with torch.no_grad():
            q_cot_prompt_ans_ids = model.generate(
                input_ids=q_cot_prompt_ids,
                attention_mask=q_cot_prompt_mask,
                max_new_tokens=ans_length,
                temperature=temperature,
                top_k=None,
                do_sample=True,
                eos_token_id=None,
            )
        generated_ans = q_cot_prompt_ans_ids[:, -ans_length:]
    elif reward_type == "answer_prob":
        # append answer
        q_cot_prompt_ans_ids = torch.cat((q_cot_prompt_ids, answer_inputs["input_ids"]), dim=1)
        generated_ans = None
    else:
        raise ValueError(f"{reward_type=}")
    q_cot_prompt_ans_mask = torch.cat((q_cot_prompt_mask, torch.ones_like(answer_inputs["input_ids"])), dim=1)
    assert q_cot_prompt_ans_ids.shape == (batch_size, question_length+cot_length+answer_prompt_length+ans_length), f"{q_cot_prompt_ans_ids.shape=}, {(batch_size, question_length, cot_length, answer_prompt_length, ans_length)=}"
    assert q_cot_prompt_ans_mask.shape == (batch_size, question_length+cot_length+answer_prompt_length+ans_length), f"{q_cot_prompt_ans_mask.shape=}, {(batch_size, question_length, cot_length, answer_prompt_length, ans_length)=}"
    # forward pass
    all_logits = model(
        input_ids=q_cot_prompt_ans_ids[:, :-1],
        attention_mask=q_cot_prompt_ans_mask[:, :-1],
        ).logits
    assert all_logits.shape == (batch_size, question_length+cot_length+answer_prompt_length+ans_length-1, vocab_size), f"{all_logits.shape=}, {(batch_size, question_length, cot_length, answer_prompt_length, ans_length, vocab_size)=}"
    # ref forward pass
    with torch.no_grad():
        ref_all_logits = ref_model(
            input_ids=q_cot_prompt_ans_ids[:, :-1],
            attention_mask=q_cot_prompt_ans_mask[:, :-1],
            ).logits
    assert ref_all_logits.shape == (batch_size, question_length+cot_length+answer_prompt_length+ans_length-1, vocab_size), f"{ref_all_logits.shape=}, {(batch_size, question_length, cot_length, answer_prompt_length, ans_length, vocab_size)=}"
    # cleanup
    del q_cot_ids, q_cot_prompt_ids, q_cot_prompt_mask, q_cot_prompt_ans_ids, q_cot_prompt_ans_mask
    torch.cuda.empty_cache()
    gc.collect()
    # return
    return all_logits, ref_all_logits, generated_cot, generated_ans

def generate_as_full_dist(
        model,
        ref_model,
        questions_inputs,
        answer_inputs,
        answer_prompts,
        cot_length,
        temperature=1.0,
        teacher_forcing=0.5,
):
    device = model.device
    # shapes
    batch_size = questions_inputs["input_ids"].shape[0]
    vocab_size = model.config.vocab_size
    question_length = questions_inputs["input_ids"].shape[1]
    ans_length = answer_inputs["input_ids"].shape[1]
    answer_prompt_length = answer_prompts.shape[1]
    # make attention mask and position ids
    attention_mask = torch.ones((batch_size, question_length+cot_length+answer_prompt_length+ans_length-1), device=device)
    attention_mask[:, :question_length] = questions_inputs["attention_mask"]
    position_ids = torch.cumsum(attention_mask, dim=1) - 1
    # question forward pass
    outputs = model(
        input_ids=questions_inputs["input_ids"][:, :-1],
        attention_mask=attention_mask[:, :question_length-1],
        position_ids=position_ids[:, :question_length-1],
    )
    past_key_values = outputs.past_key_values
    all_logits = outputs.logits
    next_input = questions_inputs["input_ids"][:, -1:]
    assert past_key_values[0][0].shape[2] == question_length-1, f"{past_key_values[0][0].shape=}, {question_length=}"
    assert past_key_values[0][0].shape[0] == batch_size, f"{past_key_values[0][0].shape=}, {batch_size=}"
    assert all_logits.shape == (batch_size, question_length-1, vocab_size), f"{all_logits.shape=}, {(batch_size, question_length, vocab_size)=}"
    assert next_input.shape == (batch_size, 1), f"{next_input.shape=}, {(batch_size, 1)=}"
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
    assert all_logits.shape == (batch_size, question_length+cot_length, vocab_size), f"{all_logits.shape=}, {(batch_size, question_length, cot_length, vocab_size)=}"
    assert next_input.shape == (batch_size, 1, vocab_size), f"{next_input.shape=}, {(batch_size, vocab_size)=}"
    # answer prompt forward pass
    logits, past_key_values = step(model,
        next_input=answer_prompts,
        attention_mask=attention_mask[:, :question_length+cot_length+answer_prompt_length],
        position_ids=position_ids[:, question_length+cot_length:question_length+cot_length+answer_prompt_length],
        past_key_values=past_key_values,
        as_full_distribution=False,
        )
    all_logits = torch.cat((all_logits, logits), dim=1)
    assert all_logits.shape == (batch_size, question_length+cot_length+answer_prompt_length, vocab_size), f"{all_logits.shape=}, {(batch_size, question_length, cot_length, answer_prompt_length, vocab_size)=}"
    # supervised answer generation
    prev_seq_length = question_length+cot_length+answer_prompt_length
    logits = logits[:, -1:]
    correct_answer_one_hots = torch.nn.functional.one_hot(answer_inputs["input_ids"], num_classes=vocab_size).float()
    assert correct_answer_one_hots.shape == (batch_size, ans_length, vocab_size), f"{correct_answer_one_hots.shape=}, {(batch_size, ans_length, vocab_size)=}"
    for i in range(ans_length-1):
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
    assert all_logits.shape == (batch_size, question_length+cot_length+answer_prompt_length+ans_length-1, vocab_size), f"{all_logits.shape=}, {(batch_size, question_length, cot_length, answer_prompt_length, ans_length, vocab_size)=}"
    # collect logits
    cot_logits = all_logits[:, question_length-1:question_length+cot_length-1]
    ans_logits = all_logits[:, -ans_length:]
    assert cot_logits.shape == (batch_size, cot_length, vocab_size), f"{cot_logits.shape=}, {(batch_size, cot_length, vocab_size)=}"
    assert ans_logits.shape == (batch_size, ans_length, vocab_size), f"{ans_logits.shape=}, {(batch_size, ans_length, vocab_size)=}"
    # ref forward pass
    with torch.no_grad():
        input_ids = torch.cat((
            torch.nn.functional.one_hot(questions_inputs["input_ids"], num_classes=vocab_size).float(),
            torch.nn.functional.softmax(cot_logits / temperature, dim=-1),
            torch.nn.functional.one_hot(answer_prompts, num_classes=vocab_size).float(),
            (
                teacher_forcing * correct_answer_one_hots[:, :-1]
                + (1.0 - teacher_forcing) * torch.nn.functional.softmax(ans_logits[:, :-1] / temperature, dim=-1)
            ),
        ), dim=1)
        assert input_ids.shape == (batch_size, question_length+cot_length+answer_prompt_length+ans_length-1, vocab_size), f"{input_ids.shape=}, {(batch_size, question_length, cot_length, answer_prompt_length, ans_length, vocab_size)=}"
        ref_all_logits, _ = step(ref_model,
            next_input=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            as_full_distribution=True,
            )
    assert ref_all_logits.shape == (batch_size, question_length+cot_length+answer_prompt_length+ans_length-1, vocab_size), f"{ref_all_logits.shape=}, {(batch_size, question_length, cot_length, answer_prompt_length, ans_length, vocab_size)=}"
    ### as a check, can check ref and model logits are the same at iter=0
    ### logits_diff will be ~0.005 with bf16, and ~1e-5 with full precision (ie. with ctx disabled).
    # all_logits_ = all_logits * attention_mask.unsqueeze(-1)
    # ref_all_logits_ = ref_all_logits * attention_mask.unsqueeze(-1)
    # logits_diff = (all_logits_ - ref_all_logits_).abs().mean(-1)
    # print(f"{logits_diff=}")
    # print(f"{logits_diff.mean()=}")

    # cleanup
    del logits, past_key_values, next_input, correct_answer_one_hots, input_ids
    torch.cuda.empty_cache()
    gc.collect()

    return all_logits, ref_all_logits

def get_generations(
        model,
        ref_model,
        tokenizer,
        questions_inputs,
        answer_inputs,
        answer_prompt_text,
        cot_length,
        temperature=1.0,
        as_full_distribution=False,
        teacher_forcing=0.5,
        generations_per_prompt=1,
        normalization_type="grpo",
        kl_type="per_token",
        reward_type="answer_generated",
        return_decoded=False,
):
    model = model.module if dist.is_initialized() else model
    ref_model = ref_model.module if dist.is_initialized() else ref_model
    # model.eval()
    # ref_model.eval()
    device = model.device
    prompt_batch_size = questions_inputs["input_ids"].shape[0]
    # add eot to answer
    answer_inputs = append_eos_token(answer_inputs, model.config.eos_token_id, pad_token_id=model.config.eos_token_id)
    # answer prompts
    answer_prompts = torch.tensor(tokenizer.encode(answer_prompt_text), dtype=torch.long, device=device).unsqueeze(0).repeat(prompt_batch_size, 1)
    # shapes
    vocab_size = model.config.vocab_size
    question_length = questions_inputs["input_ids"].shape[1]
    ans_length = answer_inputs["input_ids"].shape[1]
    answer_prompt_length = answer_prompts.shape[1]
    batch_size = prompt_batch_size*generations_per_prompt

    if as_full_distribution:
        assert generations_per_prompt == 1, f"{generations_per_prompt=}"
        assert normalization_type is None, f"{normalization_type=}"
        all_logits, ref_all_logits = generate_as_full_dist(
            model=model,
            ref_model=ref_model,
            questions_inputs=questions_inputs,
            answer_inputs=answer_inputs,
            answer_prompts=answer_prompts,
            cot_length=cot_length,
            temperature=temperature,
            teacher_forcing=teacher_forcing,
        )
        generated_cot = None
        generated_ans = None
    else:
        all_logits, ref_all_logits, generated_cot, generated_ans = generate_not_as_full_dist(
            model=model,
            ref_model=ref_model,
            questions_inputs=questions_inputs,
            answer_inputs=answer_inputs,
            answer_prompts=answer_prompts,
            cot_length=cot_length,
            temperature=temperature,
            generations_per_prompt=generations_per_prompt,
            reward_type=reward_type,
        )
    
    # collect logits
    cot_logits = all_logits[:, question_length-1:question_length+cot_length-1]
    ans_logits = all_logits[:, -ans_length:]
    ref_cot_logits = ref_all_logits[:, question_length-1:question_length+cot_length-1]
    ref_ans_logits = ref_all_logits[:, -ans_length:]
    assert all_logits.shape == (batch_size, question_length+cot_length+answer_prompt_length+ans_length-1, vocab_size), f"{all_logits.shape=}, {(batch_size, question_length, cot_length, answer_prompt_length, ans_length, vocab_size)=}"
    assert ref_all_logits.shape == (batch_size, question_length+cot_length+answer_prompt_length+ans_length-1, vocab_size), f"{ref_all_logits.shape=}, {(batch_size, question_length, cot_length, answer_prompt_length, ans_length, vocab_size)=}"
    assert cot_logits.shape == (batch_size, cot_length, vocab_size), f"{cot_logits.shape=}, {(batch_size, cot_length, vocab_size)=}"
    assert ans_logits.shape == (batch_size, ans_length, vocab_size), f"{ans_logits.shape=}, {(batch_size, ans_length, vocab_size)=}"
    assert ref_cot_logits.shape == (batch_size, cot_length, vocab_size), f"{ref_cot_logits.shape=}, {(batch_size, cot_length, vocab_size)=}"
    assert ref_ans_logits.shape == (batch_size, ans_length, vocab_size), f"{ref_ans_logits.shape=}, {(batch_size, ans_length, vocab_size)=}"

    # sample cot, answer
    if generated_cot is None:
        cot_dist = torch.distributions.Categorical(logits=cot_logits / temperature)
        generated_cot = cot_dist.sample()
    if generated_ans is None:
        ans_dist = torch.distributions.Categorical(logits=ans_logits / temperature)
        generated_ans = ans_dist.sample()
    assert generated_cot.shape == (batch_size, cot_length), f"{generated_cot.shape=}, {(batch_size, cot_length)=}"
    assert generated_ans.shape == (batch_size, ans_length), f"{generated_ans.shape=}, {(batch_size, ans_length)=}"

    # argmax cot, answer (for metrics)
    cot_argmax = cot_logits.argmax(dim=-1)
    ans_argmax = ans_logits.argmax(dim=-1)
    assert cot_argmax.shape == (batch_size, cot_length), f"{cot_argmax.shape=}, {(batch_size, cot_length)=}"
    assert ans_argmax.shape == (batch_size, ans_length), f"{ans_argmax.shape=}, {(batch_size, ans_length)=}"

    # calculate logps
    all_cot_logps = torch.nn.functional.log_softmax(cot_logits, dim=-1)
    all_ans_logps = torch.nn.functional.log_softmax(ans_logits, dim=-1)
    per_token_cot_logps = torch.gather(all_cot_logps, 2, generated_cot.unsqueeze(-1)).squeeze(-1)
    per_token_gen_ans_logps = torch.gather(all_ans_logps, 2, generated_ans.unsqueeze(-1)).squeeze(-1)
    per_token_correct_ans_logps = torch.gather(all_ans_logps, 2, answer_inputs["input_ids"].unsqueeze(-1)).squeeze(-1)
    cot_logp = per_token_cot_logps.sum(dim=-1)
    gen_ans_logp = (per_token_gen_ans_logps * answer_inputs["attention_mask"]).sum(dim=-1)
    correct_ans_logp = (per_token_correct_ans_logps * answer_inputs["attention_mask"]).sum(dim=-1)
    assert all_cot_logps.shape == (batch_size, cot_length, vocab_size), f"{all_cot_logps.shape=}, {(batch_size, cot_length, vocab_size)=}"
    assert all_ans_logps.shape == (batch_size, ans_length, vocab_size), f"{all_ans_logps.shape=}, {(batch_size, ans_length, vocab_size)=}"
    assert per_token_cot_logps.shape == (batch_size, cot_length), f"{per_token_cot_logps.shape=}, {(batch_size, cot_length)=}"
    assert per_token_gen_ans_logps.shape == (batch_size, ans_length), f"{per_token_gen_ans_logps.shape=}, {(batch_size, ans_length)=}"
    assert per_token_correct_ans_logps.shape == (batch_size, ans_length), f"{per_token_correct_ans_logps.shape=}, {(batch_size, ans_length)=}"
    assert cot_logp.shape == (batch_size,), f"{cot_logp.shape=}, {(batch_size)=}"
    assert gen_ans_logp.shape == (batch_size,), f"{gen_ans_logp.shape=}, {(batch_size)=}"
    assert correct_ans_logp.shape == (batch_size,), f"{correct_ans_logp.shape=}, {(batch_size)=}"

    # calculate reward
    with torch.no_grad():
        # answer generated correct
        generated_correct_ans = (generated_ans == answer_inputs["input_ids"]).all(dim=-1).float()
        # answer prob
        length_normalized_correct_ans_prob = torch.exp(correct_ans_logp / answer_inputs["attention_mask"].sum(dim=-1))
        if reward_type == "answer_generated":
            reward = generated_correct_ans
        elif reward_type == "answer_prob":
            reward = length_normalized_correct_ans_prob
        else:
            raise ValueError(f"{reward_type=}")
        # normalize reward
        if generations_per_prompt > 1:
            reward = reward.reshape(prompt_batch_size, generations_per_prompt)
            if normalization_type == "grpo":
                normalized_reward = (reward - reward.mean(1, keepdim=True)) / (reward.std(1, keepdim=True) + 1e-6)
            elif normalization_type == "rloo":
                group_sum = reward.sum(1, keepdim=True)
                normalized_reward = (reward - group_sum) / (generations_per_prompt - 1)
            elif normalization_type is None:
                normalized_reward = reward
            else:
                raise ValueError(f"{normalization_type=}")
            reward = reward.reshape(-1)
            normalized_reward = normalized_reward.reshape(-1)
        else:
            assert normalization_type is None, f"{normalization_type=}"
            normalized_reward = reward

    # normal kl
    ref_all_cot_logps = torch.nn.functional.log_softmax(ref_cot_logits, dim=-1)
    ref_all_ans_logps = torch.nn.functional.log_softmax(ref_ans_logits, dim=-1)
    ref_per_token_cot_logps = torch.gather(ref_all_cot_logps, 2, generated_cot.unsqueeze(-1)).squeeze(-1)
    ref_per_token_ans_logps = torch.gather(ref_all_ans_logps, 2, generated_ans.unsqueeze(-1)).squeeze(-1)
    cot_kl = (torch.exp(ref_per_token_cot_logps - per_token_cot_logps) - (ref_per_token_cot_logps - per_token_cot_logps) - 1)
    ans_kl = (torch.exp(ref_per_token_ans_logps - per_token_gen_ans_logps) - (ref_per_token_ans_logps - per_token_gen_ans_logps) - 1)
    ans_kl = ans_kl * answer_inputs["attention_mask"]
    sample_kl = cot_kl.sum(dim=-1) + ans_kl.sum(dim=-1)
    # full kl
    all_logps = torch.nn.functional.log_softmax(all_logits, dim=-1)
    ref_all_logps = torch.nn.functional.log_softmax(ref_all_logits, dim=-1)
    full_kl = (all_logps.exp() * (all_logps - ref_all_logps)).mean(-1).mean(-1)
    # kl
    kl = sample_kl if kl_type == "sample" else full_kl
    
    # entropy
    cot_pd = torch.nn.functional.softmax(cot_logits, dim=-1)
    ans_pd = torch.nn.functional.softmax(ans_logits, dim=-1)
    cot_entropy = torch.logsumexp(cot_logits, dim=-1) - torch.sum(cot_pd * cot_logits, dim=-1)
    ans_entropy = torch.logsumexp(ans_logits, dim=-1) - torch.sum(ans_pd * ans_logits, dim=-1)
    ans_entropy = ans_entropy * answer_inputs["attention_mask"]
    entropy = torch.cat((cot_entropy, ans_entropy), dim=-1).sum(dim=-1) / (cot_length + answer_inputs["attention_mask"].sum(dim=-1).float())
    assert cot_entropy.shape == (batch_size, cot_length), f"{cot_entropy.shape=}, {(batch_size, cot_length)=}"
    assert ans_entropy.shape == (batch_size, ans_length), f"{ans_entropy.shape=}, {(batch_size, ans_length)=}"
    assert entropy.shape == (batch_size,), f"{entropy.shape=}, {(batch_size)=}"

    # compute metrics
    with torch.no_grad():
        matching_positions = ((ans_argmax == answer_inputs["input_ids"]) & answer_inputs["attention_mask"]).sum(dim=-1)
        prop_correct = matching_positions.float() / answer_inputs["attention_mask"].sum(dim=-1).float()

    # metrics
    metrics = {}
    metrics["cot_logp"] = cot_logp.mean()
    metrics["gen_ans_logp"] = gen_ans_logp.mean()
    metrics["correct_ans_logp"] = correct_ans_logp.mean()
    metrics["prop_correct"] = prop_correct.mean()
    metrics["generated_correct_ans"] = generated_correct_ans.mean()
    metrics["length_normalized_correct_ans_prob"] = length_normalized_correct_ans_prob.mean()
    metrics["reward"] = reward.mean()
    metrics["normalized_reward_min"] = normalized_reward.min()
    metrics["normalized_reward_max"] = normalized_reward.max()
    metrics["cot_kl"] = cot_kl.mean()
    metrics["ans_kl"] = ans_kl.mean()
    metrics["sample_kl"] = sample_kl.mean()
    metrics["full_kl"] = full_kl.mean()
    metrics["cot_entropy"] = cot_entropy.mean()
    metrics["ans_entropy"] = ans_entropy.mean()
    metrics["entropy"] = entropy.mean()

    # return
    assert cot_logp.requires_grad == True, f"{cot_logp.requires_grad=}"
    assert correct_ans_logp.requires_grad == True, f"{correct_ans_logp.requires_grad=}"
    assert gen_ans_logp.requires_grad == True, f"{gen_ans_logp.requires_grad=}"
    assert normalized_reward.requires_grad == False, f"{normalized_reward.requires_grad=}"
    assert kl.requires_grad == True, f"{kl.requires_grad=}"
    assert entropy.requires_grad == True, f"{entropy.requires_grad=}"
    x = (cot_logp, correct_ans_logp, gen_ans_logp, normalized_reward, kl, entropy)
    if return_decoded:
        decoded_generations = decode(tokenizer, generated_cot, generated_ans, ans_argmax, answer_inputs["input_ids"])
        decoded_generations = [x + (a, b) for x, a, b in zip(decoded_generations, prop_correct, length_normalized_correct_ans_prob)]
    else:
        decoded_generations = None

    # cleanup
    del all_logits, ref_all_logits, generated_cot, generated_ans, all_cot_logps, all_ans_logps, ref_all_cot_logps, ref_all_ans_logps, per_token_cot_logps, per_token_gen_ans_logps, per_token_correct_ans_logps, cot_logp, gen_ans_logp, correct_ans_logp, reward, normalized_reward, kl, sample_kl, full_kl, cot_kl, ans_kl, cot_entropy, ans_entropy, entropy, matching_positions, prop_correct, generated_correct_ans, length_normalized_correct_ans_prob, cot_pd, ans_pd
    torch.cuda.empty_cache()
    gc.collect()
    return x, decoded_generations, metrics

    