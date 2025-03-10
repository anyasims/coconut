import torch
import torch.distributed as dist

def careful_repeat(data, num_repeats):
    if isinstance(data, dict):
        batch_size = data[list(data.keys())[0]].shape[0]
        for k, v in data.items():
            if v.ndim == 1:
                data[k] = v.unsqueeze(1).repeat(1, num_repeats).reshape(batch_size*num_repeats, *v.shape[1:])
            elif v.ndim == 2:
                data[k] = v.unsqueeze(1).repeat(1, num_repeats, 1).reshape(batch_size*num_repeats, *v.shape[1:])
    elif isinstance(data, torch.Tensor):
        if data.ndim == 1:
            data = data.unsqueeze(1).repeat(1, num_repeats).reshape(data.shape[0]*num_repeats, *data.shape[1:])
        elif data.ndim == 2:
            data = data.unsqueeze(1).repeat(1, num_repeats, 1).reshape(data.shape[0]*num_repeats, *data.shape[1:])
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
        include_cot=True,
        include_answer=True,
):
    metrics = {}
    model = model.module if dist.is_initialized() else model
    device = model.device
    # add eot to answer
    answer_inputs = append_eos_token(answer_inputs, model.config.eos_token_id, model.config.pad_token_id)
    # shapes
    vocab_size = model.config.vocab_size
    batch_size = questions_inputs["input_ids"].shape[0]
    question_length = questions_inputs["input_ids"].shape[1]
    answer_length = answer_inputs["input_ids"].shape[1]
    answer_prompt_length = answer_prompts.shape[1]
    # answer prompts
    answer_prompts = torch.tensor(tokenizer.encode(">>Answer:"), dtype=torch.int, device=device).unsqueeze(0).repeat(batch_size, 1)
    if not as_full_distribution:
        # repeat
        questions_inputs = careful_repeat(questions_inputs, generations_per_prompt)
        answer_inputs = careful_repeat(answer_inputs, generations_per_prompt)
        answer_prompts = careful_repeat(answer_prompts, generations_per_prompt)
        # generate cots
        with torch.no_grad():
            cot_generations = model.generate(
                    input_ids=questions_inputs["input_ids"],
                    attention_mask=questions_inputs["attention_mask"],
                    max_new_tokens=cot_length,
                    temperature=temperature,
                    top_k=None,
                    do_sample=True,
                    eos_token_id=None,
                )
            generated_cots = cot_generations[:, question_length:question_length+cot_length]
        # add answer prompts
        q_cot_ids = torch.cat((cot_generations, answer_prompts), dim=1)
        q_cot_mask = torch.ones_like(q_cot_ans_ids)
        q_cot_mask[:, :question_length] = questions_inputs["attention_mask"]
        if reward_type == "answer_generated":
            # generate answer
            with torch.no_grad():
                q_cot_ans_ids = model.generate(
                    input_ids=q_cot_ids,
                    attention_mask=q_cot_mask,
                    max_new_tokens=answer_length,
                    temperature=temperature,
                    top_k=None,
                    do_sample=True,
                    eos_token_id=None,
                )
            answer = q_cot_ans_ids[:, -answer_length:]
        elif reward_type == "answer_prob":
            # append answer
            answer = answer_inputs["input_ids"]
            q_cot_ans_ids = torch.cat((q_cot_ids, answer), dim=1)
        else:
            raise ValueError(f"{reward_type=}")
        # forward pass
        q_cot_ans_mask = torch.cat((q_cot_mask, torch.ones_like(answer_inputs["input_ids"])), dim=1)
        all_logits = model(
            input_ids=q_cot_ans_ids[:, :-1],
            attention_mask=q_cot_ans_mask[:, :-1],
            ).logits
        # ref forward pass
        with torch.no_grad():
            ref_all_logits = ref_model(
                input_ids=q_cot_ans_ids[:, :-1],
                attention_mask=q_cot_ans_mask,
                ).logits
        # collect logits
        cot_logits = all_logits[:, question_length:question_length+cot_length]
        answer_logits = all_logits[:, -answer_length:]
    
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
        for t in range(cot_length):
            logits, past_key_values = step(model,
                next_input=next_input,
                attention_mask=attention_mask[:, :question_length+t],
                position_ids=position_ids[:, question_length+t-1:question_length+t],
                past_key_values=past_key_values,
                as_full_distribution=False if t == 0 else True,
                )
        next_input = torch.nn.functional.softmax(logits / temperature, dim=-1)
        all_logits = torch.cat((logits, logits), dim=1)
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
        cot_logits = all_logits[:, question_length:question_length+cot_length-1]
        answer_logits = all_logits[:, -answer_length:]
        # set answer
        if reward_type == "answer_generated":
            # answer = sampled from answer_logits
            dist = torch.distributions.Categorical(logits=answer_logits)  # Create a categorical distribution from logits
            answer = dist.sample()
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
                    teacher_forcing * correct_answer_one_hots
                    + (1.0 - teacher_forcing) * torch.nn.functional.softmax(answer_logits[:, :-1] / temperature, dim=-1)
                ),
            ), dim=1)
            ref_all_logits = step(ref_model,
                next_input=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                as_full_distribution=True,
                ).logits

    ############################################################################################################

    # calculate logps
    all_cot_logps = torch.nn.functional.log_softmax(cot_logits, dim=-1)
    per_token_cot_logps = torch.gather(all_cot_logps, 2, generated_cots.unsqueeze(-1)).squeeze(-1)
    cot_logp = per_token_cot_logps.sum(dim=-1)
    all_answer_logps = torch.nn.functional.log_softmax(answer_logits, dim=-1)
    per_token_answer_logps = torch.gather(all_answer_logps, 2, answer.unsqueeze(-1)).squeeze(-1)
    answer_logp = (per_token_answer_logps * answer_inputs["attention_mask"]).sum(dim=-1)
    # calculate reward
    with torch.no_grad():
        if reward_type == "answer_generated":
            generated_answer = torch.where(answer_inputs["attention_mask"], answer, torch.tensor(-1, dtype=torch.int, device=device))
            correct_answer = torch.where(answer_inputs["attention_mask"], answer_inputs["input_ids"], torch.tensor(-1, dtype=torch.int, device=device))
            reward = (generated_answer == correct_answer).all(dim=-1).float()
        elif reward_type == "answer_prob":
            answer_logp_length_normalized = answer_logp / answer_inputs["attention_mask"].sum(dim=-1)
            reward = answer_logp_length_normalized.exp()
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
        pass
    elif kl_type == "full":
        pass
    else:
        raise ValueError(f"{kl_type=}")
    
    # entropy
    pd = torch.nn.functional.softmax(all_logits, dim=-1)
    entropy = torch.logsumexp(all_logits, dim=-1) - torch.sum(pd * all_logits, dim=-1)
    

        # # kl
        # if kl_type == "normal":
        #     kl = (torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1).mean(-1)
        # elif kl_type == "full":
        #     kl = (all_logps.exp() * (all_logps - ref_all_logps)).sum(-1).mean(-1)

        # # entropy
        # pd = torch.nn.functional.softmax(logits, dim=-1)
        # entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)

        # # return
        # assert cot_logp.requires_grad == True, f"{cot_logp.requires_grad=}"
        # assert answer_logp.requires_grad == True, f"{answer_logp.requires_grad=}"
        # assert normalized_reward.requires_grad == False, f"{normalized_reward.requires_grad=}"
        # assert kl.requires_grad == True, f"{kl.requires_grad=}"
        # assert entropy.requires_grad == True, f"{entropy.requires_grad=}"
        # x = (cot_logp, answer_logp, normalized_reward, kl, entropy)
        # generations = question_cot_ans_ids[:, question_length:]
        # return x, generations, reward, metrics

    ### ENTROPY AND KL
    all_logps = torch.nn.functional.log_softmax(all_logits, dim=-1)
    entropy = torch.logsumexp(all_logits, dim=-1) - torch.sum(all_logps.exp() * all_logits, dim=-1).mean(-1)
    full_kl = (all_logps.exp() * (all_logps - all_ref_logps)).sum(-1).mean(-1)

    metrics["full_kl"] = full_kl.mean()
    metrics["entropy"] = entropy.mean()
    metrics["prompt_length"] = torch.tensor(prompt_length, device=device)
    metrics["length"] = torch.tensor(all_generations.shape[1], device=device)
    metrics["total_length"] = torch.tensor(prompt_length + all_generations.shape[1], device=device)
    metrics.update(supervised_metrics)

    x = (answer_logp, entropy, full_kl)
    assert x[0].requires_grad == True, f"{x[0].requires_grad=}"
    assert x[1].requires_grad == True, f"{x[1].requires_grad=}"
    assert x[2].requires_grad == True, f"{x[2].requires_grad=}"

    return x, all_generations, metrics








        # full dist cot generation

        
        
        # calculate logps
        # reward
        # ref forward pass
        # calculate ref logps
        # calculate pg / logp loss
        # kl
        # entropy
        x = (pg_loss, kl, entropy)
        generations = question_cot_ans_ids[:, question_length:]
        return x, generations, reward, metrics





import re
import torch
import torch.distributed as dist


def supervise_answer(
        model,
        logits,
        all_logits,
        all_generations,
        past_key_values,
        answers_text,
        tokenizer,
        temperature,
        attention_mask,
        position_ids,
        prompt_length,
        max_length_reached=False,
        steps_if_no_eot=100,
        teacher_forcing=0.5
        ):

    batch_size = logits.shape[0]
    assert batch_size == 1, f"{batch_size=}"
    gen = torch.argmax(logits, dim=-1).item()
    answer_prompt_generated = gen in [tokenizer.encode(f"Answer")[0], tokenizer.encode(f" Answer")[0]]
    eot_generated = gen == tokenizer.eos_token_id
    min_length_reached = all_generations.shape[1] >= steps_if_no_eot
    if max_length_reached or (answer_prompt_generated and min_length_reached) or eot_generated:
        if max_length_reached:
            target_tokens = tokenizer.encode(f"... Answer: {answers_text[0]}.") + [tokenizer.eos_token_id]
            answer_inputs = torch.nn.functional.softmax(all_logits[:, steps_if_no_eot-1:steps_if_no_eot] / temperature, dim=-1)
            answer_logits = all_logits[:, steps_if_no_eot:steps_if_no_eot+1]
            past_key_values = [
                (kv[0][:, :, :prompt_length+steps_if_no_eot], kv[1][:, :, :prompt_length+steps_if_no_eot])
                for kv in past_key_values
            ]
            all_generations = all_generations[:, :steps_if_no_eot]
            all_logits = all_logits[:, :steps_if_no_eot]
        else:
            answer_inputs = torch.nn.functional.softmax(all_logits[:, -1:] / temperature, dim=-1)
            answer_logits = logits
            if gen in [tokenizer.encode(f"Answer")[0], tokenizer.encode(f" Answer")[0]]:
                target_tokens = [gen] + tokenizer.encode(f": {answers_text[0]}.") + [tokenizer.eos_token_id]
            elif gen == tokenizer.eos_token_id:
                ends_with_space = tokenizer.decode(all_generations[0, -2]).endswith(" ")
                answer_prompt = "Answer" if ends_with_space else " Answer"
                target_tokens = tokenizer.encode(f"{answer_prompt}: {answers_text[0]}.") + [tokenizer.eos_token_id]
            else:
                raise ValueError(f"Should not reach here, {gen=}")
            
        target_tokens = torch.tensor(target_tokens, device=model.device).unsqueeze(0)
        vocab_size = model.module.config.vocab_size if dist.is_initialized() else model.config.vocab_size
        target_one_hots = torch.nn.functional.one_hot(target_tokens, num_classes=vocab_size).float()
        prev_seq_length = past_key_values[0][0].shape[2]

        for i in range(target_tokens.shape[1]-1):
            prev_logits = answer_logits[:, i:i+1]
            next_input = torch.nn.functional.softmax(prev_logits / temperature, dim=-1)
            next_input = teacher_forcing * target_one_hots[:, i:i+1] + (1.0 - teacher_forcing) * next_input
            # print(f"\n{i=}/{target_tokens.shape[1]-1}")
            # print(f"prev prediction: {tokenizer.decode(torch.argmax(prev_logits, dim=-1).squeeze().tolist())}")
            # print(f"next input: {tokenizer.decode(torch.argmax(next_input, dim=-1).squeeze().tolist())}")

            logits, past_key_values = step(model,
                next_input=next_input,
                attention_mask=attention_mask[:, :prev_seq_length+i+1],
                position_ids=position_ids[:, prev_seq_length+i:prev_seq_length+i+1],
                past_key_values=past_key_values,
                as_full_distribution=True,
                )
            answer_inputs = torch.cat((answer_inputs, next_input), dim=1)
            answer_logits = torch.cat((answer_logits, logits), dim=1)
            # print(f"next prediction: {tokenizer.decode(torch.argmax(logits, dim=-1).squeeze().tolist())}")

        answer_logps = torch.nn.functional.log_softmax(answer_logits, dim=-1)
        per_token_logps = torch.gather(answer_logps, 2, target_tokens.to(torch.long).unsqueeze(-1)).squeeze(-1)
        answer_logp = per_token_logps.sum()

        # for rest of function
        supervised_answer_generations = torch.argmax(answer_logits, dim=-1)
        all_generations = torch.cat((all_generations, supervised_answer_generations), dim=1)
        all_inputs = torch.cat((torch.nn.functional.softmax(all_logits[:, :-1] / temperature, dim=-1), answer_inputs), dim=1)
        all_logits = torch.cat((all_logits, answer_logits), dim=1)

        # print(f"Supervised generation fisnished:")
        # ins = [tokenizer.decode([int(i)]) for i in torch.argmax(answer_inputs, dim=-1)[0].tolist()]
        # outs = [tokenizer.decode([int(i)]) for i in supervised_answer_generations[0].tolist()]
        # tars = [tokenizer.decode([int(i)]) for i in target_tokens[0].tolist()]
        # for i, (in_, out_, tar_) in enumerate(zip(ins, outs, tars)):
        #     print(f"{i=},               {in_}             {out_}               {tar_}")
        # print(f"DECODED: {tokenizer.batch_decode(supervised_answer_generations)}")
        # print(f"DECODED SKIP: {tokenizer.batch_decode(supervised_answer_generations, skip_special_tokens=True)}")
        # print(f"DECODED NO SKIP: {tokenizer.batch_decode(supervised_answer_generations, skip_special_tokens=False)}")
        # print(f"ALL DECODED: {tokenizer.batch_decode(all_generations)}")


        metrics = {}
        metrics["answer_logps"] = answer_logp
        metrics["answer_perplexity"] = torch.exp(-per_token_logps.mean())
        metrics["max_length_reached"] = torch.tensor(max_length_reached, device=model.device, dtype=torch.float)
        metrics["answer_prompt_generated"] = torch.tensor(answer_prompt_generated, device=model.device, dtype=torch.float)
        metrics["min_length_reached"] = torch.tensor(min_length_reached, device=model.device, dtype=torch.float)
        metrics["eot_generated"] = torch.tensor(eot_generated, device=model.device, dtype=torch.float)
        metrics["tokens_correct"] = (supervised_answer_generations == target_tokens).float().mean()

        return answer_logp, all_generations, all_inputs, all_logits, metrics
    else:
        return None


def get_model_param_stats(model, ref_model):
    model_params = torch.cat([p.view(-1) for p in model.parameters() if p.requires_grad])
    ref_model_params = torch.cat([p.view(-1) for p in ref_model.parameters()])
    assert model_params.shape == ref_model_params.shape, f"{model_params.shape=} {ref_model_params.shape=}"
    return {
        "params_with_grads_mean": model_params.mean().item(),
        "params_with_grads_std": model_params.std().item(),
        "distance_to_ref": torch.nn.functional.mse_loss(model_params, ref_model_params),
    }


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


def batch_generate_rnn_full_dist(
        model,
        ref_model,
        tokenizer,
        questions_inputs,
        answers_text,
        max_length=200,
        temperature=1.0,
        teacher_forcing=0.5,
        steps_if_no_eot=100,
    ):
    metrics = {}
    device = model.device
    assert device.type == "cuda", f"{model.device=}"
    assert ref_model.device == device, f"{ref_model.device=}, {device=}"
    model.eval()
    ref_model.eval()
    batch_size, prompt_length = questions_inputs["input_ids"].shape
    assert batch_size == 1, f"{batch_size=}"
    vocab_size = model.module.config.vocab_size if dist.is_initialized() else model.config.vocab_size

    answer_buffer = 20
    attention_mask = torch.ones((batch_size, max_length+answer_buffer), device=device)
    attention_mask[:, :prompt_length] = questions_inputs["attention_mask"]
    position_ids = (torch.cumsum(attention_mask, dim=1) - 1)

    #### PROMPT FORWARD PASS
    with torch.no_grad():
        outputs = model(
            input_ids=questions_inputs["input_ids"][:, :-1],
            attention_mask=attention_mask[:, :prompt_length-1],
            position_ids=position_ids[:, :prompt_length-1],
        )
    past_key_values = outputs.past_key_values
    next_input = questions_inputs["input_ids"][:, -1:]

    all_logits = torch.zeros((batch_size, 0, vocab_size), device=device)
    all_generations = torch.zeros((batch_size, 0), device=device, dtype=torch.int)

    ### COT FORWARD PASSES
    for t in range(max_length-prompt_length):
        logits, past_key_values = step(model,
            next_input=next_input,
            attention_mask=attention_mask[:, :prompt_length+t],
            position_ids=position_ids[:, prompt_length+t-1:prompt_length+t],
            past_key_values=past_key_values,
            as_full_distribution=False if t == 0 else True,
            )
        supervised_output = supervise_answer(
                model=model,
                logits=logits,
                all_logits=all_logits,
                all_generations=all_generations,
                past_key_values=past_key_values,
                answers_text=answers_text,
                tokenizer=tokenizer,
                temperature=temperature,
                attention_mask=attention_mask,
                position_ids=position_ids,
                prompt_length=prompt_length,
                max_length_reached=t == max_length-prompt_length-1,
                steps_if_no_eot=steps_if_no_eot,
                teacher_forcing=teacher_forcing,
            )
        if supervised_output is not None:
            answer_logp, all_generations, all_inputs, all_logits, supervised_metrics = supervised_output
            break

        next_input = torch.nn.functional.softmax(logits / temperature, dim=-1)
        next_generation = torch.argmax(logits, dim=-1)
        all_logits = torch.cat((all_logits, logits), dim=1)
        all_generations = torch.cat((all_generations, next_generation), dim=1)

    

    ### REFERENCE MODEL
    with torch.no_grad():
        # make prompt_input_ids into 1-hot vectors
        prompt_inputs = torch.nn.functional.one_hot(questions_inputs["input_ids"], num_classes=vocab_size).float()
        inputs = torch.cat((prompt_inputs, all_inputs), dim=1)

        ref_logits, _ = step(ref_model,
            next_input=inputs,
            attention_mask=attention_mask[:, :inputs.shape[1]],
            position_ids=position_ids[:, :inputs.shape[1]],
            past_key_values=None,
            as_full_distribution=True,
            )
        ref_logits = ref_logits[:, prompt_length-1:]
        all_ref_logps = torch.nn.functional.log_softmax(ref_logits, dim=-1)

    ### ENTROPY AND KL
    all_logps = torch.nn.functional.log_softmax(all_logits, dim=-1)
    entropy = torch.logsumexp(all_logits, dim=-1) - torch.sum(all_logps.exp() * all_logits, dim=-1).mean(-1)
    full_kl = (all_logps.exp() * (all_logps - all_ref_logps)).sum(-1).mean(-1)

    metrics["full_kl"] = full_kl.mean()
    metrics["entropy"] = entropy.mean()
    metrics["prompt_length"] = torch.tensor(prompt_length, device=device)
    metrics["length"] = torch.tensor(all_generations.shape[1], device=device)
    metrics["total_length"] = torch.tensor(prompt_length + all_generations.shape[1], device=device)
    metrics.update(supervised_metrics)

    x = (answer_logp, entropy, full_kl)
    assert x[0].requires_grad == True, f"{x[0].requires_grad=}"
    assert x[1].requires_grad == True, f"{x[1].requires_grad=}"
    assert x[2].requires_grad == True, f"{x[2].requires_grad=}"

    return x, all_generations, metrics


# @torch.no_grad()
def batch_generate_rnn(
        model,
        ref_model,
        tokenizer,
        questions_inputs,
        answers_text,
        max_length=200,
        temperature=1.0,
        loss_type="pg",
        logp_teacher_forcing=False,
        logp_steps_if_no_eot=100,
    ):
    if loss_type == "pg":
        return batch_generate_with_method_low_memory(
            model=model,
            ref_model=ref_model,
            questions_inputs=questions_inputs,
            max_length=max_length,
            temperature=temperature,
        )
    elif loss_type == "logp":
        return batch_generate_rnn_full_dist(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            questions_inputs=questions_inputs,
            answers_text=answers_text,
            max_length=max_length,
            temperature=temperature,
            teacher_forcing=logp_teacher_forcing,
            steps_if_no_eot=logp_steps_if_no_eot,
        )
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")





    

    



    