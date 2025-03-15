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

def extract_solution(solution_str):
    solution = re.search(r"Answer: (\-?[0-9\.\,]+)", solution_str)
    if solution is None:
        return None
    extracted_answer = solution.group(1).replace(',', '').replace('_', '').replace(' ', '')
    # Remove a trailing period (but keep periods in the middle)
    extracted_answer = re.sub(r'\.$', '', extracted_answer)
    return extracted_answer

def careful_repeat(data, num_repeats):
    batch_size = data[list(data.keys())[0]].shape[0]
    for k, v in data.items():
        if v.ndim == 1:
            data[k] = v.unsqueeze(1).repeat(1, num_repeats).reshape(batch_size*num_repeats, *v.shape[1:])
        elif v.ndim == 2:
            data[k] = v.unsqueeze(1).repeat(1, num_repeats, 1).reshape(batch_size*num_repeats, *v.shape[1:])
    return data

def get_model_param_stats(model, ref_model):
    model_params = torch.cat([p.view(-1) for p in model.parameters() if p.requires_grad])
    ref_model_params = torch.cat([p.view(-1) for p in ref_model.parameters()])
    assert model_params.shape == ref_model_params.shape, f"{model_params.shape=} {ref_model_params.shape=}"
    return {
        "params_with_grads_mean": model_params.mean().item(),
        "params_with_grads_std": model_params.std().item(),
        "distance_to_ref": torch.nn.functional.mse_loss(model_params, ref_model_params),
    }

def batch_generate_with_method_low_memory(
        model,
        ref_model,
        max_length,
        questions_inputs,
        temperature=1.0,
        top_k=None,
        do_sample=True,
    ):
    metrics = {}
    # generation
    model = model.module if dist.is_initialized() else model
    batch_size, prompt_length = questions_inputs["input_ids"].shape
    with torch.no_grad():
        prompt_generations = model.generate(
                input_ids=questions_inputs["input_ids"],
                attention_mask=questions_inputs["attention_mask"],
                max_new_tokens=max_length-prompt_length,
                temperature=temperature,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=model.config.eos_token_id, # to avoid lots of prints but shouldn't change anything
            )
    # forward pass
    attention_mask = torch.ones_like(prompt_generations[:, :-1])
    attention_mask[:, :prompt_length] = questions_inputs["attention_mask"]
    position_ids = (torch.cumsum(attention_mask, dim=1) - 1)
    logits = model(
        input_ids=prompt_generations[:, :-1],
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        ).logits
    with torch.no_grad():
        ref_logits = ref_model(
            input_ids=prompt_generations[:, :-1],
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            ).logits
    generations = prompt_generations[:, prompt_length:]
    logits = logits[:, prompt_length-1:]
    ref_logits = ref_logits[:, prompt_length-1:]
    all_logps = torch.nn.functional.log_softmax(logits, dim=-1)
    all_ref_logps = torch.nn.functional.log_softmax(ref_logits, dim=-1)
    gen_per_token_logps = torch.gather(all_logps, 2, generations.to(torch.long).unsqueeze(-1)).squeeze(-1)
    ref_per_token_logps = torch.gather(all_ref_logps, 2, generations.to(torch.long).unsqueeze(-1)).squeeze(-1)

    eos_token_id = model.config.eos_token_id
    eos_occurrences = (generations == eos_token_id).int()
    eos_mask = torch.cumsum(eos_occurrences, dim=1) <= 1
    eos_mask = torch.cat([torch.ones((batch_size, 1), device=gen_per_token_logps.device), eos_mask[:, :-1]], dim=1) # include the eos token
    gen_per_token_logps = gen_per_token_logps * eos_mask
    ref_per_token_logps = ref_per_token_logps * eos_mask

    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    entropy = (entropy * eos_mask).mean(-1)
    metrics["logps"] = gen_per_token_logps.mean()
    metrics["logps_ref"] = ref_per_token_logps.mean()
    metrics["logps_diff"] = (gen_per_token_logps - ref_per_token_logps).mean()
    metrics["entropy"] = entropy.mean()
    metrics["entropy_std"] = entropy.std() if entropy.numel() > 1 else torch.tensor(0.0, device=entropy.device)
    metrics["length"] = eos_mask.sum(-1).float().mean()
    metrics["finished"] = (eos_occurrences.sum(-1) > 0).float().mean()

    x = (gen_per_token_logps, ref_per_token_logps, entropy)
    assert x[0].requires_grad == True, f"{x[0].requires_grad=}"
    assert x[1].requires_grad == False, f"{x[1].requires_grad=}"
    assert x[2].requires_grad == True, f"{x[2].requires_grad=}"

    return x, generations, metrics


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


if __name__ == "__main__":
    # test:
    # generate using our batch_generate_rnn function
    # check equal to using the .generate() method
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import Dataset

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model_name = "Qwen/Qwen2.5-0.5b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    ref_model.eval()

    dataset = Dataset.load_from_disk(f"./data/my_data/gsm8k/train")

    batch_size = 3
    questions_text = dataset["question"][:batch_size]
    answers_text = dataset["answer"][:batch_size]

    questions_inputs = tokenizer(questions_text, return_tensors="pt", padding=True, padding_side="left")
    answers_inputs = tokenizer(answers_text, return_tensors="pt", padding=True)
    questions_inputs = {k: v.to(device) for k, v in questions_inputs.items()}
    answers_inputs = {k: v.to(device) for k, v in answers_inputs.items()}

    def our_generate(num_steps_):
        x, generations, generation_metrics, past_key_values = batch_generate_rnn(
            model=model,
            ref_model=ref_model,
            questions_inputs=questions_inputs,
            answers_inputs=answers_inputs,
            max_length=num_steps_,
            step_for_answer=20,
            temperature=0.0,
            top_k=None,
            as_full_distribution=False,
            dot_by_dot=False,
            dot_by_dot_id=None,
            inject_answer_prompt=False,
            answer_prompt_ids=None,
            loss_type="pg",
            compute_everything=True,
        )
        return generations, past_key_values
    
    def our_generate_with_method(num_steps_):
        x, generations, generation_metrics, (gen_past_kvs, forward_past_kvs) = batch_generate_with_method(
            model=model,
            ref_model=ref_model,
            questions_inputs=questions_inputs,
            max_length=num_steps_,
            temperature=1.0,
            top_k=None,
            do_sample=False,
        )
        return generations, gen_past_kvs, forward_past_kvs
    
    def generate_with_generate(num_steps_):
        outputs = model.generate(
            input_ids=questions_inputs["input_ids"],
            attention_mask=questions_inputs["attention_mask"],
            max_new_tokens=num_steps_,
            temperature=1.0,
            top_k=None,
            do_sample=False,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        generations = outputs.sequences[:, questions_inputs["input_ids"].shape[1]:]
        past_key_values = outputs.past_key_values
        return generations, past_key_values
    
    def forward_pass(generations, prompt_past_key_values=None):
        attention_mask = torch.cat([questions_inputs["attention_mask"], torch.ones((batch_size, generations.shape[1]), device=device)], dim=1)
        attention_mask[:, :questions_inputs["attention_mask"].shape[1]] = questions_inputs["attention_mask"]
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1)[:, questions_inputs["attention_mask"].shape[1]:]
        outputs = model(input_ids=generations, attention_mask=attention_mask, position_ids=position_ids, past_key_values=prompt_past_key_values)
        past_key_values = outputs.past_key_values
        return past_key_values

    num_steps = 30
    prompt_length = questions_inputs["input_ids"].shape[1]
    
    # our generate
    generations1, past_key_values1 = our_generate(num_steps)
    print(f"\nREQUIRES GRAD our generate: {past_key_values1[0][0].requires_grad=}") # True
    # our generate with method
    generations2, past_key_values2, past_key_values3 = our_generate_with_method(num_steps)
    print(f"\nREQUIRES GRAD our generate with method: {past_key_values2[0][0].requires_grad=}") # False
    print(f"\nREQUIRES GRAD our generate with method: {past_key_values3[0][0].requires_grad=}") # True
    # generate with .generate() method
    generations4, past_key_values4 = generate_with_generate(num_steps)
    print(f"\nREQUIRES GRAD .generate() method: {past_key_values4[0][0].requires_grad=}") # False
    # forward pass to get grads
    prompt_past_key_values = [(k[:, :, :prompt_length], v[:, :, :prompt_length]) for k, v in past_key_values1]
    past_key_values5 = forward_pass(generations1[:, :-1], prompt_past_key_values)
    print(f"\nREQUIRES GRAD forward pass: {past_key_values5[0][0].requires_grad=}") # True


    padding_mask = torch.ones((batch_size, prompt_length-1+num_steps), device=device)
    padding_mask[:, :prompt_length] = questions_inputs["attention_mask"]
    print(f"\n{padding_mask.unsqueeze(1).unsqueeze(-1).shape=}")
    print(f"{past_key_values1[0][0].shape=}, {past_key_values1[0][1].shape=}")
    print(f"{past_key_values2[0][0].shape=}, {past_key_values2[0][1].shape=}")
    print(f"{past_key_values3[0][0].shape=}, {past_key_values3[0][1].shape=}")
    print(f"{past_key_values4[0][0].shape=}, {past_key_values4[0][1].shape=}")
    print(f"{past_key_values5[0][0].shape=}, {past_key_values5[0][1].shape=}")
    def mask_key_values(past_key_values):
        return [
            (key * padding_mask.unsqueeze(1).unsqueeze(-1), value * padding_mask.unsqueeze(1).unsqueeze(-1))
            for key, value in past_key_values
        ]
    past_key_values1 = mask_key_values(past_key_values1)
    past_key_values2 = mask_key_values(past_key_values2)
    past_key_values3 = mask_key_values(past_key_values3)
    past_key_values4 = mask_key_values(past_key_values4)
    past_key_values5 = mask_key_values(past_key_values5)

    # compare some keys and values
    for l, (kv1, kv2, kv3, kv4, kv5) in enumerate(zip(past_key_values1, past_key_values2, past_key_values3, past_key_values4, past_key_values5)):
        if l == len(past_key_values1) - 1:
            print(f"Layer {l}")
            for t in range(kv1[0].shape[2]):
                if t < 2 or t > kv1[0].shape[2] - 3:
                    print(f"START {t=}")
                    print(f"{t=}: keys: {kv1[0].shape}, {kv2[0].shape}, {kv3[0].shape}, {kv4[0].shape}, {kv5[0].shape}")
                    print(f"{t=}: keys: {kv1[0][:, 0, t, :3].flatten().tolist()}")
                    print(f"{t=}: keys: {kv2[0][:, 0, t, :3].flatten().tolist()}")
                    print(f"{t=}: keys: {kv3[0][:, 0, t, :3].flatten().tolist()}")
                    print(f"{t=}: keys: {kv4[0][:, 0, t, :3].flatten().tolist()}")
                    print(f"{t=}: keys: {kv5[0][:, 0, t, :3].flatten().tolist()}")
                    print(f"{t=}: values: {kv1[1].shape}, {kv2[1].shape}, {kv3[1].shape}, {kv4[1].shape}, {kv5[1].shape}")
                    print(f"{t=}: values: {kv1[1][:, 0, t, :3].flatten().tolist()}")
                    print(f"{t=}: values: {kv2[1][:, 0, t, :3].flatten().tolist()}")
                    print(f"{t=}: values: {kv3[1][:, 0, t, :3].flatten().tolist()}")
                    print(f"{t=}: values: {kv4[1][:, 0, t, :3].flatten().tolist()}")
                    print(f"{t=}: values: {kv5[1][:, 0, t, :3].flatten().tolist()}\n")
                    print(f"END {t=}")
                    print(f"{t=}: keys: {kv1[0].shape}, {kv2[0].shape}, {kv3[0].shape}, {kv4[0].shape}, {kv5[0].shape}")
                    print(f"{t=}: keys: {kv1[0][:, 0, t, -3:].flatten().tolist()}")
                    print(f"{t=}: keys: {kv2[0][:, 0, t, -3:].flatten().tolist()}")
                    print(f"{t=}: keys: {kv3[0][:, 0, t, -3:].flatten().tolist()}")
                    print(f"{t=}: keys: {kv4[0][:, 0, t, -3:].flatten().tolist()}")
                    print(f"{t=}: keys: {kv5[0][:, 0, t, -3:].flatten().tolist()}")
                    print(f"{t=}: values: {kv1[1].shape}, {kv2[1].shape}, {kv3[1].shape}, {kv4[1].shape}, {kv5[1].shape}")
                    print(f"{t=}: values: {kv1[1][:, 0, t, -3:].flatten().tolist()}")
                    print(f"{t=}: values: {kv2[1][:, 0, t, -3:].flatten().tolist()}")
                    print(f"{t=}: values: {kv3[1][:, 0, t, -3:].flatten().tolist()}")
                    print(f"{t=}: values: {kv4[1][:, 0, t, -3:].flatten().tolist()}")
                    print(f"{t=}: values: {kv5[1][:, 0, t, -3:].flatten().tolist()}\n")

    # compare generations
    for i in range(batch_size):
        print(f"GENERATION {i}")
        print(tokenizer.decode(generations1[i]))
        print(tokenizer.decode(generations2[i]))
        print()

    # check equal
    for i, ((key1, value1), (key2, value2), (key3, value3), (key4, value4), (key5, value5)) in enumerate(zip(past_key_values1, past_key_values2, past_key_values3, past_key_values4, past_key_values5)):
        assert key1.shape == key2.shape == key3.shape == key4.shape == key5.shape, f"{key1.shape=}, {key2.shape=}, {key3.shape=}, {key4.shape=}, {key5.shape=}"
        assert value1.shape == value2.shape == value3.shape == value4.shape == value5.shape, f"{value1.shape=}, {value2.shape=}, {value3.shape=}, {value4.shape=}, {value5.shape=}"
        print(f"Layer {i}: Keys: std1={key1.std().item():.4f}, std2={key2.std().item():.4f}, std3={key3.std().item():.4f}, max_diff 1 vs 2={(key1-key2).abs().max().item():.4f}, max_diff 1 vs 3={(key1-key3).abs().max().item():.4f}, max_diff 1 vs 4={(key1-key4).abs().max().item():.4f}, max_diff 1 vs 5={(key1-key5).abs().max().item():.4f}")
        print(f"Layer {i}: Values: std1={value1.std().item():.4f}, std2={value2.std().item():.4f}, std3={value3.std().item():.4f}, max_diff 1 vs 2={(value1-value2).abs().max().item():.4f}, max_diff 1 vs 3={(value1-value3).abs().max().item():.4f}, max_diff 1 vs 4={(value1-value4).abs().max().item():.4f}, max_diff 1 vs 5={(value1-value5).abs().max().item():.4f}")
        assert torch.allclose(key1, key2), f"{key1=}, {key2=}"
        assert torch.allclose(value1, value2), f"{value1=}, {value2=}"
        assert torch.allclose(key1, key3, atol=1e-5, rtol=1e-3), f"{key1=}, {key3=}"
        assert torch.allclose(value1, value3, atol=1e-5, rtol=1e-3), f"{value1=}, {value3=}"
        assert torch.allclose(key1, key4, atol=1e-5, rtol=1e-3), f"{key1=}, {key4=}"
        assert torch.allclose(value1, value4, atol=1e-5, rtol=1e-3), f"{value1=}, {value4=}"
        assert torch.allclose(key1, key5, atol=1e-5, rtol=1e-3), f"{key1=}, {key5=}"
        assert torch.allclose(value1, value5, atol=1e-5, rtol=1e-3), f"{value1=}, {value5=}"

    print("Past key values are the same")
    print("Test passed")

    ### TIMING TEST
    # import time
    # start_time = time.time()
    # batch_size = 3
    # num_steps = 30
    # num_rounds = 10
    # for i in range(num_rounds):
    #     batch = dataset[i*batch_size:(i+1)*batch_size]
    #     questions_text = batch["question"]
    #     answers_text = batch["answer"]
    #     questions_inputs = tokenizer(questions_text, return_tensors="pt", padding=True, padding_side="left")
    #     answers_inputs = tokenizer(answers_text, return_tensors="pt", padding=True)
    #     questions_inputs = {k: v.to(device) for k, v in questions_inputs.items()}
    #     answers_inputs = {k: v.to(device) for k, v in answers_inputs.items()}
    #     x, generations, generation_metrics, past_key_values1 = batch_generate_rnn(
    #         model=model,
    #         ref_model=ref_model,
    #         questions_inputs=questions_inputs,
    #         answers_inputs=answers_inputs,
    #         max_length=num_steps,
    #         step_for_answer=20,
    #         temperature=0.0,
    #         top_k=None,
    #         as_full_distribution=False,
    #         dot_by_dot=False,
    #         dot_by_dot_id=None,
    #         inject_answer_prompt=False,
    #         answer_prompt_ids=None,
    #         loss_type="pg",
    #         compute_everything=False,
    #     )
    # time1 = time.time() - start_time

    # start_time = time.time()
    # for i in range(num_rounds):
    #     with torch.no_grad():
    #         batch = dataset[i*batch_size:(i+1)*batch_size]
    #         questions_text = batch["question"]
    #         answers_text = batch["answer"]
    #         questions_inputs = tokenizer(questions_text, return_tensors="pt", padding=True, padding_side="left")
    #         answers_inputs = tokenizer(answers_text, return_tensors="pt", padding=True)
    #         questions_inputs = {k: v.to(device) for k, v in questions_inputs.items()}
    #         answers_inputs = {k: v.to(device) for k, v in answers_inputs.items()}
    #         outputs = model.generate(
    #             input_ids=questions_inputs["input_ids"],
    #             attention_mask=questions_inputs["attention_mask"],
    #             # position_ids=None,
    #             # past_key_values=None,
    #             max_new_tokens=num_steps+1,
    #             temperature=0.0,
    #             top_k=None,
    #             do_sample=False,
    #             # pad_token_id=tokenizer.pad_token_id,
    #             # use_cache=True,
    #             return_dict_in_generate=True,
    #         )
    #     outputs = model(input_ids=outputs.sequences)
    # for k, v in outputs.items():
    #     print(k)
    # time2 = time.time() - start_time

    # print(f"Time for our function: {time1:.4f}")
    # print(f"Time for .generate() method: {time2:.4f}")



    



    