import torch


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

def get_next(logitss, temperature=0.0, top_k=None):
    """Get the next token"""
    batch_size, seq_length, vocab_size = logitss.shape
    assert seq_length == 1
    logitss = logitss.squeeze(1)
    if temperature == 0.0:
        token_ids = torch.argmax(logitss, dim=-1)
        probdists = torch.zeros_like(logitss)
        probdists[torch.arange(batch_size), token_ids] = 1.0
    else:
        logitss = logitss / temperature
        if top_k is not None:
            logitss_k, idxs_k = torch.topk(logitss, min(top_k, vocab_size), dim=-1) # (batch_size, top_k)
            probs_k = torch.nn.functional.softmax(logitss_k, dim=-1) # (batch_size, top_k)
            idxs = torch.multinomial(probs_k, num_samples=1).squeeze(1) # (batch_size,)
            token_ids = idxs_k[torch.arange(batch_size), idxs] # (batch_size)
            probdists = torch.zeros_like(logitss)
            # next_probdist_s[torch.arange(batch_size), idx_s_k.squeeze(1)] = probs_k.squeeze(1)
            probdists.scatter_(1, idxs_k, probs_k)
            if top_k == 1:
                token_ids2 = torch.argmax(logitss, dim=-1)
                probdists2 = torch.zeros_like(logitss)
                probdists2[torch.arange(batch_size), token_ids2] = 1.0
                assert (token_ids == token_ids2).all(), f"{token_ids=}, {token_ids2=}"
                assert torch.allclose(probdists, probdists2), f"{probdists2=}, {probdists2=}"
        else:
            probdists = torch.nn.functional.softmax(logitss, dim=-1) # (batch_size, vocab_size)
            token_ids = torch.multinomial(probdists, num_samples=1).squeeze(1) # (batch_size)

    token_ids = token_ids.unsqueeze(1)
    probdists = probdists.unsqueeze(1)
    return token_ids, probdists

def single_step(model, next_input, attention_mask, position_ids, past_key_values, as_full_distribution=False):
    batch_size = next_input.shape[0]
    prev_seq_length = past_key_values[0][0].shape[2]
    assert position_ids.shape == (batch_size, 1), f"{position_ids.shape=}"
    assert attention_mask.shape == (batch_size, prev_seq_length+1), f"{attention_mask.shape=}, {prev_seq_length=}"

    if as_full_distribution:
        # next_input is a distribution over the vocabulary (batch_size, 1, vocab_size)
        all_embeds = model.model.embed_tokens.weight
        hidden_dim = all_embeds.shape[1]
        inputs_embeds = torch.matmul(next_input, all_embeds)
        assert inputs_embeds.shape == (batch_size, 1, hidden_dim)
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values)
    else:
        assert next_input.shape == (batch_size, 1)
        outputs = model(input_ids=next_input, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values)
    logits = outputs.logits
    past_key_values = outputs.past_key_values
    return logits, past_key_values

# @torch.no_grad()
def batch_generate_rnn(
        model,
        ref_model,
        questions_inputs,
        answers_inputs,
        max_steps=30,
        step_for_answer=20,
        temperature=1.0,
        top_k=None,
        as_full_distribution=False,
        dot_by_dot=False,
        dot_by_dot_id=None,
        inject_answer_prompt=False,
        answer_prompt_ids=None,
        loss_type="pg",
        compute_everything=False,
    ):
    metrics = {}
    assert not (as_full_distribution and dot_by_dot), f"{as_full_distribution=}, {dot_by_dot=}"
    device = model.device
    assert device.type == "cuda", f"{model.device=}"
    assert ref_model.device == device, f"{ref_model.device=}, {device=}"
    model.eval()
    ref_model.eval()
    batch_size = questions_inputs["input_ids"].shape[0]
    prompt_length = questions_inputs["input_ids"].shape[1]

    #### PROMPT FORWARD PASS
    position_ids = questions_inputs["attention_mask"].cumsum(dim=1) - 1
    outputs = model(**questions_inputs, position_ids=position_ids)
    with torch.no_grad():
        ref_outputs = ref_model(**questions_inputs)
    prompt_attention_mask = questions_inputs["attention_mask"]
    prompt_end_position_ids = position_ids[:, -1:] + 1
    past_key_values = outputs.past_key_values
    ref_past_keys_values = ref_outputs.past_key_values
    logits = outputs.logits[:, -1:]
    ref_logits = ref_outputs.logits[:, -1:]
    vocab_size = logits.shape[-1]

    def make_attention_mask(t, new_seq_length=1):
        return torch.cat([prompt_attention_mask, torch.ones((batch_size, t+new_seq_length), device=device)], dim=1)
    def make_position_ids(t, new_seq_length=1):
        if new_seq_length == 1:
            return prompt_end_position_ids + t
        else:
            return prompt_end_position_ids + t + torch.arange(new_seq_length).unsqueeze(0).to(device)

    all_gen_logits = torch.zeros((batch_size, 0, vocab_size), device=device)
    all_ref_logits = torch.zeros((batch_size, 0, vocab_size), device=device)
    generations = torch.zeros((batch_size, 0), device=device, dtype=torch.int)
    generations_without_injection = torch.zeros((batch_size, 0), device=device, dtype=torch.int)

    ### REASONING FORWARD PASSES
    for t in range(step_for_answer):
        next_token_ids, next_probdists = get_next(logits, temperature=temperature, top_k=top_k)
        all_gen_logits = torch.cat((all_gen_logits, logits), dim=1)
        all_ref_logits = torch.cat((all_ref_logits, ref_logits), dim=1)
        generations_without_injection = torch.cat((generations_without_injection, next_token_ids), dim=1)
        generations = torch.cat((generations, next_token_ids), dim=1)
        # forward pass of next token
        if as_full_distribution:
            next_input = next_probdists
        elif dot_by_dot:
            next_input = torch.full((batch_size, 1), dot_by_dot_id, dtype=torch.long, device=device)
        else:
            next_input = next_token_ids
        logits, past_key_values = single_step(model,
            next_input=next_input,
            attention_mask=make_attention_mask(t),
            position_ids=make_position_ids(t),
            past_key_values=past_key_values,
            as_full_distribution=as_full_distribution)
        with torch.no_grad():
            ref_logits, ref_past_keys_values = single_step(ref_model,
                next_input=next_input,
                attention_mask=make_attention_mask(t),
                position_ids=make_position_ids(t),
                past_key_values=ref_past_keys_values,
                as_full_distribution=as_full_distribution)
            
    ### INJECT ANSWER PROMPT FORWARD PASS
    t = step_for_answer
    if inject_answer_prompt:
        repeated_answer_prompt_ids = torch.tensor(answer_prompt_ids).unsqueeze(0).repeat(batch_size, 1).to(next_token_ids.dtype).to(device)
        generations = torch.cat((generations, repeated_answer_prompt_ids), dim=1)
        outputs = model(
            input_ids=repeated_answer_prompt_ids,
            attention_mask=make_attention_mask(step_for_answer, new_seq_length=len(answer_prompt_ids)),
            position_ids=make_position_ids(step_for_answer, new_seq_length=len(answer_prompt_ids)),
            past_key_values=past_key_values)
        with torch.no_grad():
            ref_outputs = ref_model(
                input_ids=repeated_answer_prompt_ids,
                attention_mask=make_attention_mask(t, new_seq_length=len(answer_prompt_ids)),
                position_ids=make_position_ids(t, new_seq_length=len(answer_prompt_ids)),
                past_key_values=ref_past_keys_values)
        logits = outputs.logits
        ref_logits = ref_outputs.logits
        past_key_values = outputs.past_key_values
        ref_past_keys_values = ref_outputs.past_key_values
        t += len(answer_prompt_ids)

    ### ANSWER FORWARD PASS
    if loss_type == "logp" or compute_everything:
        answer_length = answers_inputs["input_ids"].shape[1]
        answer_logits = model(
            input_ids=answers_inputs["input_ids"][:, :-1],
            attention_mask=make_attention_mask(t, new_seq_length=answer_length-1),
            position_ids=make_position_ids(t, new_seq_length=answer_length-1),
            past_key_values=past_key_values).logits
        with torch.no_grad():
            ref_answer_logits = ref_model(
                input_ids=answers_inputs["input_ids"][:, :-1],
                attention_mask=make_attention_mask(t+1, new_seq_length=answer_length-1),
                position_ids=make_position_ids(t+1, new_seq_length=answer_length-1),
                past_key_values=ref_past_keys_values).logits
        answer_logits = torch.cat((all_gen_logits[:, -1:], answer_logits), dim=1)
        ref_answer_logits = torch.cat((all_ref_logits[:, -1:], ref_answer_logits), dim=1)
        per_token_answer_logps = torch.gather(answer_logits, 2, answers_inputs["input_ids"].to(torch.long).unsqueeze(-1)).squeeze(-1)
        per_token_ref_answer_logps = torch.gather(ref_answer_logits, 2, answers_inputs["input_ids"].to(torch.long).unsqueeze(-1)).squeeze(-1)
        assert per_token_answer_logps.shape == answers_inputs["attention_mask"].shape, f"{per_token_answer_logps.shape=}, {answers_inputs['attention_mask'].shape=}"
        answer_logps = (per_token_answer_logps * answers_inputs["attention_mask"]).sum(dim=-1)
        ref_answer_logps = (per_token_ref_answer_logps * answers_inputs["attention_mask"]).sum(dim=-1)

        answer_perplexity = torch.exp(-answer_logps / answers_inputs["attention_mask"].sum(dim=-1))
        answer_perplexity_ref = torch.exp(-ref_answer_logps / answers_inputs["attention_mask"].sum(dim=-1))
        metrics["answer_logps"] = answer_logps.mean()
        metrics["answer_logps_ref"] = ref_answer_logps.mean()
        metrics["answer_logps_diff"] = (answer_logps - ref_answer_logps).mean()
        metrics["answer_perplexity"] = answer_perplexity.mean()
        metrics["answer_perplexity_ref"] = answer_perplexity_ref.mean()
        metrics["answer_perplexity_diff"] = (answer_perplexity - answer_perplexity_ref).mean()
    else:
        answer_logps = None
        ref_answer_logps = None

    ### CONTINUE FORWARD PASS STEPS
    if loss_type == "pg" or compute_everything:
        for t in range(t, max_steps):
            next_token_ids, next_probdists = get_next(logits[:, -1:], temperature=temperature, top_k=top_k)
            all_gen_logits = torch.cat((all_gen_logits, logits[:, -1:]), dim=1)
            all_ref_logits = torch.cat((all_ref_logits, ref_logits[:, -1:]), dim=1)
            generations_without_injection = torch.cat((generations_without_injection, next_token_ids), dim=1)
            generations = torch.cat((generations, next_token_ids), dim=1)
            # forward pass of next token
            if as_full_distribution:
                next_input = next_probdists
            elif dot_by_dot:
                next_input = torch.full((batch_size, 1), dot_by_dot_id, dtype=torch.long, device=device)
            else:
                next_input = next_token_ids
            logits, past_key_values = single_step(model,
                next_input=next_input,
                attention_mask=make_attention_mask(t),
                position_ids=make_position_ids(t),
                past_key_values=past_key_values,
                as_full_distribution=as_full_distribution)
            with torch.no_grad():
                ref_logits, ref_past_keys_values = single_step(ref_model,
                    next_input=next_input,
                    attention_mask=make_attention_mask(t),
                    position_ids=make_position_ids(t),
                    past_key_values=ref_past_keys_values,
                    as_full_distribution=as_full_distribution)
            
    next_token_ids, next_probdists = get_next(logits, temperature=temperature, top_k=top_k)
    all_gen_logits = torch.cat((all_gen_logits, logits), dim=1)
    all_ref_logits = torch.cat((all_ref_logits, ref_logits), dim=1)
    generations_without_injection = torch.cat((generations_without_injection, next_token_ids), dim=1)
    generations = torch.cat((generations, next_token_ids), dim=1)

    gen_per_token_logps = torch.gather(all_gen_logits, 2, generations_without_injection.to(torch.long).unsqueeze(-1)).squeeze(-1)
    ref_per_token_logps = torch.gather(all_ref_logits, 2, generations_without_injection.to(torch.long).unsqueeze(-1)).squeeze(-1)

    pd = torch.nn.functional.softmax(all_gen_logits, dim=-1)
    entropy = torch.logsumexp(all_gen_logits, dim=-1) - torch.sum(pd * all_gen_logits, dim=-1)
    entropy = entropy.mean(-1)
    metrics["logps"] = gen_per_token_logps.mean()
    metrics["logps_ref"] = ref_per_token_logps.mean()
    metrics["logps_diff"] = (gen_per_token_logps - ref_per_token_logps).mean()
    metrics["entropy"] = entropy.mean()
    metrics["entropy_std"] = entropy.std()

    x = (answer_logps, ref_answer_logps, gen_per_token_logps, ref_per_token_logps, entropy)
    
    if loss_type == "logp":
        assert x[0].requires_grad == True or x[0] is None, f"{x[0].requires_grad=}"
        assert x[1].requires_grad == False or x[1] is None, f"{x[1].requires_grad=}"
    assert x[2].requires_grad == True, f"{x[2].requires_grad=}"
    assert x[3].requires_grad == False, f"{x[3].requires_grad=}"
    assert x[4].requires_grad == True, f"{x[4].requires_grad=}"
    return x, generations, metrics, past_key_values


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

    print(f"{tokenizer.eos_token_id=}, {questions_inputs['input_ids'][:, 0]=}")

    

    def our_generate(num_steps_):
        x, generations, generation_metrics, past_key_values = batch_generate_rnn(
            model=model,
            ref_model=ref_model,
            questions_inputs=questions_inputs,
            answers_inputs=answers_inputs,
            max_steps=num_steps_,
            step_for_answer=20,
            temperature=0.0,
            top_k=None,
            as_full_distribution=False,
            dot_by_dot=False,
            dot_by_dot_id=None,
            inject_answer_prompt=False,
            answer_prompt_ids=None,
        )
        return generations, past_key_values
    
    def generate_with_generate(num_steps_):
        outputs = model.generate(
            input_ids=questions_inputs["input_ids"],
            attention_mask=questions_inputs["attention_mask"],
            max_new_tokens=num_steps_+1,
            temperature=0.0,
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
    # generate with .generate() method
    generations2, past_key_values2 = generate_with_generate(num_steps)
    print(f"\nREQUIRES GRAD .generate() method: {past_key_values2[0][0].requires_grad=}") # False
    # forward pass to get grads
    prompt_past_key_values = [(k[:, :, :prompt_length], v[:, :, :prompt_length]) for k, v in past_key_values1]
    past_key_values3 = forward_pass(generations1[:, :-1], prompt_past_key_values)
    print(f"\nREQUIRES GRAD forward pass: {past_key_values3[0][0].requires_grad=}") # True


    padding_mask = torch.ones((batch_size, prompt_length+num_steps), device=device)
    padding_mask[:, :prompt_length] = questions_inputs["attention_mask"]
    def mask_key_values(past_key_values):
        return [
            (key * padding_mask.unsqueeze(1).unsqueeze(-1), value * padding_mask.unsqueeze(1).unsqueeze(-1))
            for key, value in past_key_values
        ]
    past_key_values1 = mask_key_values(past_key_values1)
    past_key_values2 = mask_key_values(past_key_values2)
    past_key_values3 = mask_key_values(past_key_values3)

    # compare some keys and values
    for l, (kv1, kv2, kv3) in enumerate(zip(past_key_values1, past_key_values2, past_key_values3)):
        if l == len(past_key_values1) - 1:
            for t in range(kv1[0].shape[2]):
                if t < 3 or t > kv1[0].shape[2] - 3:
                    print(f"START {t=}")
                    print(f"{t=}: keys: {kv1[0].shape}, {kv2[0].shape}, {kv3[0].shape}")
                    print(f"{t=}: keys: {kv1[0][:, 0, t, :3].flatten().tolist()}")
                    print(f"{t=}: keys: {kv2[0][:, 0, t, :3].flatten().tolist()}")
                    print(f"{t=}: keys: {kv3[0][:, 0, t, :3].flatten().tolist()}")
                    print(f"{t=}: values: {kv1[1].shape}, {kv2[1].shape}, {kv3[1].shape}")
                    print(f"{t=}: values: {kv1[1][:, 0, t, :3].flatten().tolist()}")
                    print(f"{t=}: values: {kv2[1][:, 0, t, :3].flatten().tolist()}")
                    print(f"{t=}: values: {kv3[1][:, 0, t, :3].flatten().tolist()}\n")
                    print(f"END {t=}")
                    print(f"{t=}: keys: {kv1[0].shape}, {kv2[0].shape}, {kv3[0].shape}")
                    print(f"{t=}: keys: {kv1[0][:, 0, t, -3:].flatten().tolist()}")
                    print(f"{t=}: keys: {kv2[0][:, 0, t, -3:].flatten().tolist()}")
                    print(f"{t=}: keys: {kv3[0][:, 0, t, -3:].flatten().tolist()}")
                    print(f"{t=}: values: {kv1[1].shape}, {kv2[1].shape}, {kv3[1].shape}")
                    print(f"{t=}: values: {kv1[1][:, 0, t, -3:].flatten().tolist()}")
                    print(f"{t=}: values: {kv2[1][:, 0, t, -3:].flatten().tolist()}")
                    print(f"{t=}: values: {kv3[1][:, 0, t, -3:].flatten().tolist()}\n")

    # compare generations
    for i in range(batch_size):
        print(f"GENERATION {i}")
        print(tokenizer.decode(generations1[i]))
        print(tokenizer.decode(generations2[i]))
        print()

    # check equal
    for i in range(batch_size):
        key1 = past_key_values1[i][0]
        value1 = past_key_values1[i][1]
        key2 = past_key_values2[i][0]
        value2 = past_key_values2[i][1]
        key3 = past_key_values3[i][0]
        value3 = past_key_values3[i][1]
        assert key1.shape == key2.shape == key3.shape, f"{key1.shape=}, {key2.shape=}, {key3.shape=}"
        assert value1.shape == value2.shape == value3.shape, f"{value1.shape=}, {value2.shape=}, {value3.shape=}"
        print(f"Layer {i}: Keys: std1={key1.std().item():.4f}, std2={key2.std().item():.4f}, std3={key3.std().item():.4f}, max_diff 1 vs 2={(key1-key2).abs().max().item():.4f}, max_diff 1 vs 3={(key1-key3).abs().max().item():.4f}")
        print(f"Layer {i}: Values: std1={value1.std().item():.4f}, std2={value2.std().item():.4f}, std3={value3.std().item():.4f}, max_diff 1 vs 2={(value1-value2).abs().max().item():.4f}, max_diff 1 vs 3={(value1-value3).abs().max().item():.4f}")
        assert torch.allclose(key1, key2), f"{key1=}, {key2=}"
        assert torch.allclose(value1, value2), f"{value1=}, {value2=}"
        assert torch.allclose(key1, key3, atol=1e-5, rtol=1e-3), f"{key1=}, {key3=}"
        assert torch.allclose(value1, value3, atol=1e-5, rtol=1e-3), f"{value1=}, {value3=}"

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
    #         max_steps=num_steps,
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



    



    