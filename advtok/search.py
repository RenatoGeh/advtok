import math, random
import transformers, tqdm, torch, Levenshtein
import advtok.multi_rooted_mdd as mrmdd, advtok.mdd as mdd, advtok.utils as utils
import advtok.jailbreak as jailbreak

def greedy(model: transformers.AutoModelForCausalLM, tok: transformers.AutoTokenizer, S: str,
           k: int, sys_prompt: str, response: str, batch_size: int, X_0: list = None,
           max_neighbors: int = math.inf, frozen_prefix: str = [], frozen_suffix: str = [],
           return_ll: bool = False, only_dist2: bool = True, early_stop: bool = True, **kwargs) -> list:
    """Greedy local search for argmax_X p(sys_prompt, X, response) with X_0 as initial tokenization
    and k the number of iterations."""

    dd = mdd.build_mdd(tok, S, prefix_space=False)
    iter_rng = tqdm.tqdm(range(k), desc="Iteration")
    X_c = tok.encode(S, add_special_tokens=False)
    if isinstance(frozen_prefix, str):
        frozen_prefix = tok.encode(frozen_prefix, add_special_tokens=False)
    if isinstance(frozen_suffix, str):
        frozen_suffix = tok.encode(frozen_suffix, add_special_tokens=False)
    if (X_0 is None) or (X_0 == "canonical"): X_0 = X_c
    elif X_0 == "random": X_0 = dd.sample()
    X = X_0
    # Apply chat template and separate prompts from user request.
    sys_tok, user_head, user_tail, response_tok = jailbreak.template_chunks(sys_prompt, response, tok)
    prefix = torch.tensor(sys_tok + user_head + frozen_prefix, dtype=torch.int64)
    suffix = torch.tensor(frozen_suffix + user_tail + response_tok, dtype=torch.int64)
    best_ll, last_ll = -math.inf, -math.inf
    num_eq = 0
    len_affixes = prefix.numel() + suffix.numel()
    ne2_els = math.inf if math.isinf(max_neighbors) else int(math.floor(0.25*max_neighbors))
    neo_els = math.inf if math.isinf(max_neighbors) else int(math.ceil(0.75*max_neighbors))
    for i in iter_rng:
        mrdd = mrmdd.build_mrmdd(tok, X, 2 if only_dist2 else 4, prefix_space=False, reuse_mdd=dd)
        # Neighbors of X, i.e. {V : d(X, V)=2}, where d is edge distance.
        if only_dist2:
            Ne = mrdd.enumerate(2)
            l_ne = len(Ne)
            if len(Ne) > max_neighbors:
                random.shuffle(Ne)
                Ne = Ne[:max_neighbors]
        else:
            Ne = mrdd.enumerate(4) if math.isinf(neo_els) else mrdd.uniform(4, neo_els)
            Ne2 = mrdd.enumerate(2)
            l_ne = len(Ne2)+neo_els
            if len(Ne2) > ne2_els:
                random.shuffle(Ne2)
                Ne2 = Ne2[:ne2_els]
            Ne.extend(Ne2)
        Ne.append(X)
        # Compute log-likelihood of candidates.
        ll = utils.loglikelihood_anyfix(model, tok, Ne, prefix, suffix, batch_size,
                                        only_suffix=True, offset_suffix=len(frozen_suffix), **kwargs)
        # Normalize by tokenization length.
        #nll = ll.to("cpu")/torch.tensor(list(map(len, Ne)))
        nll = ll.to("cpu")-torch.log(torch.tensor(list(map(len, Ne)))+len_affixes)
        # Greedily pick highest candidate.
        j_best = torch.argmax(nll).item()
        X = Ne[j_best]
        best_ll = nll[j_best].item()
        iter_rng.set_description(f"Iteration {i} | #toks={l_ne} | d(X_c,X)={Levenshtein.distance(X_c, X)} | d(X_0,X)={Levenshtein.distance(X_0, X)} | loss={-best_ll:0.5f}")
        if early_stop and math.isclose(last_ll, best_ll, abs_tol=1e-5):
            num_eq += 1
            if num_eq > 3: break
        last_ll = best_ll
    if return_ll: return frozen_prefix + X + frozen_suffix, best_ll
    return frozen_prefix + X + frozen_suffix

def stochastic(model: transformers.AutoModelForCausalLM, tok: transformers.AutoTokenizer, S: str,
           k: int, sys_prompt: str, response: str, batch_size: int, X_0: list = None,
               max_neighbors: int = math.inf, frozen_prefix: str = [], return_ll: bool = False, **kwargs) -> list:
    """Stochastic local search for argmax_X p(sys_prompt, X, response) with X_0 as initial
    tokenization and k the number of iterations."""

    dd = mdd.build_mdd(tok, S, prefix_space=False)
    iter_rng = tqdm.tqdm(range(k), desc="Iteration")
    X_c = tok.encode(S, add_special_tokens=False)
    if (X_0 is None) or (X_0 == "canonical"): X_0 = X_c
    elif X_0 == "random": X_0 = dd.sample()
    X = X_0
    # Apply chat template and separate prompts from user request.
    sys_tok, user_head, user_tail, response_tok = jailbreak.template_chunks(sys_prompt, response, tok)
    prefix = torch.tensor(sys_tok + user_head + frozen_prefix, dtype=torch.int64)
    suffix = torch.tensor(user_tail + response_tok, dtype=torch.int64)
    len_affixes = prefix.numel() + suffix.numel()
    for i in iter_rng:
        mrdd = mrmdd.build_mrmdd(tok, X, 2, prefix_space=False, reuse_mdd=dd)
        # Neighbors of X, i.e. {V : d(X, V)=2}, where d is edge distance.
        Ne = mrdd.enumerate(2)
        if len(Ne) > max_neighbors:
            random.shuffle(Ne)
            Ne = Ne[:max_neighbors-1]
        Ne.append(X)
        # Compute log-likelihood of candidates.
        ll = utils.loglikelihood_anyfix(model, tok, Ne, prefix, suffix, batch_size,
                                        only_suffix=True, **kwargs)
        # Normalize by tokenization length.
        nll = ll.to("cpu")-torch.log(torch.tensor(list(map(len, Ne)))+len_affixes)
        # Sample candidates from the distribution induced by nll.
        j_best = torch.multinomial(torch.softmax(nll, dim=-1), 1, replacement=True)[0].item()
        X = Ne[j_best]
        best_ll = nll[j_best].item()
        iter_rng.set_description(f"Iteration {i} | #toks={len(Ne)} | d(X_c,X)={Levenshtein.distance(X_c, X)} | d(X_0,X)={Levenshtein.distance(X_0, X)} | loss={-best_ll:0.5f}")
    if return_ll: return X, best_ll
    return X
