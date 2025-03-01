import random, math, sys, heapq, multiprocessing, pickle, itertools, os, time
import torch, transformers, tqdm, scipy, Levenshtein, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import advtok.utils as utils

sys.setrecursionlimit(100_000)
torch.multiprocessing.set_start_method("spawn", force=True)

MAR_BATCH_LIST = [(256, 50000), (128, 50000), (64, 50000), (32, 60000), (16, 70000), (8, 80000), (4, 90000), (2, 100000)]

class MDD:
    "Multi-valued decision diagram."

    def __init__(self, var: int, ch: list = None, largest: int = 0, tokenizer: AutoTokenizer = None):
        "Argument `var` is the variable ID, and `ch` is a list of tuples `(token_id, child)`."
        self._state = None
        self._var = var
        self._ch = ch
        self._largest = largest
        self._tok = tokenizer

    def __str__(self, tok: AutoTokenizer = None):
        if tok is None: tok = self._tok
        return '{' + f"var={self._var}, ch=[" + (', '.join((t for t, _ in self._ch) if tok is None else
                                                           (f"{t} ('{tok.convert_ids_to_tokens(t)}')"
                                                            for t, _ in self._ch))) + "]}"

    def enumerate_it(self, prepend: list = None, eos_token: int = 2, bos_token: int = 1) -> list:
        "Performs model enumeration iteratively (slower compared to recursive)."
        if prepend is None: prepend = []
        L, L_c = [], []
        Pa = {self: (None, None)}
        Q = [(self, bos_token)]
        while len(Q) > 0:
            mdd, tok = Q.pop()
            L_c.append(tok)
            exp = False
            for t, c in mdd._ch:
                if t == eos_token or c is None:
                    L_c.append(t)
                    L.append(L_c.copy())
                    L_c.pop()
                else:
                    Q.append((c, t))
                    exp = True
            if not exp: L_c.pop()
        return L

    def enumerate(self, prepend: list = None, add_eos: bool = True, eos_token: int = None) -> list:
        "Performs model enumeration."
        if prepend is None: prepend = []
        if eos_token is None: eos_token = self._tok.eos_token_id
        L = []
        for t, c in self._ch:
            L_c = prepend.copy()
            if t != eos_token or add_eos: L_c.append(t)
            if c is None: L.append(L_c)
            else: L.extend(c.enumerate(L_c, add_eos=add_eos, eos_token=eos_token))
        return L

    def enumerate_gen(self, prepend: list = None, add_eos: bool = False, eos_token: int = None):
        "Performs model enumeration."
        if prepend is None: prepend = []
        if eos_token is None: eos_token = self._tok.eos_token_id
        for t, c in self._ch:
            L_c = prepend.copy()
            if t != eos_token or add_eos: L_c.append(t)
            if c is None: yield L_c.copy()
            else: yield from c.enumerate(L_c, add_eos=add_eos, eos_token=eos_token)

    def children(self, tokenizer: AutoTokenizer = None) -> list:
        "Returns the children as tokens."
        if tokenizer is None: tokenizer = self._tok
        return tokenizer.convert_ids_to_tokens([x[0] for x in self._ch])

    def size(self) -> int:
        "Returns the number of nodes of the MDD."
        memo = set()
        Q = [self]
        i = 0
        while len(Q) > 0:
            i += 1
            mdd = Q.pop(0)
            for _, c in mdd._ch:
                if c is not None and c not in memo:
                    Q.append(c)
                    memo.add(c)
        return i

    def count(self, memo: dict = None) -> int:
        "Performs model counting."
        if memo is None: memo = {}
        n = 0
        for _, c in self._ch:
            if c is None: n += 1
            else:
                if c in memo: n += memo[c]
                else:
                    k = c.count(memo=memo)
                    memo[c] = k
                    n += k
        return n

    def first(self, S: list = None) -> list:
        "Gets the first model of this MDD. 'First' is defined according to the vocabulary order."
        if S is None: S = []
        t, c = self._ch[0]
        S.append(t)
        if c is None: return S
        return c.first(S=S)

    def shortest(self, eos_token: int = None, add_eos: bool = True) -> list:
        """Gets the shortest model of this MDD. 'Shortest' is defined as the one with the least
        amount of non-end-of-sentence tokens or the one that has the fewer tokens."""
        if eos_token is None: eos_token = self._tok.eos_token_id
        memo = set()
        Q = [self]
        Pa = {self: (None, None)}
        while len(Q) > 0:
            mdd = Q.pop(0)
            for t, c in mdd._ch:
                if t == eos_token or c is None:
                    prev_t, pa = t, mdd
                    model = []
                    while pa is not None:
                        model.insert(0, prev_t)
                        prev_t, pa = Pa[pa]
                    return model if add_eos else model[:len(model)-(t == eos_token)]
                elif c not in memo:
                    Pa[c] = (t, mdd)
                    Q.append(c)
                    memo.add(c)
        # Something has gone terribly wrong.
        return None

    def greedy(self, eos_token: int = None, P: list = None, add_eos: bool = True) -> list:
        """Gets the greedy model of this MDD. 'Greedy' is defined as taking the largest token at
        each decision node."""
        if P is None: P = []
        if eos_token is None: eos_token = self._tok.eos_token_id
        t, c = self._ch[self._largest]
        if add_eos or (t != eos_token): P.append(t)
        if (c is None) or (t == eos_token): return P
        return c.greedy(eos_token=eos_token, P=P, add_eos=add_eos)

    def sample(self, eos_token: int = None, P: list = None, counts: dict = None) -> list:
        "Samples a tokenization from this diagram."
        if eos_token is None: eos_token = self._tok.eos_token_id
        if P is None: P = []
        if counts is None:
            counts = {}
            counts[self] = self.count(memo=counts)
        n = counts[self]
        t, c = random.choices(self._ch, weights=[(counts[k[1]] if k[1] is not None else 1)/n for k in self._ch])[0]
        if (c is None) or (t == eos_token): return P
        P.append(t)
        return c.sample(eos_token=eos_token, P=P, counts=counts)

    @torch.no_grad()
    def sample_lm(self, model: AutoModelForCausalLM, n: int, eos_token: int = None, prefix: list = None,
                  use_tqdm: bool = True, use_cache: bool = True, keep_bos: bool = True,
                  position: int = 0, bos_token: int = 1, **kwargs) -> tuple:
        if prefix is None: prefix = [bos_token]
        if eos_token is None: eos_token = self._tok.eos_token_id
        Q, LL = [None for _ in range(n)], np.zeros(n, dtype=np.float32)
        for j in (tqdm.tqdm(range(n), desc="Sampling from proposal", position=position, **kwargs)
                  if use_tqdm else range(n)):
            P, ll = [prefix.copy()], 0.0
            dd = self
            while dd is not None:
                T = [x[0] for x in dd._ch]
                logits = utils.logits_softmax(model, P, T, use_cache=use_cache)
                i = utils.sample_logspace(logits)
                t, c = dd._ch[i]
                ll += logits[i]
                P[0].append(t)
                if (c is None) or (t == eos_token): break
                dd = c
            Q[j], LL[j] = P[0][(not keep_bos)*len(prefix):-1], ll
        return Q, LL

    @torch.no_grad()
    def sample_lm_batch(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, n: int,
                        prefix: list = None, use_cache: bool = True, bos_token: int = 1) -> tuple:
        if prefix is None: prefix = [bos_token]
        prefix_len = len(prefix)
        X = torch.tensor(n*[prefix], dtype=torch.int32).to(model.device)
        # kv_cache = model(X, use_cache=True, return_dict=True)["past_key_values"]
        out = model.generate(X,
                             logits_processor=transformers.LogitsProcessorList([
                               MaskProcessor(self, tokenizer.pad_token_id, n)]),
                             output_scores=True, return_dict_in_generate=True, do_sample=True,
                             top_k=0, top_p=1, pad_token_id=tokenizer.pad_token_id, temperature=1.0,
                             use_cache=use_cache, max_new_tokens=512)
        scores = out.scores
        L = np.zeros(n, dtype=np.float32)
        S = [None for _ in range(n)]
        for i in range(n):
            for j in range(len(scores)):
                t = out.sequences[i,prefix_len+j]
                if t == tokenizer.pad_token_id or t == tokenizer.eos_token_id: break
                L[i] += scores[j][i,t]
            S[i] = out.sequences[i,prefix_len:prefix_len+j].tolist()
        return S, L

    @torch.no_grad()
    def mar_by_sample(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, n: int,
                      batch_size: int = 128, use_tqdm: bool = True, use_cache: bool = True,
                      position: int = 0, prefix: list = None, size_estimate: int = 0,
                      sample_batch_size: int = 0, bos_token: int = 1,
                      canonical: list = None, return_extra: bool = False, **kwargs) -> tuple:
        if prefix is None: prefix = [bos_token]
        # go_batch = size_estimate < 100
        #def detect_batch(x: int): return 32 if x < 76 else 16 if x < 110 else 8 if x < 170 else 4 if x < 260 else 2
        def detect_batch(x: int, i: int):
            if i >= len(MAR_BATCH_LIST): return 1
            b, l = MAR_BATCH_LIST[i]
            return b if x < l else detect_batch(x, i+1)
        #def detect_batch(x: int): return 32 if x < 100 else 16 if x < 150 else 8 if x < 180 else 4 if x < 380 else 1
        # if go_batch:
        SAMPLE_BATCH_SIZE = min(detect_batch(size_estimate, 0) if sample_batch_size <= 0 else
                                sample_batch_size, n)
        print("SAMPLE_BATCH_SIZE", SAMPLE_BATCH_SIZE, "size_estimate", size_estimate)
        k = n//SAMPLE_BATCH_SIZE
        include_canonical = canonical is not None
        S, LL_Q = [], np.zeros(n+include_canonical, dtype=np.float32)
        # if SAMPLE_BATCH_SIZE == 1:
            # S, LL_Q = self.sample_lm(model, n, prefix=prefix, keep_bos=False, position=position+1)
        # else:
        for i in (tqdm.tqdm(range(k+((n % SAMPLE_BATCH_SIZE) > 0)), desc="Sampling from proposal",
                            position=position+1, **kwargs) if use_tqdm else range(k)):
            n_samples = min((i+1)*SAMPLE_BATCH_SIZE, n)-i*SAMPLE_BATCH_SIZE
            s, ll_q = self.sample_lm_batch(model, tokenizer, n_samples, prefix=prefix,
                                           use_cache=use_cache)
            S.extend(s)
            LL_Q[i*SAMPLE_BATCH_SIZE:(i+1)*SAMPLE_BATCH_SIZE] = ll_q
        if canonical is not None:
            S.append(canonical)
            LL_Q[-1] = self.pr_from_lm(model, canonical, prefix=prefix) if utils.is_mamba_tokenizer(tokenizer) \
                       else self.pr_from_lm_batch(model, tokenizer, [canonical], batch_size=1, prefix=prefix)
        # else:
            # S, LL_Q = self.sample_lm(model, n, prefix=prefix, keep_bos=False, position=position+1)
        LL_P = utils.rhloglikelihood(model, tokenizer, S, batch_size=batch_size, prefix=prefix,
                                     prefix_cutoff=True, online_memory=True, use_tqdm=True)
        mar = scipy.special.logsumexp(LL_P-LL_Q)-np.log(n+include_canonical)
        if return_extra: return mar, S, LL_P, LL_Q, np.log(n+include_canonical)
        return mar, S

    @torch.no_grad()
    def pr_from_lm(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, T: list,
                   prefix: list = None, use_cache: bool = True, bos_token: int = None) -> float:
        if bos_token is None: bos_token = tokenizer.bos_token_id
        if prefix is None: prefix = [bos_token]
        N, i, ll, O, P = self, 0, 0.0, prefix + T, [None]
        while N is not None:
            P[0] = O[:len(prefix)+i]
            C = [x[0] for x in N._ch]
            j = C.index(T[i])
            logits = utils.logits_softmax(model, P, C, use_cache=use_cache)
            ll += logits[j]
            c = C[j]
            if (c is None) or (i+1 >= len(T)): break
            N = N._ch[j][1]
            i += 1
        return ll

    def forward_mask(self, T: list, out: torch.Tensor, zero: float = -1e16) -> torch.Tensor:
        out[:] = zero
        for j in range(len(T)):
            N, i = self, 0
            while N is not None:
                K = None
                for t, C in N._ch:
                    if C is None: break
                    out[j,i,t] = 0.
                    if t == T[j][i]: K = C
                N = K
                i += 1
        return out

    @torch.no_grad()
    def pr_from_lm_batch(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, T: list,
                         batch_size: int, prefix: list = None, pad_token_id: int = None) -> torch.Tensor:
        if pad_token_id is None: pad_token_id = tokenizer.pad_token_id
        l, m = max(map(len, T)), len(T)
        n = min(batch_size, m)
        B = torch.empty((n, l), dtype=torch.int32).to(model.device)
        P = torch.tensor(n*[prefix], dtype=torch.int32)
        mask = self.forward_mask(T, torch.empty((m, l, len(tokenizer.vocab)), dtype=torch.float32))
        _pre = model(P.to(model.device), use_cache=True, return_dict=True)
        pre, pre_logits = _pre["past_key_values"], _pre["logits"].to("cpu")
        L = torch.empty((n, l, len(tokenizer.vocab)), dtype=torch.float32)
        out = torch.empty(m, dtype=torch.float32)
        # Add the last logits in the prefix (i.e. the logits for the first token in T).
        L[:,0,:] = pre_logits[:,-1,:]
        # Tokenization in gather format for indexing.
        I = torch.tensor([t + (l-len(t))*[pad_token_id] for t in T], dtype=torch.int64).unsqueeze(-1)
        for i in range(0, m, batch_size):
            k = min(batch_size, m-i)
            for j in range(k):
                B[j,:len(T[i+j])] = torch.tensor(T[i+j], dtype=torch.int32)
                B[j,len(T[i+j]):] = pad_token_id
            # Logits up to the last token.
            L[:,0,:] = pre_logits[:,-1,:]
            L[:,1:,:] = model.forward(B, use_cache=True, past_key_values=pre)["logits"][:,:-1,:].to("cpu")
            # Mask zero probability tokens according to MDD.
            L[:k] += mask[i:i+batch_size]
            # Normalize over nonzero tokens.
            L[:] = torch.log_softmax(L, dim=-1)
            # Index logits and sum.
            J = torch.gather(L, -1, I[i:i+batch_size])
            for j, l in enumerate(map(len, T[i:i+batch_size])): J[j,l:,:] = 0.
            out[i:i+batch_size] = torch.sum(J, dim=1).flatten()
        return out

    def __lt__(self, other): return True
    def __le__(self, other): return True
    def __gt__(self, other): return True
    def __ge__(self, other): return True

    @torch.no_grad()
    def bb(self, tokenizer: AutoTokenizer, model: AutoModelForCausalLM, k: int = 10,
           eos_token: int = None, time_budget: float = math.inf,
           bound_tok: list = None) -> list:
        dd = self
        if eos_token is None: eos_token = tokenizer.eos_token_id
        if tokenizer.pad_token is None: tokenizer.pad_token = eos_token
        if bound_tok is None: bound_tok = self.shortest(add_eos=True)
        bound = utils.rhloglikelihood(model, tokenizer, [bound_tok + [tokenizer.eos_token_id]], batch_size=1).item()
        Q = [(-math.inf, self, [tokenizer.bos_token_id])]
        top_k = []
        mark = time.time()
        while (len(Q) > 0) and time_budget > 0:
            p, mdd, path = heapq.heappop(Q)
            P = [path + [t] for t, _ in mdd._ch]
            LL = utils.rhloglikelihood(model, tokenizer, P, batch_size=16)
            #print(bound, time_budget, min(map(len, P)), max(map(len, P)), LL)
            for ll, path_c, (t, c) in zip(LL, P, mdd._ch):
                if ll.item() < bound: continue
                if t == eos_token or c is None:
                    bound = ll.item()
                    if len(top_k) >= k: heapq.heapreplace(top_k, (ll, path))
                    else: heapq.heappush(top_k, (ll, path))
                else: heapq.heappush(Q, (-ll, c, path_c))
            time_budget -= time.time()-mark
            mark = time.time()
        return [(bound, bound_tok)] if len(top_k) == 0 else top_k, time_budget

    def depth(self, eos_token: int = 2, memo: dict = None, depth: int = 0) -> int:
        if memo is None: memo = {}
        d_max = depth+1
        for _, c in self._ch:
            if c is not None:
                if c in memo: d = max(memo[c], depth+1)
                else: d = c.depth(eos_token=eos_token, memo=memo, depth=depth+1)
                memo[c] = d
                if d > d_max: d_max = d
        return d_max

    @torch.no_grad()
    def top_by_enumeration(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                           prefix: list = None, suffix: list = None, batch_size: int = 10,
                           return_all: bool = False, use_perplexity: bool = False,
                           use_cache: bool = True) -> list:
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        T = self.enumerate(add_eos=False, eos_token=tokenizer.eos_token_id)
        if len(T) < 2: return T[0], None
        LL = np.zeros(len(T))
        f = utils.perplexity_batch if use_perplexity else utils.loglikelihood_batch
        if prefix is None: T_eval = T
        elif suffix is None: T_eval = utils.parallelize(utils.partial(_list_prepend, prefix), T)
        else: T_eval = utils.parallelize(utils.partial(_list_surround, prefix, suffix), T)
        for i in range(0, len(T_eval), batch_size):
            LL[i:i+batch_size] = f(model, tokenizer, T_eval[i:i+batch_size], use_cache=use_cache)
        im = np.argmax(LL)
        return (im, T[im], T, LL) if return_all else T[im], LL[im]

    @staticmethod
    def _get_mask_f(M, X: torch.LongTensor, mask: torch.LongTensor, prefix_len: int, i: int,
                    pad_token: int = 2):
        N = M
        j = prefix_len
        while (N is not None) and (j < X.shape[1]):
            for t, C in N._ch:
                if X[i,j] == t:
                    N = C
                    j += 1
                    break
        mask[i,pad_token if N is None else [t for t, _ in N._ch]] = 0.0

    def get_mask(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, prefix_len: int,
                 parallel: bool = False, pad_token: int = 2, **kwargs) -> torch.FloatTensor:
        if not parallel: return self.get_mask_seq(input_ids, scores, prefix_len, **kwargs)
        mask = torch.full(scores.shape, -1e10, dtype=torch.float32)
        utils.parallelize(utils.partial(MDD._get_mask_f, self, input_ids, mask, prefix_len,
                                        pad_token=pad_token, **kwargs), range(input_ids.shape[0]))
        return mask

    def get_mask_seq(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, prefix_len: int,
                     pad_token: int = 2) -> torch.FloatTensor:
        mask = torch.full(scores.shape, -1e10, dtype=torch.float32)
        for i in range(input_ids.shape[0]):
            N = self
            j = prefix_len
            while (N is not None) and (j < input_ids.shape[1]):
                for t, C in N._ch:
                    if input_ids[i,j] == t:
                        N = C
                        j += 1
                        break
            mask[i,pad_token if N is None else [t for t, _ in N._ch]] = 0.0
        return mask

    def dot(self, path: str = None, tokenizer: AutoTokenizer = None, label: bool = True):
        text = "strict digraph {\n"""
        Q, V, i = [self], {self: "n0"}, 1
        if tokenizer is not None: self._tok = tokenizer
        f = lambda t: t if self._tok is None else self._tok.convert_ids_to_tokens(t)
        if label: text += f"  n0 [label=\"\"];\n"
        while len(Q) > 0:
            N = Q.pop()
            ne = []
            for t, C in N._ch:
                if C is None: continue
                if C not in V:
                    if label: text += f"  n{i} [label=\"({C._var[0]}, {f(C._var[1])})\"];\n"
                    V[C] = f"n{i}"
                    i += 1
                    Q.append(C)
                ne.append((f(t), V[C]))
            for e, n in ne:
                text += f"  {V[N]} -> {n}[label=\"{e}\"];\n"
        text += "}\n"
        if path is not None:
            with open(path, "w") as file: file.write(text)
        return text

    def dfa(self, tokenizer: AutoTokenizer = None, add_dead_state: bool = False):
        if tokenizer is not None: self._tok = tokenizer
        accept_states = []
        dead_state, initial_state = 0, 1
        edges = []
        dead_tokens = lambda T: list(itertools.filterfalse(T.__contains__, range(self._tok.vocab_size)))
        Q, V, i = [self], {self: 1}, 2
        while len(Q) > 0:
            N = Q.pop()
            for t, C in N._ch:
                if C is None:
                    accept_states.append(V[N])
                    continue
                if C not in V:
                    V[C] = i
                    i += 1
                    Q.append(C)
                edges.append((V[N], V[C], [t]))
            if add_dead_state: edges.append((V[N], dead_state, dead_tokens([t for t, _ in N._ch])))
        return {"accept_states": accept_states, "initial_state": 1, "edges": edges}

    @staticmethod
    def process_mask(P, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.Tensor:
        if P.mask is None: P.mask = torch.full(scores.shape, -1e10, dtype=torch.float32).to(scores.device)
        else:
            P.mask[:] = -1e10
            # Get next state.
            for i, m in enumerate(P.M):
                if m is None: continue
                for t, C in m._ch:
                    if t == input_ids[i,-1].item():
                        P.M[i] = C
                        break
        for i in range(len(P.M)):
            if P.M[i] is None: P.mask[i,P.pad_token] = 0.0
            else:
                for t, _ in P.M[i]._ch: P.mask[i,t] = 0.0
        # mask = self.M.get_mask(input_ids, scores, self.prefix_len, pad_token=self.pad_token)
        return torch.log_softmax(P.mask+scores, dim=-1)

class MaskProcessor(transformers.LogitsProcessor):
    def __init__(self, M: MDD, pad_token: int, batch_size: int):
        self.process_mask_f = M.process_mask
        self.M = [M for _ in range(batch_size)]
        self.mask = None
        self.pad_token = pad_token
        self.prev_logits = None
        self.logits_path = None
        self.Z = None if type(M) is MDD else M.partition(return_memo=True)
        self.last_dist = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # input_ids: num_batch x len_prefix
        # scores:    num_batch x vocab_size
        return self.process_mask_f(self, input_ids, scores)

def _list_prepend(prefix: list, t: list) -> list: return prefix+t
def _list_surround(prefix: list, suffix: list, t: list) -> list: return prefix+t+suffix

def tstate(S: str, token: int, last_token: int, level: int, tokenizer: AutoTokenizer) -> tuple:
    return token, len(S)
def lstate(S: str, token: int, last_token: int, level: int, tokenizer: AutoTokenizer) -> tuple:
    return token, len(S)-len(tokenizer.convert_ids_to_tokens(token))
def ostate(S: str, token: int, last_token: int, level: int, tokenizer: AutoTokenizer) -> tuple:
    return level, token, len(S)
def sstate(S: str, *args) -> tuple: return len(S)

def build_from_fixed(S: str, token: int, vocab: dict, level: int, max_level: int,
                     eos_token: int, unk_token: int, memo: dict, tokenizer: AutoTokenizer,
                     last_token: int, fstate, o_len: int):
    k = fstate(S, token, last_token, level, tokenizer)
    if k in memo: return token, memo[k]
    mdd = build(S[1:], vocab, max_level, tokenizer, token, fstate, level=level+1,
                eos_token=eos_token, unk_token=unk_token, memo=memo, o_len=o_len)
    memo[k] = mdd
    return token, mdd

def build_from_newline(S: str, vocab: dict, level: int, max_level: int, eos_token: int,
                       unk_token: int, memo: dict, tokenizer: AutoTokenizer,
                       last_token: int, fstate, o_len: int):
    "Newline: [29871, 13]."
    k = fstate(S, 29871, last_token, level, tokenizer)
    if k in memo: return 29871, memo[k]
    c = build(S[1:], vocab, max_level, tokenizer, 13, fstate, level=level+2, eos_token=eos_token,
              unk_token=unk_token, memo=memo, o_len=o_len)
    mdd = MDD((o_len-len(S), last_token), [(13, c)])
    memo[k] = mdd
    return 29871, mdd

def build_from_unicode(S: str, vocab: dict, level: int, max_level: int, eos_token: int,
                       unk_token: int, memo: dict, tokenizer: AutoTokenizer,
                       last_token: int, fstate, o_len: int):
    C = tokenizer.encode(S[0], add_special_tokens=False)[1:]
    K = [fstate(S, c, last_token if i == 0 else C[i-1], level, tokenizer) for i, c in enumerate(C)]
    if K[0] in memo: return K[0][0], memo[K[0]]
    ch = build(S[1:], vocab, max_level, tokenizer, C[-1], fstate, level=level+len(C),
               eos_token=eos_token, unk_token=unk_token, memo=memo, o_len=o_len)
    def f(i, X): return ch if i >= len(C) else MDD(level+i, [(X[i], f(i+1, X))])
    mdd = f(1, C)
    memo[K[0]] = mdd
    return C[0], mdd

def build(S: str, vocab: dict, max_level: int, tokenizer: AutoTokenizer, last_token: int, fstate,
          level: int = 1, eos_token: int = None, unk_token: int = 0, memo: dict = None, o_len: int = None) -> MDD:
    if memo is None: memo = {}
    if o_len is None: o_len = len(S)
    if eos_token is None: eos_token = tokenizer.eos_token_id
    if level > max_level: return None
    state = (o_len-len(S), last_token)
    if len(S) == 0:
        # If we want to make the circuit smooth, then uncomment this line.
        #ch = build(S, vocab, max_level, level+1, eos_token=eos_token, unk_token=unk_token, memo=memo)
        ch = None
        return MDD(state, [(eos_token, ch)])
    ch = []
    ls, li, j = 0, 0, -1
    if S.startswith("🩏"):
        ch.append(build_from_fixed(S, unk_token, vocab, level, max_level, eos_token, unk_token,
                                   memo, tokenizer, last_token, fstate, o_len))
        return MDD(state, ch, largest=0)
    if S.startswith("\n") and utils.is_llama_tokenizer(tokenizer):
        ch.append(build_from_fixed(S, 13, vocab, level, max_level, eos_token, unk_token, memo,
                                   tokenizer, last_token, fstate, o_len))
        return MDD(state, ch, largest=0)
    for v in vocab:
        if S.startswith(v):
            i = vocab[v]
            k = fstate(S, i, last_token, level, tokenizer)
            j += 1
            if (v_len := len(v)) > ls: ls, li = v_len, j
            if k in memo: ch.append((i, memo[k]))
            else:
                mdd = build(S[len(v):], vocab, max_level, tokenizer, i, fstate, level=level+1,
                            eos_token=eos_token, unk_token=unk_token, memo=memo, o_len=o_len)
                memo[k] = mdd
                ch.append((i, mdd))
    if j < 0:
        ch.append(build_from_unicode(S, vocab, level, max_level, eos_token, unk_token, memo,
                                     tokenizer, last_token, fstate, o_len))
    return MDD(state, ch, largest=li)

def _no_vocab_cache(tokenizer: AutoTokenizer = None) -> bool:
    return (not hasattr(build_from_tokenizer, "V_cache")) or (build_from_tokenizer.V_cache is None) \
        or ((tokenizer is not None) and (utils._tok_name(tokenizer) not in build_from_tokenizer.V_cache))

def _llama_vocab_const(tokenizer: AutoTokenizer, k: str) -> tuple:
    return k.replace('▁', ' '), tokenizer.get_vocab()[k]
def _vocab_const(tokenizer: AutoTokenizer, k: str) -> tuple:
    return (' ' if k.startswith('▁') else '') + tokenizer.convert_tokens_to_string([k]), tokenizer.get_vocab()[k]
def _gemma_vocab_const(tokenizer: AutoTokenizer, k: str) -> tuple:
    return tokenizer.convert_tokens_to_string([k]), tokenizer.get_vocab()[k]
def _gemma2_vocab_const(tokenizer: AutoTokenizer, k: str) -> tuple:
    a = tokenizer.convert_tokens_to_string([k])
    return a, tokenizer.encode(a, add_special_tokens=False)[0]
def _llama3_vocab_const(tokenizer: AutoTokenizer, k: str) -> tuple:
    return tokenizer.convert_tokens_to_string([k]), tokenizer.get_vocab()[k]
def _select_vocab_const(tokenizer: AutoTokenizer):
    if utils.is_llama3_tokenizer(tokenizer): return utils.partial(_llama3_vocab_const, tokenizer)
    elif utils.is_llama_tokenizer(tokenizer): return utils.partial(_llama_vocab_const, tokenizer)
    elif utils.is_gemma2_tokenizer(tokenizer) or utils.is_shieldgemma_tokenizer(tokenizer):
        return utils.partial(_gemma2_vocab_const, tokenizer)
    elif utils.is_gemma_tokenizer(tokenizer): return utils.partial(_gemma_vocab_const, tokenizer)
    return utils.partial(_vocab_const, tokenizer)

def cache_vocab(tokenizer: AutoTokenizer, cache_file: str = None, cache: dict = None,
                persistent: bool = True):
    if cache_file is None: cache_file = f"{utils._tok_name(tokenizer)}_vocab_cache.pkl"
    exists = os.path.isfile(cache_file)
    if exists:
        with open(cache_file, "rb") as f: V = pickle.load(f)
    elif cache is not None: V = cache
    else:
        f_const = _select_vocab_const(tokenizer)
        V = dict(utils.parallelize(f_const, tokenizer.get_vocab().keys(),
                                   desc="Constructing vocabulary"))
        #V = dict(map(f_const, tokenizer.get_vocab().keys()))
        if utils.is_llama_tokenizer(tokenizer) and ('\n' not in V): V['\n'] = 13
        # Make sure there is no empty string in the vocabulary.
        V.pop('', None)
    if _no_vocab_cache(): build_from_tokenizer.V_cache = {utils._tok_name(tokenizer): V}
    else: build_from_tokenizer.V_cache[utils._tok_name(tokenizer)] = V
    if (not exists) and persistent:
        with open(cache_file, "wb") as f: pickle.dump(V, f)

def build_from_tokenizer(S: str, tokenizer: AutoTokenizer, max_level: int = math.inf,
                         fstate = tstate, cached = None, prefix_space: bool = None) -> MDD:
    if _no_vocab_cache(tokenizer): cache_vocab(tokenizer, cache=cached)
    if prefix_space is None: prefix_space = utils.is_llama_tokenizer(tokenizer)
    M = build(" " + S if prefix_space else S, build_from_tokenizer.V_cache[utils._tok_name(tokenizer)],
              max_level, tokenizer, None, fstate, eos_token=tokenizer.eos_token_id,
              unk_token=tokenizer.unk_token_id, memo={})
    M._tok = tokenizer
    return M

def build_mdd(tokenizer: AutoTokenizer, text: str, **kwargs):
    return build_from_tokenizer(text, tokenizer, **kwargs)

def build_mdd_parallel(tokenizer: AutoTokenizer, T: list, tqdm_kwargs: dict = {}, **kwargs):
    bound_build_mdd = utils.partial(build_mdd, tokenizer, **kwargs)
    if _no_vocab_cache(tokenizer): cache_vocab(tokenizer)
    return utils.parallelize(bound_build_mdd, T, **tqdm_kwargs)

def _build_mdd_parallel_batch(tokenizer: AutoTokenizer, T: list, prefix_space: list, **kwargs):
    return [build_mdd(tokenizer, T[i], prefix_space=prefix_space[i], **kwargs) for i in range(len(T))]
def build_mdd_parallel_pargs(tokenizer: AutoTokenizer, T: list, prefix_space: list = None, **kwargs):
    if _no_vocab_cache(tokenizer): cache_vocab(tokenizer)
    n = multiprocessing.cpu_count()
    if prefix_space is None: prefix_space = np.ones(len(T), dtype=np.bool_)
    with multiprocessing.Pool(n) as pool:
        S = utils.batch_slices(n, len(T))
        R = [pool.apply_async(_build_mdd_parallel_batch, (tokenizer, T[s], prefix_space[s]), kwargs) for s in S]
        M = [x for r in tqdm.tqdm(R, desc="Building MDDs") for x in r.get()]
    return M

def build_blob(tokenizer: AutoTokenizer, S: str, threshold: int = -1, **kwargs) -> list:
    if threshold < 1:
        # Automatically find a good threshold.
        threshold = len(S)//multiprocessing.cpu_count()
    I, n, m = [0], len(S)//threshold, len(S)
    for i in range(1, n):
        j = S.find(' ', threshold*i)
        if j == m: break
        I.append(j)
    I.append(len(S))
    Z = [S[I[i-1]+(i != 1):I[i]] for i in range(1, len(I))]
    return build_mdd_parallel(tokenizer, Z, **kwargs)

def blob_f(M: list, f: str, **kwargs) -> list:
    return list(itertools.chain.from_iterable(getattr(m, f)(**kwargs) for m in M))
