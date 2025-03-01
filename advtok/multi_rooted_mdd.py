import torch, transformers, tqdm, scipy, Levenshtein, functools, numpy as np, graphviz
from typing import Optional, Dict, Set, Tuple, List
from dataclasses import dataclass
from collections import deque
import random, math, sys, heapq, multiprocessing, pickle, itertools, os, time, csv
from transformers import AutoTokenizer, AutoModelForCausalLM
import advtok.mdd as mdd, advtok.utils as utils

def _token_toxicity(toxicity_model: dict, token_id: int) -> float:
    if token_id is None: return 0.0
    return toxicity_model['coeffs'].get(token_id, 0.0)

def _compile_memo(V: list, tok: transformers.AutoTokenizer) -> set:
    assert len(V) > 0, "Origin tokenization must be non-empty."
    M = set()
    T = [tok.convert_tokens_to_string([x]) for x in tok.convert_ids_to_tokens(V)]
    M.add((0, None))
    p = 0
    for x, t in zip(V, T):
        p += len(t)
        M.add((p, x))
    return M

class MultiRootedMDD:
    def __init__(self, mdd: mdd.MDD, K: int, canonical_tokens: list, text, **kwargs):
        self.K = K
        self.canonical_tokens = canonical_tokens
        self.tokenizer = mdd._tok
        self.nodes = {}  # (var, column) -> NodeInfo
        self.text = text
        # Dictionary mapping var -> MDD node
        self.var_to_mdd_map = {}
        #self._compute_token_boundaries()
        self.toxicity_info = None
        self.origin_memo = _compile_memo(canonical_tokens, self.tokenizer)
        self._build_from_mdd(mdd, **kwargs)

    def load_toxicity_model(self, data_dir: str):
        """
        Load the toxicity linear regression model from three CSV files located in data_dir:
        - count_importanceWeighted_l10_toxic_coefficients.csv
        - count_importanceWeighted_l10_toxic_intercept.csv
        - count_importanceWeighted_l10_toxic_vocabulary.csv
        """
        coeff_path = os.path.join(data_dir, "count_importanceWeighted_l10_toxic_coefficients.csv")
        intercept_path = os.path.join(data_dir, "count_importanceWeighted_l10_toxic_intercept.csv")
        vocab_path = os.path.join(data_dir, "count_importanceWeighted_l10_toxic_vocabulary.csv")

        # Read intercept
        with open(intercept_path, 'r', newline='', encoding='utf-8') as f:
            intercept_line = f.read().strip()
            intercept_value = float(intercept_line)

        # Read vocabulary
        vocab = set()
        with open(vocab_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)  # skip header if present
            for row in reader:
                if row:
                    token_id = int(row[0])
                    vocab.add(token_id)

        # Read coefficients
        coeffs = {}
        with open(coeff_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)  # skip "Token ID,Coefficient"
            for row in reader:
                if len(row) == 2:
                    token_id_str, coeff_str = row
                    token_id = int(token_id_str)
                    if token_id in vocab:
                        coeffs[token_id] = float(coeff_str)

        # Assign 0 to tokens in vocab not in coeffs
        for token_id in vocab:
            if token_id not in coeffs:
                coeffs[token_id] = 0.0

        # If a token is not in vocab at all, we will treat it as 0 later by using .get(token_id, 0.0)

        self.toxicity_model = {
            'intercept': intercept_value,
            'coeffs': coeffs
        }

    def enumerate(self, col: int, **kwargs) -> list:
        return self.model_enumeration(((0, None), col), **kwargs)

    def _compute_token_boundaries(self):
        """
        Compute the character positions in 'self.text' where each canonical token starts.
        This must be done before building columns so that _is_canonical_edge can use it.
        """
        f_const = mdd._select_vocab_const(self.tokenizer)
        self.token_boundaries = []
        current_char = 0
        for t_id in self.canonical_tokens:
            token_str = self.tokenizer.convert_ids_to_tokens([t_id])[0]
            # Handle any prefix or special chars if needed. For example, if using a SentencePiece model:
            # token_str = token_str.replace('▁', ' ')
            substring = self.text[current_char:current_char+len(token_str)]
            assert substring == f_const(token_str)[0], f"Canonical token '{token_str}' does not match text substring '{substring}' at {current_char}"
            self.token_boundaries.append(current_char)
            current_char += len(token_str)

    class NodeInfo:
        def __init__(self, var: tuple, column: int):
            self.var = var  # Variable tuple (position, last_token)
            self.column = column  # Distance from canonical path (0 to K)
            self.outgoing = []  # List of (token, target_var, target_column)
            self.incoming = []  # List of (token, source_var, source_column)

        def add_outgoing(self, token: int, target_var: tuple, target_column: int):
            edge = (token, target_var, target_column)
            if edge not in self.outgoing:
                self.outgoing.append(edge)

        def add_incoming(self, token: int, source_var: tuple, source_column: int):
            edge = (token, source_var, source_column)
            if edge not in self.incoming:
                self.incoming.append(edge)

    def _is_canonical_edge(self, var: tuple) -> bool: return var in self.origin_memo

    def _ensure_node_exists(self, var: tuple, column: int) -> NodeInfo:
        key = (var, column)
        if key not in self.nodes:
            self.nodes[key] = self.NodeInfo(var, column)
        return self.nodes[key]

    def _build_from_mdd(self, MDD_root: mdd.MDD, **kwargs):
        self._build_column_zero(MDD_root, **kwargs)
        for col in range(1, self.K + 1):
            self._build_column(MDD_root, col)

    def _build_column_zero(self, MDD_root: mdd.MDD):
        """Build column 0 with only canonical edges."""
        # First pass: create all nodes
        queue = [(MDD_root, None)]
        visited = set()

        while queue:
            current_mdd, parent_var = queue.pop(0)
            if current_mdd is None: continue

            current_var = current_mdd._var
            if current_var in visited: continue
            visited.add(current_var)

            # Create node in column 0
            self._ensure_node_exists(current_var, 0)
            self.var_to_mdd_map[current_var] = current_mdd

            # Add all nodes to queue
            for token, next_mdd in current_mdd._ch:
                if next_mdd is not None: queue.append((next_mdd, current_var))

        # Second pass: add only canonical edges
        queue = [(MDD_root, None)]
        visited = set()

        while queue:
            current_mdd, parent_var = queue.pop(0)
            if current_mdd is None: continue

            current_var = current_mdd._var
            if current_var in visited: continue
            visited.add(current_var)

            position, _ = current_var
            # If at end, add terminal edge immediately
            if position == len(self.text):
                self._add_edge(current_var, 0, None, None, 0)

            for token, next_mdd in current_mdd._ch:
                if next_mdd is not None:
                    if self._is_canonical_edge(next_mdd._var):
                        self._add_edge(current_var, 0, token, next_mdd._var, 0)
                    queue.append((next_mdd, current_var))

    def _build_column(self, MDD_root: mdd.MDD, col: int):
        """Build column col (> 0) with both canonical and non-canonical edges."""
        # First pass: create all nodes for this column
        queue = [(MDD_root, None)]
        visited = set()

        while queue:
            current_mdd, parent_var = queue.pop(0)
            if current_mdd is None:
                continue

            current_var = current_mdd._var
            if current_var in visited:
                continue
            visited.add(current_var)

            self._ensure_node_exists(current_var, col)
            self.var_to_mdd_map[current_var] = current_mdd

            # Add all nodes to queue
            for token, next_mdd in current_mdd._ch:
                if next_mdd is not None:
                    queue.append((next_mdd, current_var))

        # Second pass: add edges from the nodes in this column
        column_nodes = [(var, col) for (var, c) in self.nodes.keys() if c == col]
        # For each var in this column, get the original MDD node
        queue = []
        visited = set()

        for (var, c) in column_nodes:
            original_mdd_node = self.var_to_mdd_map.get(var, None)
            if original_mdd_node is not None:
                queue.append((original_mdd_node, None))

        while queue:
            current_mdd, parent_var = queue.pop(0)
            if current_mdd is None:
                continue

            current_var = current_mdd._var
            key = (current_var, col)
            if key in visited:
                continue
            visited.add(key)

            position, _ = current_var
            if position == len(self.text):
                # Connect to terminal without token
                self._add_edge(current_var, col, None, None, col)

            # Copy all edges from original MDD
            for token, next_mdd in current_mdd._ch:
                if next_mdd is not None:
                    is_canonical = self._is_canonical_edge(next_mdd._var)
                    if is_canonical:
                        # Canonical edge stays in same column
                        self._add_edge(current_var, col, token, next_mdd._var, col)
                    else:
                        # Non-canonical edge goes one column closer to 0
                        self._add_edge(current_var, col, token, next_mdd._var, col - 1)
                    queue.append((next_mdd, current_var))

    def _add_edge(self, from_var: tuple, from_col: int, token: int,
                  to_var: Optional[tuple], to_col: int):
        from_node = self._ensure_node_exists(from_var, from_col)
        if to_var is not None:
            self._ensure_node_exists(to_var, to_col)

        from_node.add_outgoing(token, to_var, to_col)
        if to_var is not None:
            self.nodes[(to_var, to_col)].add_incoming(token, from_var, from_col)

    def prune(self):
        terminal_node = (None, 0)
        root_nodes = [((0, None), col) for col in range(self.K + 1) if ((0, None), col) in self.nodes]

        # Forward search from all roots to find root_reachable nodes
        root_reachable = set()
        queue = root_nodes[:]
        while queue:
            node_key = queue.pop()
            if node_key in root_reachable:
                continue
            root_reachable.add(node_key)

            var, col = node_key
            # Terminal node check
            if var is None:
                # terminal node reached from roots
                continue

            current_node = self.nodes[node_key]
            for token, target_var, target_col in current_node.outgoing:
                if target_var is None:
                    # Terminal edge
                    t_key = (None, target_col)
                    if t_key not in root_reachable:
                        queue.append(t_key)
                else:
                    next_key = (target_var, target_col)
                    if next_key not in root_reachable:
                        queue.append(next_key)

        # Backward search from terminal in column 0 to find terminal_reachable nodes
        terminal_reachable = set()
        queue = [terminal_node]
        while queue:
            node_key = queue.pop()
            if node_key in terminal_reachable:
                continue
            terminal_reachable.add(node_key)

            var, col = node_key
            if var is None:
                # From terminal_0, find all nodes leading to it
                for (node_var, node_col), node_data in self.nodes.items():
                    for token, target_var, target_col in node_data.outgoing:
                        if target_var is None and target_col == col:
                            prev_key = (node_var, node_col)
                            if prev_key not in terminal_reachable:
                                queue.append(prev_key)
            else:
                # Regular node: follow incoming edges backwards
                current_node = self.nodes[node_key]
                for token, source_var, source_col in current_node.incoming:
                    if source_var is None:
                        term_key = (None, source_col)
                        if term_key not in terminal_reachable:
                            queue.append(term_key)
                    else:
                        prev_key = (source_var, source_col)
                        if prev_key not in terminal_reachable:
                            queue.append(prev_key)

        # Valid nodes are those that are both root_reachable and terminal_reachable
        valid_nodes = root_reachable.intersection(terminal_reachable)

        # Prune nodes and edges
        new_nodes = {}
        for node_key, node_data in self.nodes.items():
            if node_key in valid_nodes:
                new_outgoing = []
                for token, target_var, target_col in node_data.outgoing:
                    if target_var is None:
                        # Terminal edge: keep if the terminal node is valid
                        if (None, target_col) in valid_nodes:
                            new_outgoing.append((token, target_var, target_col))
                    else:
                        target_key = (target_var, target_col)
                        if target_key in valid_nodes:
                            new_outgoing.append((token, target_var, target_col))

                new_incoming = []
                for token, source_var, source_col in node_data.incoming:
                    if source_var is None:
                        # Terminal incoming: keep if terminal node is valid
                        if (None, source_col) in valid_nodes:
                            new_incoming.append((token, source_var, source_col))
                    else:
                        source_key = (source_var, source_col)
                        if source_key in valid_nodes:
                            new_incoming.append((token, source_var, source_col))

                node_data.outgoing = new_outgoing
                node_data.incoming = new_incoming
                new_nodes[node_key] = node_data

        self.nodes = new_nodes
        return self

    def visualize(self, filename: str = "multi_rooted_mdd", view: bool = False):
        dot = graphviz.Digraph(comment="Multi-rooted MDD")
        dot.attr(rankdir="LR")

        # Create clusters for each column
        for col in range(self.K + 1):
            with dot.subgraph(name=f"cluster_{col}") as c:
                c.attr(label=f"Distance {col}")

                # Add regular nodes
                for (var, column), node in self.nodes.items():
                    if column == col:
                        position, last_token = var
                        node_id = f"{position}_{last_token}_{column}"
                        if last_token is not None:
                            last_token_str = self.tokenizer.convert_ids_to_tokens([last_token])[0]
                            label = f"pos={position}\nlast={last_token_str}"
                        else:
                            label = f"pos={position}\nlast=None"
                        c.node(node_id, label, shape="box")

                # Add terminal node for this column
                terminal_id = f"terminal_{col}"
                c.node(terminal_id, "Terminal", shape="doublecircle")

        # Add edges
        edges_added = set()
        for (var, column), node in self.nodes.items():
            position, last_token = var
            src_id = f"{position}_{last_token}_{column}"

            for token, target_var, target_col in node.outgoing:
                edge_key = (src_id, target_var, target_col, token)
                if edge_key not in edges_added:
                    if target_var is None:
                        # Terminal edge
                        target_id = f"terminal_{target_col}"
                        style = "solid"
                        color = "black"
                        dot.edge(src_id, target_id, color=color, style=style)
                    else:
                        # Regular edge
                        token_str = self.tokenizer.convert_ids_to_tokens([token])[0]
                        target_pos, target_last = target_var
                        target_id = f"{target_pos}_{target_last}_{target_col}"

                        is_canonical = self._is_canonical_edge(target_var)
                        color = "blue" if is_canonical else "red"
                        style = "solid" if target_col == column else "dashed"

                        dot.edge(src_id, target_id,
                                label=f"{token}\n({token_str})",
                                color=color,
                                style=style)
                    edges_added.add(edge_key)

        # Save the dot file without rendering
        dot.save(filename + ".dot")
        # If you need to view the file locally, you can copy it to a machine with Graphviz
        # installed and run `dot -Tpdf filename.dot -o filename.pdf`.


    def model_count(self) -> Dict[Tuple[Optional[tuple], int], int]:
        """
        Compute the number of tokenizations for each node using bottom-up counting.
        Returns a dictionary mapping (var, column) -> count.
        """
        counts = {}

        # Gather all nodes including terminal nodes
        all_nodes = set(self.nodes.keys())
        for col in range(self.K + 1):
            all_nodes.add((None, col))

        # Compute children_count and reverse_edges
        children_count = {n: 0 for n in all_nodes}
        reverse_edges = {n: [] for n in all_nodes}
        for node_key, node_data in self.nodes.items():
            for token, t_var, t_col in node_data.outgoing:
                child = (None, t_col) if t_var is None else (t_var, t_col)
                children_count[node_key] += 1
                reverse_edges[child].append(node_key)

        # Initialize terminal nodes with count=1
        queue = deque()
        terminal_node = (None, 0)
        counts[terminal_node] = 1
        queue.append(terminal_node)

        # Topological order (in reverse) to fill counts
        while queue:
            node = queue.popleft()
            node_count = counts[node]
            # Update parents
            for parent in reverse_edges[node]:
                children_count[parent] -= 1
                if children_count[parent] == 0:
                    # Compute parent's count
                    p_count = 0
                    for token, t_var, t_col in self.nodes[parent].outgoing:
                        child = (None, t_col) if t_var is None else (t_var, t_col)
                        p_count += counts[child]
                    counts[parent] = p_count
                    queue.append(parent)

        return counts

    def get_root_counts(self, counts: Dict[Tuple[Optional[tuple], int], int]) -> Tuple[int, Dict[int, int]]:
        """
        Using the counts dictionary returned by model_count(),
        compute the total number of valid tokenizations and also
        return a dictionary of counts for each root node (0, None, col).

        Returns:
        (total_count, root_counts_dict)
        total_count: int, the sum of counts over all root nodes.
        root_counts_dict: {col: count_for_that_root_node}
        """
        root_counts = {}
        for col in range(self.K + 1):
            root_node = ((0, None), col)
            root_counts[col] = counts.get(root_node, 0)
        total_count = sum(root_counts.values())
        return total_count, root_counts

    def model_enumeration(self, start_node: Tuple[Optional[tuple], int], prefix: list = [],
                          suffix: list = []) -> List[List[int]]:
        """
        Enumerate all tokenizations (as lists of token IDs) from a given start node to terminal nodes.
        """
        var, col = start_node
        # If it's a terminal node
        if var is None:
            return [[]]  # One valid tokenization: empty, since we're at the end

        enumerations = []
        node_data = self.nodes[start_node]
        for token, t_var, t_col in node_data.outgoing:
            if t_var is None:
                # Edge to a terminal node
                # If token is None, it means a direct connection to terminal with no token added
                # If token is not None, it forms a single-token tokenization
                enumerations.append(([] if token is None else [token])+suffix)
            else:
                # Enumerate from the child node
                child_enums = self.model_enumeration((t_var, t_col), prefix=[], suffix=suffix)
                # Prepend current token if not None
                if token is not None:
                    for seq in child_enums:
                        enumerations.append(prefix + [token] + seq)
                else:
                    enumerations.extend(child_enums)
        return enumerations

    import random

    def uniform_sample_across_roots(self, counts: dict) -> List[int]:
        """
        Uniformly sample a tokenization from the entire multi-rooted MDD, considering all root nodes
        (0, None, col) for col in [0..K].
        First choose a root node based on model counts, then do uniform sampling from that root.
        """
        # Compute total_count and root_counts
        total_count, root_counts = self.get_root_counts(counts)

        # If no valid tokenizations
        if total_count == 0:
            return []

        # Choose which root column to start from
        # Weighted by root_counts[col] / total_count
        r = random.uniform(0, total_count)
        running_sum = 0
        chosen_col = 0
        for col, ccount in root_counts.items():
            running_sum += ccount
            if r <= running_sum:
                chosen_col = col
                break

        # Now sample uniformly from the chosen root
        start_node = ((0, None), chosen_col)
        return self.uniform_sample(start_node, counts)

    def uniform(self, k: int, n: int = 1, counts: Dict[Tuple[Optional[tuple], int], int] = None) -> List[int]:
        "Sample tokenizations uniformly conditioned on edge distance k."
        if counts is None: counts = self.model_count()
        f = functools.partial(self.uniform_sample, ((0, None), k), counts)
        return [f() for _ in range(n)] if n > 1 else f()

    def sat_at_distance(self, k: int) -> bool:
        """
        Returns whether the MDD is satisfiable conditioned on distance k, i.e. whether there exists
        a tokenization with distance k from canonical.
        """
        return ((0, None), k) in self.nodes

    def uniform_sample(self, start_node: Tuple[Optional[tuple], int], counts: Dict[Tuple[Optional[tuple], int], int]) -> List[int]:
        """
        Uniformly sample a tokenization from the MDD starting at 'start_node', using counts for renormalization.
        """
        var, col = start_node
        # Terminal node
        if var is None:
            return []

        node_data = self.nodes[start_node]
        total_count = counts[start_node]

        # Compute cumulative probabilities based on child's counts
        cumulative = []
        running_sum = 0
        edges_info = []
        for token, t_var, t_col in node_data.outgoing:
            if t_var is None:
                child_count = counts[(None, t_col)]
            else:
                child_count = counts[(t_var, t_col)]
            running_sum += child_count
            cumulative.append(running_sum)
            edges_info.append((token, t_var, t_col))

        # Select edge based on cumulative distribution
        r = random.uniform(0, running_sum)
        for i, val in enumerate(cumulative):
            if r <= val:
                chosen_token, chosen_t_var, chosen_t_col = edges_info[i]
                break

        # Recurse
        if chosen_t_var is None:
            # Terminal edge
            return [] if chosen_token is None else [chosen_token]
        else:
            suffix = self.uniform_sample((chosen_t_var, chosen_t_col), counts)
            return [chosen_token] + suffix if chosen_token is not None else suffix

    def _compute_toxicity(self):
        if self.toxicity_info is not None: return

        # Gather all nodes including terminal nodes
        all_nodes = set(self.nodes.keys())
        for col in range(self.K + 1):
            all_nodes.add((None, col))

        # Compute children_count and reverse_edges
        # reverse_edges[node] = list of parent nodes that point to `node`
        children_count = {n: 0 for n in all_nodes}
        reverse_edges = {n: [] for n in all_nodes}

        for node_key, node_data in self.nodes.items():
            for token, t_var, t_col in node_data.outgoing:
                child = (None, t_col) if t_var is None else (t_var, t_col)
                children_count[node_key] += 1
                reverse_edges[child].append((node_key, token))  # store parent and token on that edge


        # toxicity_info[node] = (toxicity_value, path_list)
        self.toxicity_info = {}
        queue = deque()

        # Initialize terminal nodes with toxicity=0.0 and path=[]
        terminal_node = (None, 0)
        self.toxicity_info[terminal_node] = (0.0, [])
        queue.append(terminal_node)

        # Topological order (in reverse) to fill toxicity_info
        while queue:
            node = queue.popleft()
            node_toxic, node_path = self.toxicity_info[node]

            # Update parents
            for (parent, parent_token) in reverse_edges[node]:
                # We have a candidate path through this child
                candidate_toxic = node_toxic + _token_toxicity(self.toxicity_model, parent_token)
                if parent not in self.toxicity_info or candidate_toxic > self.toxicity_info[parent][0]:
                    # Update parent's best toxicity and path
                    if parent_token is None:
                        # No token added
                        self.toxicity_info[parent] = (candidate_toxic, node_path)
                    else:
                        self.toxicity_info[parent] = (candidate_toxic, [parent_token] + node_path)

                children_count[parent] -= 1
                if children_count[parent] == 0:
                    queue.append(parent)

    def get_most_toxic_tokenization(self, k: int = -1, return_toxicity: bool = False,
                                    return_all_distances: bool = False) -> Tuple[float, List[int]]:
        """
        Start from the terminal nodes (None, col) with toxicity = 0.0 and path = [].
        Move backwards in a topological order (just like model_count does) but take the max toxicity
        instead of summation.

        Returns:
        (max_toxicity_score, token_ids_list)
        """
        if not hasattr(self, 'toxicity_model'):
            raise RuntimeError("Toxicity model not loaded. Call load_toxicity_model(data_dir) first.") 
        if k >= 0: assert self.sat_at_distance(k), f"There are no tokenizations of distance {k} for the given string."

        # Compute toxicity and store data in `self.toxicity_info`.
        self._compute_toxicity()

        # Now choose the best root node ((0, None), col)
        root_nodes = [((0, None), col) for col in range(self.K + 1)] if k < 0 else [((0, None), k)]
        best_per_distance = {}
        max_toxic = float('-inf')
        max_path = []

        for r in root_nodes:
            if r in self.toxicity_info:
                dist_toxic = self.toxicity_model['intercept'] + self.toxicity_info[r][0]
                dist_path = self.toxicity_info[r][1]
                best_per_distance[r[1]] = (dist_toxic, dist_path)
                if dist_toxic > max_toxic:
                    max_toxic = dist_toxic
                    max_path = dist_path

        ret_val = []
        if return_toxicity: ret_val.append(max_toxic)
        ret_val.append(max_path)
        if return_all_distances: ret_val.append(best_per_distance)

        return ret_val[0] if len(ret_val) == 1 else tuple(ret_val)

def _build_mrmdd_parallel_batch(tok: transformers.AutoTokenizer, S: list, k: list, prefix_space: bool,
                                reuse_mdd: list, **kwargs) -> list:
    return [build_mrmdd(tok, S, k[i], prefix_space=prefix_space, reuse_mdd=reuse_mdd[i], **kwargs) for i in range(len(S))]
def build_mrmdd_parallel_pargs(tok: transformers.AutoTokenizer, S: list, k: list, prefix_space: bool,
                               reuse_mdd: list, **kwargs) -> list:
    n = multiprocessing.cpu_count()
    with multiprocessing.Pool(n) as pool:
        S = utils.batch_slices(n, len(S))
        R = [pool.apply_async(_build_mrmdd_parallel_batch, (tok, S[s], k[s], prefix_space, reuse_mdd[s]), kwargs) for s in S]
        M = [x for r in tqdm.tqdm(R, desc="Building MDDs") for x in r.get()]
    return M

def build_mrmdd(tok: transformers.AutoTokenizer, S: str, k: int, prefix_space: bool = False,
                add_special_tokens: bool = False, reuse_mdd: mdd.MDD = None, **kwargs) -> MultiRootedMDD:
    if isinstance(S, str):
        origin_tok = tok.encode(S, add_special_tokens=add_special_tokens)
        if prefix_space: S = ' ' + S
    else:
        assert reuse_mdd is not None, "If a tokenization is provided, reuse_mdd must be given."
        origin_tok = S
        S = tok.decode(S, clean_up_tokenization_spaces=False)
    if reuse_mdd is None: reuse_mdd = mdd.build_mdd(tok, S, prefix_space=False)
    return MultiRootedMDD(reuse_mdd, k, origin_tok, S, **kwargs).prune()
