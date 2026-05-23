import torch
from typing import Dict, Optional, Set, Tuple, Any, List


class FEXCacheManager:

    # ---- Core cache control ----
    def __init__(self, encoder):
        self.encoder = encoder
        self._caches: Dict[str, Any] = {}
        self._device_caches: Dict[str, Dict[torch.device, Any]] = {}

    def clear_all(self):
        self._caches.clear()
        self._device_caches.clear()

    def clear_frozen_cache(self):
        self._caches.pop('frozen_node_outputs', None)
        self._caches.pop('frozen_top1_signature', None)

    # ---- Device-aware cache helpers ----
    def get_or_create_device_cache(self, cache_name: str, device: torch.device, factory_fn):
        if cache_name not in self._device_caches:
            self._device_caches[cache_name] = {}
        if device not in self._device_caches[cache_name]:
            self._device_caches[cache_name][device] = factory_fn(device)
        return self._device_caches[cache_name][device]

    # ---- Vocab LUT cache ----
    def get_vocab_lut(self, device: torch.device) -> Dict[str, torch.Tensor]:
        return self.get_or_create_device_cache('vocab_lut', device, self._build_vocab_lut)

    def _build_vocab_lut(self, device: torch.device) -> Dict[str, torch.Tensor]:
        w2id = self.encoder.equation_word2id
        vocab_size = self.encoder.env.n_words

        lut = {
            'is_mantissa': torch.zeros(vocab_size, device=device),
            'mantissa_val': torch.zeros(vocab_size, device=device),
            'is_exponent': torch.zeros(vocab_size, device=device),
            'exponent_val': torch.zeros(vocab_size, device=device),
            'is_sign': torch.zeros(vocab_size, device=device),
            'sign_val': torch.zeros(vocab_size, device=device),
            'var_idx': torch.full((vocab_size,), -1, device=device, dtype=torch.long),
        }

        for word, idx in w2id.items():
            if idx >= vocab_size:
                continue
            if word.startswith('N'):
                try:
                    lut['is_mantissa'][idx] = 1.0
                    lut['mantissa_val'][idx] = float(word[1:])
                except Exception:
                    pass
            elif word.startswith('E'):
                try:
                    val = float(word[1:].replace('m', '-')) if 'm' in word else float(word[1:])
                    lut['is_exponent'][idx] = 1.0
                    lut['exponent_val'][idx] = val
                except Exception:
                    pass
            elif word in ['+', '-']:
                lut['is_sign'][idx] = 1.0
                lut['sign_val'][idx] = 1.0 if word == '+' else -1.0
            elif word.startswith('x_'):
                try:
                    lut['var_idx'][idx] = int(word.split('_')[1])
                except Exception:
                    pass

        return lut

    # ---- Relaxed eval plan cache ----
    def get_relaxed_eval_plan(self, device: torch.device) -> Dict[str, Any]:
        return self.get_or_create_device_cache('eval_plan', device, self._build_eval_plan)

    def _build_eval_plan(self, device: torch.device) -> Dict[str, Any]:
        nodes = self.encoder.tree.nodes
        seq_pos_cache = self.encoder._get_seq_pos_cache()
        seq_pos_tensor = torch.as_tensor(seq_pos_cache, dtype=torch.long, device=device)
        token_row_idx = seq_pos_tensor + 1

        max_layer = max(node['layer'] for node in nodes)
        nodes_by_layer = [[] for _ in range(max_layer + 1)]
        leaf_indices = []
        parent = {}

        for idx, node in enumerate(nodes):
            nodes_by_layer[node['layer']].append(idx)
            if node['type'] == 'leaf':
                leaf_indices.append(idx)
            elif node['type'] == 'unary':
                child = self.encoder._get_unary_child_idx(idx)
                if child >= 0:
                    parent[child] = idx
            elif node['type'] == 'binary':
                left, right = self.encoder._get_binary_child_indices(idx)
                if left >= 0:
                    parent[left] = idx
                if right >= 0:
                    parent[right] = idx

        seq_pos_to_node = {}
        for idx, node in enumerate(nodes):
            pos = int(seq_pos_cache[idx]) + 1
            seq_pos_to_node[pos] = idx
            if node['type'] == 'leaf':
                seq_pos_to_node[pos + 1] = idx

        return {
            'nodes': nodes,
            'num_nodes': len(nodes),
            'max_layer': max_layer,
            'nodes_by_layer': nodes_by_layer,
            'leaf_indices': leaf_indices,
            'token_row_idx': token_row_idx,
            'token_row_max': int(token_row_idx.max().item()) if token_row_idx.numel() > 0 else -1,
            'parent': parent,
            'seq_pos_to_node': seq_pos_to_node,
        }

    # ---- Frozen subtree cache ----
    def get_frozen_cache_key(
        self,
        topk_indices: torch.Tensor,
        frozen_positions: Set[int],
        X: torch.Tensor
    ) -> str:
        if not frozen_positions:
            return ""

        # Extract top-1 token IDs for frozen positions
        frozen_list = sorted(frozen_positions)
        top1_frozen = topk_indices[frozen_list, 0].cpu().tolist()

        # Include data shape in key
        key_parts = [
            "frozen",
            f"pos={frozen_list}",
            f"top1={top1_frozen}",
            f"Xshape={X.shape}",
            f"device={X.device}",
        ]
        return "|".join(key_parts)

    def get_cached_frozen_outputs(
        self,
        cache_key: str
    ) -> Optional[Dict[int, torch.Tensor]]:
        if not cache_key:
            return None
        frozen_cache = self._caches.get('frozen_node_outputs', {})
        return frozen_cache.get(cache_key)

    def cache_frozen_outputs(
        self,
        cache_key: str,
        node_outputs: Dict[int, torch.Tensor],
        frozen_positions: Set[int],
        plan: Dict[str, Any]
    ):
        if not cache_key or not frozen_positions:
            return

        seq_pos_to_node = plan['seq_pos_to_node']

        frozen_node_outputs = {}
        for pos in frozen_positions:
            node_idx = seq_pos_to_node.get(pos)
            if node_idx is not None and 0 <= node_idx < len(node_outputs):
                val = node_outputs[node_idx]
                if val is not None:
                    frozen_node_outputs[node_idx] = val.detach().clone()

        if 'frozen_node_outputs' not in self._caches:
            self._caches['frozen_node_outputs'] = {}

        frozen_cache = self._caches['frozen_node_outputs']
        if len(frozen_cache) >= 10:
            oldest_key = next(iter(frozen_cache))
            del frozen_cache[oldest_key]

        frozen_cache[cache_key] = frozen_node_outputs

    def apply_cached_frozen_outputs(
        self,
        node_outputs: List[Optional[torch.Tensor]],
        cache_key: str,
        frozen_positions: Set[int],
        plan: Dict[str, Any]
    ) -> int:
        cached = self.get_cached_frozen_outputs(cache_key)
        if cached is None:
            return 0

        seq_pos_to_node = plan['seq_pos_to_node']
        restored = 0

        for pos in frozen_positions:
            node_idx = seq_pos_to_node.get(pos)
            if node_idx is not None and node_idx in cached:
                node_outputs[node_idx] = cached[node_idx]
                restored += 1

        return restored

    # ---- Fast decode cache ----
    def get_fast_decode_cache(self) -> Dict[str, Any]:
        cache = self._caches.get('fast_decode')
        if cache is not None:
            return cache

        cache = self._build_fast_decode_cache()
        self._caches['fast_decode'] = cache
        return cache

    def _build_fast_decode_cache(self) -> Dict[str, Any]:
        import sympy as sp
        from .fixed_tree_encoder import _neg, _inv, _sqrt, _pow2, _pow3, _identity

        nodes = self.encoder.tree.nodes

        nodes_with_idx = list(enumerate(nodes))
        nodes_sorted = sorted(nodes_with_idx, key=lambda x: (x[1]['layer'], -x[0]), reverse=True)

        execution_order = []
        node_seq_map = {}
        curr_seq_idx = 0
        for idx, node in enumerate(nodes):
            node_seq_map[idx] = curr_seq_idx
            if node['type'] == 'leaf':
                curr_seq_idx += 2
            else:
                curr_seq_idx += 1

        for idx, node in nodes_sorted:
            children = []
            if node['type'] == 'unary':
                children = [self.encoder._get_unary_child_idx(idx)]
            elif node['type'] == 'binary':
                children = self.encoder._get_binary_child_indices(idx)

            execution_order.append({
                'idx': idx,
                'type': node['type'],
                'seq_idx': node_seq_map[idx],
                'children': children
            })

        max_id = max(self.encoder.equation_id2word.keys()) if self.encoder.equation_id2word else 0
        token_cache = [None] * (max_id + 1)

        mantissa_ids = []
        exponent_ids = []
        sign_ids = []
        var_ids = []

        neg_one = sp.Integer(-1)
        half = sp.Rational(1, 2)
        two = sp.Integer(2)
        three = sp.Integer(3)

        def make_add(a, b): return sp.Add(a, b, evaluate=False)
        def make_sub(a, b): return sp.Add(a, sp.Mul(neg_one, b, evaluate=False), evaluate=False)
        def make_mul(a, b): return sp.Mul(a, b, evaluate=False)
        def make_div(a, b): return sp.Mul(a, sp.Pow(b, neg_one, evaluate=False), evaluate=False)
        def make_pow(a, b): return sp.Pow(a, b, evaluate=False)
        def make_id(a, b=None): return a

        def make_unary_op(op):
            if op == sp.sin: return lambda x: sp.sin(x, evaluate=False)
            if op == sp.cos: return lambda x: sp.cos(x, evaluate=False)
            if op == sp.tan: return lambda x: sp.tan(x, evaluate=False)
            if op == sp.asin: return lambda x: sp.asin(x, evaluate=False)
            if op == sp.acos: return lambda x: sp.acos(x, evaluate=False)
            if op == sp.atan: return lambda x: sp.atan(x, evaluate=False)
            if op == sp.exp: return lambda x: sp.exp(x, evaluate=False)
            if op == sp.log: return lambda x: sp.log(x, evaluate=False)
            if op == sp.Abs: return lambda x: sp.Abs(x, evaluate=False)
            if op == _neg: return lambda x: sp.Mul(neg_one, x, evaluate=False)
            if op == _inv: return lambda x: sp.Pow(x, neg_one, evaluate=False)
            if op == _sqrt: return lambda x: sp.Pow(x, half, evaluate=False)
            if op == _pow2: return lambda x: sp.Pow(x, two, evaluate=False)
            if op == _pow3: return lambda x: sp.Pow(x, three, evaluate=False)
            if op == _identity: return lambda x: x
            return lambda x: op(x)

        bin_op_funcs = [None] * (max_id + 1)
        unary_op_funcs = [None] * (max_id + 1)

        for tid, word in self.encoder.equation_id2word.items():
            if word in ['add', 'sub', 'mul', 'div', 'pow', '<ID_Binary>']:
                if word == 'add': func = make_add
                elif word == 'sub': func = make_sub
                elif word == 'mul': func = make_mul
                elif word == 'div': func = make_div
                elif word == 'pow': func = make_pow
                else: func = make_id
                bin_op_funcs[tid] = func

            token_to_unary = getattr(self.encoder, 'token_to_unary', {})
            if word in token_to_unary or word == '<ID_Unary>':
                if word == '<ID_Unary>':
                    op = _identity
                else:
                    op = token_to_unary[word]
                unary_op_funcs[tid] = make_unary_op(op)

            if word.startswith('N') or word.startswith('-N'):
                mantissa_ids.append(tid)
            elif word.startswith('E'):
                exponent_ids.append(tid)
            elif word in ['+', '-']:
                sign_ids.append(tid)
            elif word.startswith('x_'):
                var_ids.append(tid)

        leaf_pair_cache = {}

        for m_id in mantissa_ids:
            m_word = self.encoder.equation_id2word[m_id]
            try:
                if m_word.startswith('-N'):
                    m_val = -int(m_word[2:])
                else:
                    m_val = int(m_word[1:])

                for e_id in exponent_ids:
                    e_word = self.encoder.equation_id2word[e_id]
                    e_val = int(e_word[1:])

                    if e_val >= 0:
                        val = m_val * (10 ** e_val)
                        s_val = sp.Integer(val)
                    else:
                        val = float(m_val) * (10.0 ** e_val)
                        if abs(val - round(val)) < 1e-10:
                            s_val = sp.Integer(round(val))
                        else:
                            s_val = sp.Float(val)
                    leaf_pair_cache[(m_id, e_id)] = s_val
            except: pass

        for s_id in sign_ids:
            s_word = self.encoder.equation_id2word[s_id]
            is_neg = (s_word == '-')
            for v_id in var_ids:
                v_word = self.encoder.equation_id2word[v_id]
                sym = sp.Symbol(v_word)
                if is_neg:
                    res = sp.Mul(neg_one, sym, evaluate=False)
                else:
                    res = sym
                leaf_pair_cache[(s_id, v_id)] = res

        pad_id = self.encoder.env.equation_word2id.get('<PAD>', -1)
        if pad_id != -1:
            leaf_pair_cache[(pad_id, pad_id)] = sp.Integer(0)

        return {
            'execution_order': execution_order,
            'token_cache': token_cache,
            'bin_op_funcs': bin_op_funcs,
            'unary_op_funcs': unary_op_funcs,
            'leaf_pair_cache': leaf_pair_cache,
        }

    def clear_fast_decode_cache(self):
        self._caches.pop('fast_decode', None)

    # ---- Sequence position cache ----
    def get_seq_pos_cache(self) -> List[int]:
        cache = self._caches.get('seq_pos')
        if cache is not None:
            return cache

        cache = []
        for node in self.encoder.tree.nodes:
            cache.append(self._compute_sequence_position(node['inorder_idx']))
        self._caches['seq_pos'] = cache
        return cache

    def _compute_sequence_position(self, inorder_idx: int) -> int:
        from .fixed_tree_encoder import NODE_LEAF
        seq_pos = 0
        for i in range(inorder_idx):
            node = self.encoder.tree.nodes[i]
            if node['type'] == NODE_LEAF:
                seq_pos += 2
            else:
                seq_pos += 1
        return seq_pos

    def clear_seq_pos_cache(self):
        self._caches.pop('seq_pos', None)

    # ---- Structure mask cache for topk building ----
    def get_structure_mask(
        self,
        active_seq_len: int,
        fallback_id: int,
        vocab_size: int,
        device: torch.device,
        pos2type: Dict[int, str],
        groups: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Get cached structure mask for _build_topk_indices."""
        cache_name = 'structure_mask'
        cache_key = (active_seq_len, fallback_id, vocab_size)
        
        if cache_name not in self._device_caches:
            self._device_caches[cache_name] = {}
        
        if cache_key not in self._device_caches[cache_name]:
            # Build mask: [active_seq_len, vocab_size]
            allowed_mask = torch.zeros(active_seq_len, vocab_size, device=device, dtype=torch.float32)
            
            # Position 0 and last are fixed to fallback_id
            allowed_mask[0, fallback_id] = 1.0
            allowed_mask[active_seq_len - 1, fallback_id] = 1.0
            
            # Fill allowed tokens based on position type
            for pos, ptype in pos2type.items():
                if pos <= 0 or pos >= active_seq_len - 1:
                    continue
                if ptype == 'bin':
                    allowed_mask[pos, groups['bin']] = 1.0
                elif ptype == 'una':
                    allowed_mask[pos, groups['una']] = 1.0
                elif ptype == 'leaf_p1':
                    allowed_mask[pos, groups['man']] = 1.0
                    allowed_mask[pos, groups['sign']] = 1.0
                elif ptype == 'leaf_p2':
                    allowed_mask[pos, groups['exp']] = 1.0
                    allowed_mask[pos, groups['var']] = 1.0
            
            self._device_caches[cache_name][cache_key] = allowed_mask
        
        return self._device_caches[cache_name][cache_key]

    def clear_structure_mask_cache(self):
        self._device_caches.pop('structure_mask', None)

    # ---- Stats ----
    def get_stats(self) -> Dict[str, Any]:
        stats = {
            'general_caches': list(self._caches.keys()),
            'device_caches': {
                name: list(device_dict.keys())
                for name, device_dict in self._device_caches.items()
            },
        }

        frozen_cache = self._caches.get('frozen_node_outputs', {})
        stats['frozen_cache_entries'] = len(frozen_cache)

        return stats
