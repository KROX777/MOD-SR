from typing import List, Dict, Tuple, Optional, Any
from collections import Counter
import sympy as sp
from sympy import symbols, sympify, Add, Mul, Pow, Symbol, Integer, Float, Rational
import numpy as np
import torch
from pathlib import Path
try:
    import torch_npu
except:
    pass

from symbolicregression.visualization.guidance_video import render_fex_tree, render_relaxed_subtree
from .inner_loop_executor import FEXInnerLoopExecutor


# ============ Helper functions for pickling (must be at module level) ============
def _neg(x):
    return -x

def _inv(x):
    return 1/x

def _pow2(x):
    return x**2

def _pow3(x):
    return x**3

def _sqrt(x):
    return x**sp.Rational(1, 2)

def _identity(e):
    return e

def _expand(e):
    return sp.expand(e)

def _factor(e):
    return sp.factor(e)

def _simplify(e):
    return sp.simplify(e)



# ============ Operator categories (used for FEX structure validation and decoding) ============
# Binary operators - these tokens must appear at Binary node positions
BINARY_OPS = ['add', 'sub', 'mul', 'div', 'pow', '<ID_Binary>']

# Unary operators - these tokens must appear at Unary node positions  
UNARY_OPS = ['sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'exp', 'log', 'sqrt', 'abs', 'neg', 'inv', 'pow2', 'pow3', '<ID_Unary>']

# Leaf node Position 1 tokens (constants use N00-N99, variables use +/-)
LEAF_POS1_MANTISSA = [f'N{i:02d}' for i in range(100)]  # N00, N01, ..., N99
LEAF_POS1_SIGN = ['+', '-']

# Leaf node Position 2 tokens (constants use E-3 to E3, variables use x_0...x_9)
LEAF_POS2_EXPONENT = ['E-3', 'E-2', 'E-1', 'E0', 'E1', 'E2', 'E3']
LEAF_POS2_VARIABLE = [f'x_{i}' for i in range(10)]


# ============ Node Type Constants ============
NODE_BINARY = 'binary'
NODE_UNARY = 'unary'
NODE_LEAF = 'leaf'


class FEXTree:
    """
    Fixed-depth FEX-style tree structure
    
    Note: leaf nodes occupy 2 positions in the token sequence!
    """
    
    def __init__(self, depth: int = 3):
        """
        Initialize a fixed-depth FEX tree
        
        Args:
            depth: Tree depth (number of Binary layers)
        """
        self.depth = depth
        self.nodes = self._build_tree_structure()
        self.total_nodes = len(self.nodes)
        self.leaf_count = sum(1 for n in self.nodes if n['type'] == NODE_LEAF)
        # Sequence length = non-leaf node count + leaf node count * 2 (each leaf occupies 2 token positions)
        self.sequence_length = (self.total_nodes - self.leaf_count) + self.leaf_count * 2
        
    def _build_tree_structure(self) -> List[Dict]:
        """
        Build the FEX tree structure and return the node list ordered by inorder index
        
        Depth-3 structure (21 nodes):
        - Layer 0: 1 Binary (root)     = 2^0
        - Layer 1: 2 Unary             = 2^1
        - Layer 2: 2 Binary            = 2^1
        - Layer 3: 4 Unary             = 2^2
        - Layer 4: 4 Binary            = 2^2
        - Layer 5: 8 Leaf              = 2^3 (leaf layer)
        
        Pattern:
        - There are 2 * depth layers in total (including the leaf layer)
        - Layers 0, 2, 4, ... (even layers): Binary, with depth Binary layers total
        - Layers 1, 3, ... (odd layers, excluding leaves): Unary, with depth-1 Unary layers total
        - The last layer is Leaf
        
        For depth=3: non-leaf layers 0,1,2,3,4 + leaf layer 5 = 6 layers
        - Layer 0 (B): 2^0=1, Layer 1 (U): 2^1=2, Layer 2 (B): 2^1=2
        - Layer 3 (U): 2^2=4, Layer 4 (B): 2^2=4, Layer 5 (L): 2^3=8
        - Total nodes: 1+2+2+4+4+8 = 21
        - Sequence length: 13 + 8*2 = 29
        """
        nodes_by_layer = []
        
        # Non-leaf layers: layer 0 to layer (2*depth-2), with 2*depth-1 layers total
        # Layer 0: Binary, layer 1: Unary, layer 2: Binary, ...
        # The last Binary layer is layer (2*depth-2)
        num_non_leaf_layers = 2 * self.depth - 1  # depth=3 => 5 layers (0-4)
        
        for layer in range(num_non_leaf_layers):
            if layer % 2 == 0:  # Binary layers: 0, 2, 4, ...
                count = 2 ** (layer // 2)
                node_type = NODE_BINARY
            else:  # Unary layers: 1, 3, ...
                count = 2 ** ((layer + 1) // 2)
                node_type = NODE_UNARY
            
            layer_nodes = []
            for i in range(count):
                layer_nodes.append({
                    'layer': layer,
                    'type': node_type,
                    'index_in_layer': i,
                    'token': None
                })
            nodes_by_layer.append(layer_nodes)
        
        # Leaf layer (layer 2*depth-1)
        leaf_layer_idx = num_non_leaf_layers  # depth=3 => 5
        leaf_count = 2 ** self.depth  # depth=3 => 8
        leaf_layer = []
        for i in range(leaf_count):
            leaf_layer.append({
                'layer': leaf_layer_idx,
                'type': NODE_LEAF,
                'index_in_layer': i,
                'token': None
            })
        nodes_by_layer.append(leaf_layer)
        
        nodes = self._to_inorder_list(nodes_by_layer)
        self._build_child_cache(nodes)
        return nodes
    
    def _to_inorder_list(self, nodes_by_layer: List[List[Dict]]) -> List[Dict]:
        """Convert the layered structure into an inorder traversal list"""
        def inorder_traverse(layer: int, node_idx: int) -> List[Dict]:
            if layer >= len(nodes_by_layer) or node_idx >= len(nodes_by_layer[layer]):
                return []
                
            current_node = nodes_by_layer[layer][node_idx]
            result = []
            
            if current_node['type'] == NODE_LEAF:
                return [current_node]
            elif current_node['type'] == NODE_UNARY:
                child_result = inorder_traverse(layer + 1, node_idx)
                result.extend(child_result)
                result.append(current_node)
            elif current_node['type'] == NODE_BINARY:
                left_result = inorder_traverse(layer + 1, 2 * node_idx)
                result.extend(left_result)
                result.append(current_node)
                right_result = inorder_traverse(layer + 1, 2 * node_idx + 1)
                result.extend(right_result)
            
            return result
        
        inorder_nodes = inorder_traverse(0, 0)
        for idx, node in enumerate(inorder_nodes):
            node['inorder_idx'] = idx
        return inorder_nodes

    def _build_child_cache(self, nodes: List[Dict]) -> None:
        """Precompute each node's child inorder indices to avoid repeated scans later."""
        self._layer_index_to_inorder = {}
        for node in nodes:
            self._layer_index_to_inorder[(node['layer'], node['index_in_layer'])] = node['inorder_idx']

        self._binary_child_cache = {}
        self._unary_child_cache = {}

        for node in nodes:
            layer = node['layer']
            idx = node['index_in_layer']
            inorder = node['inorder_idx']

            if node['type'] == NODE_UNARY:
                child = self._layer_index_to_inorder.get((layer + 1, idx), -1)
                self._unary_child_cache[inorder] = child
            elif node['type'] == NODE_BINARY:
                left = self._layer_index_to_inorder.get((layer + 1, 2 * idx), -1)
                right = self._layer_index_to_inorder.get((layer + 1, 2 * idx + 1), -1)
                self._binary_child_cache[inorder] = (left, right)
    
    def get_node_by_inorder_idx(self, idx: int) -> Optional[Dict]:
        """Get a node by inorder index"""
        if 0 <= idx < len(self.nodes):
            return self.nodes[idx]
        return None
    
    def get_root_inorder_idx(self) -> int:
        """Get the inorder index of the root node"""
        for node in self.nodes:
            if node['layer'] == 0:
                return node['inorder_idx']
        return -1

    def get_inorder_idx_from_layer(self, layer: int, index_in_layer: int) -> int:
        """
        Return the corresponding inorder index for (layer, index_in_layer).
        Return -1 if it does not exist.
        """
        return self._layer_index_to_inorder.get((layer, index_in_layer), -1)


class FixedTreeEncoder:
    """
    Encoder that maps symbolic expressions to a fixed-depth FEX tree
    
    Key: leaf nodes occupy 2 token positions!
    - Constants: [N00-N99, E-X...EX] (X is determined by environment.max_exponent_prefactor)
    - Variables: [+/-, x_0...x_9]
    
    Note: env must be provided so the encoder can use the environment's shared vocabulary!
    """
    
    def __init__(self, depth: int = 3, env=None, max_transform_attempts: int = 0):
        """
        Initialize the encoder
        
        Args:
            depth: Tree depth (number of Binary layers)
            env: FunctionEnvironment instance (required), provides the shared vocabulary and float_encoder
            max_transform_attempts: Maximum number of transforms to try when encoding fails (0-4)
                                    0: original expression only
                                    1: original + expand
                                    2: original + expand + factor
                                    3: original + expand + factor + simplify
                                    4: original + expand + factor + simplify + collect
        """
        if env is None:
            raise ValueError("env must be provided! FixedTreeEncoder needs the environment's shared vocabulary")
        
        self.depth = depth
        self.env = env
        self.max_transform_attempts = min(max(0, max_transform_attempts), 4)  # Clamp to the 0-4 range
        self.tree = FEXTree(depth)
        self.restrict_pow_top1 = False
        self._inner_loop_executor = FEXInnerLoopExecutor(self)
        self._cache_manager = self._inner_loop_executor.cache_manager
        
        # Use the environment vocabulary directly (including all float_words)
        self.equation_word2id = env.equation_word2id
        self.equation_id2word = env.equation_id2word
        self.float_encoder = env.float_encoder
        
        # Get the available exponent range from float_encoder
        # FloatSequences uses the range [-max_exponent_prefactor, max_exponent_prefactor]
        self.max_exponent = env.float_encoder.max_exponent if hasattr(env.float_encoder, 'max_exponent') else 3
        
        # Mapping from SymPy types to operator tokens
        self.sympy_binary_map = {
            sp.Add: 'add',
            sp.Mul: 'mul',
        }
        self.sympy_func_map = {
            'sin': 'sin', 'cos': 'cos', 'tan': 'tan',
            'asin': 'arcsin', 'acos': 'arccos', 'atan': 'arctan',
            'exp': 'exp', 'log': 'log', 'sqrt': 'sqrt',
            'Abs': 'abs',
        }
        # Reverse mapping (used for decoding)
        self.token_to_unary = {
            'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
            'arcsin': sp.asin, 'arccos': sp.acos, 'arctan': sp.atan,
            'exp': sp.exp, 'log': sp.log, 'sqrt': sp.sqrt,
            'abs': sp.Abs, 'neg': _neg,
            'inv': _inv, 'pow2': _pow2, 'pow3': _pow3
        }
    
    def encode(self, expr: Any) -> List[int]:
        """
        Encode a SymPy expression or Node object into a sequence of token indices
        
        Return sequence length = non-leaf node count + leaf node count * 2
        
        Strategy: try different SymPy transformations in order and return on the first success
        """
        # If this is a Node object, first convert it to a SymPy expression (without evaluation to preserve structure)
        if not isinstance(expr, sp.Expr):
            if hasattr(expr, 'infix'):
                infix = expr.infix()
            else:
                infix = str(expr)
            
            local_dict = {}
            if self.env is not None and hasattr(self.env, 'simplifier'):
                local_dict = self.env.simplifier.local_dict
            
            # Use evaluate=False to ensure 1+1 is not folded into 2
            # sympify in 1.11 uses 'locals' dict, not 'local_dict'
            try:
                expr = sp.sympify(infix, evaluate=False, locals=local_dict)
            except TypeError:
                # Fallback if 'locals' matches unexpected arg or logic
                from sympy.parsing.sympy_parser import parse_expr
                expr = parse_expr(infix, local_dict=local_dict, evaluate=False)

        # Define the trial order: original -> expand -> factor -> simplify -> collect
        transformations = [
            ("original", _identity),
            ("expand", _expand),
            ("factor", _factor),
            ("simplify", _simplify),
            ("collect", self._apply_collect),
        ]
        
        # Limit attempts according to max_transform_attempts
        # max_transform_attempts=0: original only [0:1]
        # max_transform_attempts=1: original+expand [0:2]
        # max_transform_attempts=4: all transforms [0:5]
        transformations = transformations[:self.max_transform_attempts + 1]
        
        last_error = None
        for method_name, transform_func in transformations:
            try:
                # Apply the transformation
                transformed_expr = transform_func(expr)
                
                # Try to encode
                tree_tokens = [None] * self.tree.total_nodes
                root_idx = self.tree.get_root_inorder_idx()
                self._fill_tree_balanced(transformed_expr, root_idx, tree_tokens)
                self._fill_identity(tree_tokens)
                
                # Success! Return the result
                # if method_name != "original":
                #     print(f"  [Encode] Success with {method_name}")
                return self._tree_tokens_to_sequence(tree_tokens)
                
            except Exception as e:
                last_error = e
                # Failure, try the next transform
                continue
        
        # All transforms failed, raise the last error
        raise last_error if last_error else ValueError(f"Unable to encode expression: {expr}")
    
    def _apply_collect(self, expr: sp.Expr) -> sp.Expr:
        """Apply collect to all variables"""
        free_symbols = expr.free_symbols
        if not free_symbols:
            return expr
        result = expr
        for var in free_symbols:
            result = sp.collect(result, var)
        return result
    
    def visualize(self, expr_str: str, filename: str) -> None:
        """
        Visualize the FEX tree structure
        
        Args:
            expr_str: expression string
            filename: output filename
        """
        # Fill the tree structure to obtain tree_tokens
        # Use evaluate=False to ensure the structure is not simplified
        expr = sp.sympify(expr_str, evaluate=False)
        tree_tokens = [None] * self.tree.total_nodes
        root_idx = self.tree.get_root_inorder_idx()
        self._fill_tree_balanced(expr, root_idx, tree_tokens)
        self._fill_identity(tree_tokens)
        
        # Call the visualization function
        self._visualize_tree(tree_tokens, expr_str, filename)
    
    def _visualize_tree(self, tree_tokens, expr_str, filename):
        """Internal visualization method (moved to a separate module)."""
        render_fex_tree(self, tree_tokens=tree_tokens, expr_str=expr_str, output_path=filename)
        print(f"  ✓ Tree structure saved to: {filename}")
    def _tree_tokens_to_sequence(self, tree_tokens: List) -> List[int]:
        """
        Convert the tree token list into a sequence
        
        Key: leaf nodes expand into 2 token positions
        """
        sequence = []
        for i, node in enumerate(self.tree.nodes):
            token = tree_tokens[i]
            
            if node['type'] == NODE_LEAF:
                # Leaf node: token is a (pos1, pos2) tuple
                if isinstance(token, tuple) and len(token) == 2:
                    pos1, pos2 = token
                    if pos1 not in self.equation_word2id:
                        raise ValueError(f"Token '{pos1}' is not in the vocabulary")
                    if pos2 not in self.equation_word2id:
                        raise ValueError(f"Token '{pos2}' is not in the vocabulary")
                    sequence.append(self.equation_word2id[pos1])
                    sequence.append(self.equation_word2id[pos2])
                else:
                    # Default padding: use PAD
                    sequence.append(self.equation_word2id['<PAD>'])
                    sequence.append(self.equation_word2id['<PAD>'])
            else:
                # Binary/Unary node: single token
                if token not in self.equation_word2id:
                    raise ValueError(f"Token '{token}' is not in the vocabulary")
                sequence.append(self.equation_word2id[token])
        
        return sequence
    
    def _fill_tree_balanced(self, expr: sp.Expr, inorder_idx: int, 
                           tree_tokens: List) -> None:
        """
        Recursively fill the tree using a balanced strategy (improved version: place operators deeper when possible)
        
        Core improvement: place unary functions as deep as possible so upper layers can remain identity passthroughs
        This leaves more room in upper layers for future expansion
        """
        node = self.tree.get_node_by_inorder_idx(inorder_idx)
        if node is None:
            return
            
        node_type = node['type']
        
        if node_type == NODE_LEAF:
            # Leaf node: encode as a 2-token format
            token = self._encode_leaf_2token(expr)
            tree_tokens[inorder_idx] = token
            return
        
        # Compute how many layers are still needed from the current expression to the leaves
        expr_depth = self._get_expr_depth(expr)
        # Compute how many layers remain from the current node to the leaves
        layers_to_leaf = self._get_layers_to_leaf(inorder_idx)
        
        if node_type == NODE_UNARY:
            # Check whether this is a unary function
            is_unary_func = (hasattr(expr, 'func') and expr.func.__name__ in self.sympy_func_map)
            is_pow2 = isinstance(expr, sp.Pow) and expr.args[1] == 2
            is_pow3 = isinstance(expr, sp.Pow) and expr.args[1] == 3
            is_inv = isinstance(expr, sp.Pow) and expr.args[1] == -1
            is_sqrt = isinstance(expr, sp.Pow) and expr.args[1] == sp.Rational(1, 2)
            
            is_unary_op = is_unary_func or is_pow2 or is_pow3 or is_inv or is_sqrt
            
            if is_unary_op:
                child_expr = expr.args[0]
                current_layer = node['layer']
                
                # Count how many Unary layers are available from the current layer to the leaves (including the current layer)
                unary_remaining = self._count_unary_layers_remaining(current_layer)
                
                # Compute how many Unary layers the expression needs
                # Use _get_expr_depth to estimate the required layers accurately (one Unary unit per 2 layers)
                child_depth = self._get_expr_depth(child_expr)
                unary_needed = (child_depth + 1) // 2
                
                # Leave at least one layer of space (similar to the previous logic 1 + count) to preserve centering behavior for shallow expressions
                if unary_needed < 1:
                    unary_needed = 1

                # Remaining space
                spare = unary_remaining - unary_needed
                
                # Midpoint strategy:
                # - Even spare (e.g. 2): leave spare/2 on both sides and place the operator in the middle
                # - Odd spare (e.g. 1): two middle candidates, choose the one closer to the leaves
                # So the passthrough count = (spare + 1) // 2 (ceiling division, closer to leaves)
                passthrough = (spare + 1) // 2
                
                if passthrough > 0:
                    # Passthrough, but mark it so we do not count it twice
                    # Wrap the expression and pass along the remaining passthrough count
                    tree_tokens[inorder_idx] = '<ID_Unary>'
                    child_idx = self._get_unary_child_idx(inorder_idx)
                    # Pass the tagged state: (original expression, remaining passthrough count - 1)
                    self._fill_tree_with_passthrough(expr, child_idx, tree_tokens, passthrough - 1)
                else:
                    # Place the unary operator
                    if is_unary_func:
                        tree_tokens[inorder_idx] = self.sympy_func_map[expr.func.__name__]
                    elif is_pow2:
                        tree_tokens[inorder_idx] = 'pow2'
                    elif is_pow3:
                        tree_tokens[inorder_idx] = 'pow3'
                    elif is_inv:
                        tree_tokens[inorder_idx] = 'inv'
                    elif is_sqrt:
                        tree_tokens[inorder_idx] = 'sqrt'
                    
                    child_idx = self._get_unary_child_idx(inorder_idx)
                    self._fill_tree_balanced(child_expr, child_idx, tree_tokens)
            else:
                # Not a unary operator, use identity passthrough
                tree_tokens[inorder_idx] = '<ID_Unary>'
                child_idx = self._get_unary_child_idx(inorder_idx)
                self._fill_tree_balanced(expr, child_idx, tree_tokens)
            return
        
        if node_type == NODE_BINARY:
            # Special check: is this a negation (Mul(-1, x))?
            is_negation = (isinstance(expr, sp.Mul) and 
                          len(expr.args) == 2 and 
                          expr.args[0] == -1)
            
            if is_negation:
                # Negative sign: represent as sub(0, x)
                tree_tokens[inorder_idx] = 'sub'
                left_idx, right_idx = self._get_binary_child_indices(inorder_idx)
                self._fill_tree_balanced(sp.Integer(0), left_idx, tree_tokens)
                self._fill_tree_balanced(expr.args[1], right_idx, tree_tokens)
                return
            
            if isinstance(expr, sp.Add):
                tree_tokens[inorder_idx] = 'add'
                args = list(expr.args)
                left_idx, right_idx = self._get_binary_child_indices(inorder_idx)
                
                if len(args) >= 2:
                    # Choose the split point intelligently: consider depth requirements and remaining space for both subtrees
                    split_point = self._find_optimal_split(args, inorder_idx)
                    
                    left_expr = sp.Add(*args[:split_point], evaluate=False) if split_point > 1 else args[0]
                    right_expr = sp.Add(*args[split_point:], evaluate=False) if len(args) - split_point > 1 else args[split_point]
                    self._fill_tree_balanced(left_expr, left_idx, tree_tokens)
                    self._fill_tree_balanced(right_expr, right_idx, tree_tokens)
                else:
                    self._fill_tree_balanced(args[0], left_idx, tree_tokens)
                    self._fill_tree_balanced(sp.Integer(0), right_idx, tree_tokens)
                    
            elif isinstance(expr, sp.Mul):
                # Standard multiplication
                tree_tokens[inorder_idx] = 'mul'
                args = list(expr.args)
                left_idx, right_idx = self._get_binary_child_indices(inorder_idx)
                
                if len(args) >= 2:
                    # Choose the split point intelligently: consider depth requirements and remaining space for both subtrees
                    split_point = self._find_optimal_split(args, inorder_idx)
                    
                    left_expr = sp.Mul(*args[:split_point], evaluate=False) if split_point > 1 else args[0]
                    right_expr = sp.Mul(*args[split_point:], evaluate=False) if len(args) - split_point > 1 else args[split_point]
                    self._fill_tree_balanced(left_expr, left_idx, tree_tokens)
                    self._fill_tree_balanced(right_expr, right_idx, tree_tokens)
                else:
                    self._fill_tree_balanced(args[0], left_idx, tree_tokens)
                    self._fill_tree_balanced(sp.Integer(1), right_idx, tree_tokens)
                    
            elif isinstance(expr, sp.Pow):
                # Pow expression: base ** exponent
                base = expr.args[0]
                exponent = expr.args[1]
                
                # Check whether this is a special exponent (2, 3, -1, 1/2)
                # These should be handled in the Unary layer, not the Binary layer
                if exponent in [2, 3, -1, sp.Rational(1, 2)]:
                    # Special exponents should be passed through to the Unary layer for handling
                    tree_tokens[inorder_idx] = '<ID_Binary>'
                    left_idx, right_idx = self._get_binary_child_indices(inorder_idx)
                    self._fill_tree_balanced(expr, left_idx, tree_tokens)
                    self._fill_identity_subtree(right_idx, tree_tokens)
                else:
                    # General Pow: decompose into base and exponent
                    tree_tokens[inorder_idx] = 'pow'
                    left_idx, right_idx = self._get_binary_child_indices(inorder_idx)
                    self._fill_tree_balanced(base, left_idx, tree_tokens)
                    self._fill_tree_balanced(exponent, right_idx, tree_tokens)
            else:
                # Other cases: use identity passthrough
                tree_tokens[inorder_idx] = '<ID_Binary>'
                left_idx, right_idx = self._get_binary_child_indices(inorder_idx)
                self._fill_tree_balanced(expr, left_idx, tree_tokens)
                self._fill_identity_subtree(right_idx, tree_tokens)
    
    def _fill_tree_with_passthrough(self, expr: sp.Expr, inorder_idx: int, 
                                    tree_tokens: List, remaining_passthrough: int) -> None:
        """
        Filling method with passthrough counting - used for midpoint placement of unary operators
        
        remaining_passthrough: how many more Unary layers need passthrough
        """
        node = self.tree.get_node_by_inorder_idx(inorder_idx)
        if node is None:
            return
            
        node_type = node['type']
        
        if node_type == NODE_BINARY:
            # Direct passthrough for Binary layers
            tree_tokens[inorder_idx] = '<ID_Binary>'
            left_idx, right_idx = self._get_binary_child_indices(inorder_idx)
            self._fill_tree_with_passthrough(expr, left_idx, tree_tokens, remaining_passthrough)
            self._fill_identity_subtree(right_idx, tree_tokens)
            return
        
        if node_type == NODE_UNARY:
            if remaining_passthrough > 0:
                # Still need to passthrough
                tree_tokens[inorder_idx] = '<ID_Unary>'
                child_idx = self._get_unary_child_idx(inorder_idx)
                self._fill_tree_with_passthrough(expr, child_idx, tree_tokens, remaining_passthrough - 1)
            else:
                # Passthrough completed, now place the unary operator
                is_unary_func = (hasattr(expr, 'func') and expr.func.__name__ in self.sympy_func_map)
                is_pow2 = isinstance(expr, sp.Pow) and expr.args[1] == 2
                is_pow3 = isinstance(expr, sp.Pow) and expr.args[1] == 3
                is_inv = isinstance(expr, sp.Pow) and expr.args[1] == -1
                is_sqrt = isinstance(expr, sp.Pow) and expr.args[1] == sp.Rational(1, 2)
                
                child_expr = expr.args[0]
                
                if is_unary_func:
                    tree_tokens[inorder_idx] = self.sympy_func_map[expr.func.__name__]
                elif is_pow2:
                    tree_tokens[inorder_idx] = 'pow2'
                elif is_pow3:
                    tree_tokens[inorder_idx] = 'pow3'
                elif is_inv:
                    tree_tokens[inorder_idx] = 'inv'
                elif is_sqrt:
                    tree_tokens[inorder_idx] = 'sqrt'
                else:
                    tree_tokens[inorder_idx] = '<ID_Unary>'
                
                child_idx = self._get_unary_child_idx(inorder_idx)
                self._fill_tree_balanced(child_expr, child_idx, tree_tokens)
            return
        
        if node_type == NODE_LEAF:
            # Should not reach here, but handle it defensively
            token = self._encode_leaf_2token(expr)
            tree_tokens[inorder_idx] = token
            return
    
    def _encode_leaf_2token(self, expr: sp.Expr) -> Tuple[str, str]:
        """
        Encode an expression as a 2-token leaf format
        
        Returns:
            (pos1_token, pos2_token) tuple
            - Constant: (N00-N99, E-2/E-1/E0)
            - Variable: (+/-, x_0...x_9)
        """
        if isinstance(expr, sp.Symbol):
            # Variable leaf: [sign, variable name]
            var_name = str(expr)
            sign = '+'
            
            # Check whether this is a negative variable (e.g. -x)
            # In SymPy, -x is represented as Mul(-1, x), so we only handle positive variables here
            
            if var_name.startswith('x'):
                # Extract the variable index
                try:
                    var_idx = var_name.replace('x', '').replace('_', '')
                    idx = int(var_idx) if var_idx else 0
                    return (sign, f'x_{idx}')
                except:
                    return (sign, 'x_0')
            return (sign, 'x_0')
            
        elif isinstance(expr, (sp.Integer, int)):
            # Integer constant
            value = int(expr)
            return self._encode_constant_2token(float(value))
            
        elif isinstance(expr, (sp.Float, sp.Rational, float)):
            # Floating-point/rational constant
            value = float(expr)
            return self._encode_constant_2token(value)
            
        else:
            raise ValueError(f"Unable to encode leaf expression: {expr}")
    
    def _find_optimal_split(self, args: List[sp.Expr], inorder_idx: int) -> int:
        """
        Choose the optimal split point for a polynomial expression intelligently
        
        Strategy:
        1. Compute each possible split point (1 to len(args)-1)
        2. For each split, evaluate the depth requirements of the left and right subtrees
        3. Measure how well the subtrees fit the available depth
        4. Choose the split point with the best fit
        
        Args:
            args: expression argument list (e.g. [a, b, c, d] for a+b+c+d)
            inorder_idx: inorder index of the current Binary node
            
        Returns:
            optimal split index (1 to len(args)-1)
        """
        n = len(args)
        if n <= 2:
            return 1  # With only 2 terms, the only split is [0] and [1]
        
        # Get the available depth for the left and right subtrees
        left_idx, right_idx = self._get_binary_child_indices(inorder_idx)
        left_available = self._get_layers_to_leaf(left_idx)
        right_available = self._get_layers_to_leaf(right_idx)
        
        best_split = n // 2  # Default to a middle split
        best_score = float('inf')
        
        # Try every possible split point
        for split in range(1, n):
            # Compute the depth requirements of the left and right subexpressions
            if split == 1:
                left_depth = self._get_expr_depth(args[0])
            else:
                # For polynomial addition/multiplication, depth needs an extra +2 (Binary + Unary layers)
                left_depth = 2 + max(self._get_expr_depth(arg) for arg in args[:split])
            
            if split == n - 1:
                right_depth = self._get_expr_depth(args[split])
            else:
                right_depth = 2 + max(self._get_expr_depth(arg) for arg in args[split:])
            
            # Compute mismatch (how much the depth requirement exceeds available space)
            left_mismatch = max(0, left_depth - left_available)
            right_mismatch = max(0, right_depth - right_available)
            
            # Total mismatch (penalize the overflow)
            total_mismatch = left_mismatch + right_mismatch
            
            # Extra reward: try to balance the number of terms on both sides (avoid extreme imbalance)
            balance_penalty = abs(split - (n - split)) * 0.1
            
            score = total_mismatch + balance_penalty
            
            # If both sides fit, choose the most balanced split
            if score < best_score:
                best_score = score
                best_split = split
        
        return best_split
    
    def _get_expr_depth(self, expr: sp.Expr) -> int:
        """
        Compute the minimum number of layers required for an expression in the FEX tree
        
        - Leaf (variable/constant): 0 layers
        - Unary function: 1 + child expression depth (needs 1 Unary + 1 Binary layer)
        - Binary function: 2 + max(left depth, right depth) (needs 1 Binary + 1 Unary layer)
        
        Note: in the FEX structure, Binary and Unary layers alternate, so the depth must be computed accordingly
        """
        if isinstance(expr, (sp.Symbol, sp.Integer, sp.Float, sp.Rational, int, float)):
            return 0
        
        # Unary functions (sin, cos, exp, log, sqrt, pow2, pow3, inv)
        is_unary = False
        if hasattr(expr, 'func') and expr.func.__name__ in self.sympy_func_map:
            is_unary = True
        elif isinstance(expr, sp.Pow) and expr.args[1] in [2, 3, -1]:
            is_unary = True
        
        if is_unary:
            child_depth = self._get_expr_depth(expr.args[0])
            # A unary operation passes through one Unary node, then the child expression sits in the left subtree of a Binary node
            # In FEX, Unary is followed by Binary, so this needs 2 layers (Unary + Binary)
            return 2 + child_depth
        
        # Binary functions (add, mul, sub, div)
        if isinstance(expr, (sp.Add, sp.Mul)):
            args = list(expr.args)
            if len(args) >= 2:
                # Compute the depth of the left and right subtrees
                mid = len(args) // 2
                left_expr = sp.Add(*args[:mid], evaluate=False) if isinstance(expr, sp.Add) else sp.Mul(*args[:mid], evaluate=False)
                right_expr = sp.Add(*args[mid:], evaluate=False) if isinstance(expr, sp.Add) else sp.Mul(*args[mid:], evaluate=False)
                if mid == 1:
                    left_expr = args[0]
                if len(args) - mid == 1:
                    right_expr = args[mid]
                    
                left_depth = self._get_expr_depth(left_expr)
                right_depth = self._get_expr_depth(right_expr)
                # A binary operation needs 1 Binary + 2 Unary layers, then the respective subtrees
                return 2 + max(left_depth, right_depth)
            else:
                return 2 + self._get_expr_depth(args[0])
        
        # Other cases
        return 0
    
    def _get_layers_to_leaf(self, inorder_idx: int) -> int:
        """
        Compute how many layers remain from the current node to the leaf layer
        """
        node = self.tree.get_node_by_inorder_idx(inorder_idx)
        if node is None:
            return 0
        
        # The leaf layer is 2*depth - 1 (depth=3 gives 5)
        leaf_layer = 2 * self.tree.depth - 1
        current_layer = node['layer']
        return leaf_layer - current_layer
    
    def _count_unary_layers_remaining(self, current_layer: int) -> int:
        """
        Compute how many Unary layers are available from the current layer to the leaf layer (including the current layer if it is Unary)
        
        In the FEX structure, Unary layers are odd-numbered layers: 1, 3, 5, ...
        For depth=4: layer 0(B), 1(U), 2(B), 3(U), 4(B), 5(U), 6(B), 7(Leaf)
        """
        leaf_layer = 2 * self.tree.depth - 1  # depth=4 gives 7
        count = 0
        for layer in range(current_layer, leaf_layer):
            if layer % 2 == 1:  # Unary layers are odd-numbered
                count += 1
        return count
    
    def _count_unary_ops_in_expr(self, expr: sp.Expr) -> int:
        """
        Compute how many Unary operations are still needed in the expression
        
        - Leaf (variable/constant): 0
        - Unary function: 1 + the number of Unary operations needed by the child expression
        - Binary function: max(left child needs, right child needs), because the Unary operations of the two subtrees are independent
        """
        if isinstance(expr, (sp.Symbol, sp.Integer, sp.Float, sp.Rational, int, float)):
            return 0
        
        # Check whether this is a unary operation
        is_unary = False
        if hasattr(expr, 'func') and expr.func.__name__ in self.sympy_func_map:
            is_unary = True
        elif isinstance(expr, sp.Pow) and expr.args[1] in [2, 3, -1, sp.Rational(1, 2)]:
            is_unary = True
        elif isinstance(expr, sp.Mul) and len(expr.args) == 2 and expr.args[0] == -1:
            is_unary = True  # neg
        
        if is_unary:
            return 1 + self._count_unary_ops_in_expr(expr.args[0])
        
        # Binary operation
        if isinstance(expr, (sp.Add, sp.Mul)):
            args = list(expr.args)
            if len(args) >= 2:
                mid = len(args) // 2
                left_expr = sp.Add(*args[:mid], evaluate=False) if isinstance(expr, sp.Add) else sp.Mul(*args[:mid], evaluate=False)
                right_expr = sp.Add(*args[mid:], evaluate=False) if isinstance(expr, sp.Add) else sp.Mul(*args[mid:], evaluate=False)
                if mid == 1:
                    left_expr = args[0]
                if len(args) - mid == 1:
                    right_expr = args[mid]
                
                return max(self._count_unary_ops_in_expr(left_expr), 
                          self._count_unary_ops_in_expr(right_expr))
            else:
                return self._count_unary_ops_in_expr(args[0])
        
        return 0
    
    def _encode_constant_2token(self, value: float) -> Tuple[str, str]:
        """
        Encode a floating-point number into a 2-token format: (mantissa, exponent)
        Use the environment's float_encoder to determine the available exponent range and mantissa width
        
        Formula: value ≈ mantissa_int × 10^exponent
        - mantissa_int ∈ [0, max_token-1] (max_token is determined by base)
        - exponent ∈ [-max_exponent, max_exponent] (determined by env.max_exponent_prefactor)
        
        Example (assuming base=1, max_exponent=3):
        - 0.05 → ('N5', 'E-2') = 5 × 10^(-2)
        - 3.1 → ('N31', 'E-1') = 31 × 10^(-1)
        - 100 → ('N100', 'E0') = 100 × 10^0
        
        Note: base is the formatting width (used for N tokens) and is defined by float_encoder
        Special case: when float_precision > 1, the actual number of significant digits is float_precision,
                 but the encoding width is base=(float_precision+1)//mantissa_len,
                 and the extra trailing digits are padded with zeros
        """
        # Get base and max_token from float_encoder
        base = self.float_encoder.base
        max_token = self.float_encoder.max_token  # 10^base
        float_precision = self.float_encoder.float_precision
        
        # Actual significant-digit range (used for matching)
        effective_max = 10 ** float_precision
        
        if value == 0:
            return (f'N{0:0{base}d}', 'E0')
        
        abs_val = abs(value)
        
        # Try different exponents to find the best representation
        best_mantissa = 0
        best_exponent = 0
        best_error = float('inf')
        
        # Use the environment's max_exponent range
        for exp in range(-self.max_exponent, self.max_exponent + 1):
            mantissa_float = abs_val / (10 ** exp)
            mantissa_int = int(round(mantissa_float))
            
            # Restrict to [1, effective_max-1] (based on the actual significant digits)
            if mantissa_int >= effective_max or mantissa_int < 1:
                continue
                
            # Compute reconstruction error
            reconstructed = mantissa_int * (10 ** exp)
            error = abs(reconstructed - abs_val)
            
            if error < best_error:
                best_error = error
                best_mantissa = mantissa_int
                best_exponent = exp
        
        # If all exponents are out of range, use a truncation strategy
        if best_mantissa == 0:
            min_val = 10 ** (-self.max_exponent)
            max_val = (effective_max - 1) * (10 ** self.max_exponent)
            if abs_val < min_val:
                best_mantissa = 1
                best_exponent = -self.max_exponent
            else:  # abs_val > max_val
                best_mantissa = effective_max - 1
                best_exponent = self.max_exponent
        
        # Expand to the base width (pad zeros at the end)
        # Example: float_precision=2, base=3 gives 92 → 920
        mantissa_encoded = best_mantissa * (10 ** (base - float_precision))
        best_exponent -= (base - float_precision)
        
        # Build the token: if negative constants are enabled and the value is negative, use a negative-constant token
        assert self.float_encoder.use_negative_constants == True, "FixedTreeEncoder currently only supports a float_encoder with use_negative_constants=True"
        if value < 0:
            mantissa_token = f'-N{mantissa_encoded:0{base}d}'
        else:
            mantissa_token = f'N{mantissa_encoded:0{base}d}'
        
        exponent_token = f'E{best_exponent}' if best_exponent >= 0 else f'E{best_exponent}'
        
        return (mantissa_token, exponent_token)
    
    def _get_unary_child_idx(self, unary_inorder_idx: int) -> int:
        """Get the inorder index of a Unary node's child (a Binary node)"""
        return self.tree._unary_child_cache.get(unary_inorder_idx, -1)

    def _get_binary_child_indices(self, binary_inorder_idx: int) -> Tuple[int, int]:
        """Get the inorder indices of a Binary node's two children"""
        return self.tree._binary_child_cache.get(binary_inorder_idx, (-1, -1))
    
    def _fill_identity(self, tree_tokens: List) -> None:
        """Fill all empty slots with identity operations"""
        for i, token in enumerate(tree_tokens):
            if token is None:
                node = self.tree.get_node_by_inorder_idx(i)
                if node['type'] == NODE_BINARY:
                    tree_tokens[i] = '<ID_Binary>'
                elif node['type'] == NODE_UNARY:
                    tree_tokens[i] = '<ID_Unary>'
                elif node['type'] == NODE_LEAF:
                    # Leaf padding: use the PAD token
                    tree_tokens[i] = ('<PAD>', '<PAD>')
    
    def _fill_identity_subtree(self, inorder_idx: int, tree_tokens: List) -> None:
        """Fill a subtree with identity operations"""
        node = self.tree.get_node_by_inorder_idx(inorder_idx)
        if node is None:
            return
            
        if node['type'] == NODE_BINARY:
            tree_tokens[inorder_idx] = '<ID_Binary>'
            left_idx, right_idx = self._get_binary_child_indices(inorder_idx)
            self._fill_identity_subtree(left_idx, tree_tokens)
            self._fill_identity_subtree(right_idx, tree_tokens)
        elif node['type'] == NODE_UNARY:
            tree_tokens[inorder_idx] = '<ID_Unary>'
            child_idx = self._get_unary_child_idx(inorder_idx)
            self._fill_identity_subtree(child_idx, tree_tokens)
        elif node['type'] == NODE_LEAF:
            tree_tokens[inorder_idx] = ('<PAD>', '<PAD>')  # Padding
    
    def _init_fast_cache(self):
        """Initialize fast decode cache via cache manager."""
        _ = self._cache_manager.get_fast_decode_cache()

    def decode(self, token_sequence: List[int]) -> sp.Expr:
        """
        Fast decode implementation v2 (Iterative + Cached Objects + Evaluate=False)
        Uses cache manager for all cached data.
        """
        cache = self._cache_manager.get_fast_decode_cache()
        execution_order = cache['execution_order']
        leaf_cache = cache['leaf_pair_cache']
        unary_funcs = cache['unary_op_funcs']
        bin_funcs = cache['bin_op_funcs']

        results = [None] * len(self.tree.nodes)
        seq_len = len(token_sequence)

        token_len = len(unary_funcs)  # Assumes all arrays same length
        default_val = sp.Integer(0)

        for step in execution_order:
            idx = step['idx']
            node_type = step['type']
            seq_idx = step['seq_idx']

            res = default_val

            if seq_idx < seq_len:
                if node_type == NODE_LEAF:
                    if seq_idx + 1 < seq_len:
                        id1 = token_sequence[seq_idx]
                        id2 = token_sequence[seq_idx+1]
                        # Direct Pair Lookup (No SymPy creation overhead)
                        res = leaf_cache.get((id1, id2), default_val)

                elif node_type == NODE_UNARY:
                    child_idx = step['children'][0]
                    # Direct list access is faster than get
                    child_val = results[child_idx]

                    tid = token_sequence[seq_idx]
                    if tid < token_len:
                        func = unary_funcs[tid]
                        if func:
                            res = func(child_val)
                        else:
                            res = child_val  # Identity/Fallback
                    else:
                        res = child_val

                elif node_type == NODE_BINARY:
                    left_idx, right_idx = step['children']
                    l_val = results[left_idx]
                    r_val = results[right_idx]

                    tid = token_sequence[seq_idx]
                    if tid < token_len:
                        func = bin_funcs[tid]
                        if func:
                            res = func(l_val, r_val)
                        else:
                            res = l_val  # Unknown -> Left
                    else:
                        res = l_val

            results[idx] = res

        root_idx = self.tree.get_root_inorder_idx()
        return results[root_idx]

    def decode_to_node(self, token_sequence: List[int]) -> Any:
        """
        Decode a token sequence directly into a Node tree object, skipping SymPy conversion
        """
        from symbolicregression.envs.generators import Node
        
        if not hasattr(self, '_fast_cache_initialized'):
            self._init_fast_cache()
            
        results = [None] * len(self.tree.nodes)
        seq_len = len(token_sequence)
        params = self.env.params 
        
        # Local lookup
        leaf_cache_val = {} # Cache for leaf values (str or float)
        # Pre-populate the leaf cache with Node-friendly values if possible
        # Or just compute values on the fly. Since we cached _leaf_pair_cache with SymPy, we cannot reuse it directly
        # unless we extract values.
        
        # Do we need a new cache for Node values, or just parse on the fly?
        # Parsing on the fly is fast enough if we do not do heavy work.
        # But we can use _token_cache.

        for step in self._execution_order:
            idx = step['idx']
            node_type = step['type']
            seq_idx = step['seq_idx']
            
            res = None
            
            if seq_idx < seq_len:
                if node_type == NODE_LEAF:
                    if seq_idx + 1 < seq_len:
                        id1 = token_sequence[seq_idx]
                        id2 = token_sequence[seq_idx+1]
                        
                        # Use _token_cache to reconstruct value string
                                                # Use _token_cache to reconstruct the value string
                        if id1 < len(self._token_cache) and id2 < len(self._token_cache):
                            info1 = self._token_cache[id1]
                            info2 = self._token_cache[id2]
                            
                            if info1 and info2:
                                if info1.get('leaf_pos1_type') == 'sign' and info2.get('leaf_pos2_type') == 'var':
                                    # Variable
                                    var_name = self.equation_id2word[id2]
                                    sign = info1['leaf_pos1_val']
                                    if sign == -1:
                                        # Node('sub', [Node(0), Node(x)]) or Node('mul', [-1, x])?
                                        # Or just a leaf value like "-x_0"? No, the generator expects "x_0".
                                        # The generator usually handles negation as "sub(0, x)" or "mul(-1, x)"
                                        # But let's check how the generator produces trees.
                                        # Usually unary 'neg' is used, but here it is encoded in a leaf.
                                        # Use the Mul(-1, x) equivalent: Node("mul", params, [Node("-1", params), Node(var_name, params)])
                                        # Or simpler: Node(var_name) and wrap it in neg if sign is -?
                                        # In this branch we return a Node.
                                        var_node = Node(var_name, params)
                                        if sign == -1:
                                            # Wrap in mul -1
                                            neg_node = Node("-1", params)
                                            res = Node("mul", params, children=[neg_node, var_node])
                                        else:
                                            res = var_node
                                            
                                elif info1.get('leaf_pos1_type') == 'mantissa' and info2.get('leaf_pos2_type') == 'exponent':
                                    # Constant
                                    m = info1['leaf_pos1_val']
                                    e = info2['leaf_pos2_val']
                                    if e >= 0:
                                        val = m * (10 ** e)
                                        s_val = str(val)
                                    else:
                                        val = float(m) * (10.0 ** e)
                                        if abs(val - round(val)) < 1e-10:
                                            s_val = str(int(round(val)))
                                        else:
                                            s_val = str(val)
                                    res = Node(s_val, params)
                    if res is None:
                        res = Node("0", params) # Fallback

                elif node_type == NODE_UNARY:
                    child_idx = step['children'][0]
                    child_node = results[child_idx]
                    if child_node is None: child_node = Node("0", params)
                    
                    tid = token_sequence[seq_idx]
                    if tid < len(self._token_cache):
                        info = self._token_cache[tid]
                        # Check identity
                        # Check identity
                        if info and info.get('unary_op') == _identity:
                            res = child_node
                        elif info and 'unary_op' in info:
                            # Operation name
                            op_name = self.equation_id2word[tid]
                            # Special mappings from SymPy to generator names if needed
                            # Here we used 'sin', 'cos', etc.
                            # 'neg' -> 'sub(0, x)' or 'mul(-1, x)'?
                            # 'inv' -> 'div(1, x)'? or 'inv'?
                            # The generator has 'inv', 'neg', 'abs', etc. in operators_real?
                            # operators_real in the generator has: abs, inv, sqrt, log, exp, sin...
                            # It does NOT have 'neg'. 'sub' is binary.
                            # So 'neg' should be converted.
                                                    # Check identity
                            
                            if op_name == 'neg':
                                neg_one = Node("-1", params)
                                res = Node("mul", params, children=[neg_one, child_node])
                            elif op_name == 'pow2':
                                res = Node("pow2", params, children=[child_node])
                            elif op_name == 'pow3':
                                res = Node("pow3", params, children=[child_node])
                            else:
                                res = Node(op_name, params, children=[child_node])
                        else:
                            res = child_node
                    else:
                        res = child_node

                elif node_type == NODE_BINARY:
                    left_idx, right_idx = step['children']
                    l_node = results[left_idx]
                    r_node = results[right_idx]
                    if l_node is None: l_node = Node("0", params)
                    if r_node is None: r_node = Node("0", params)
                    
                    tid = token_sequence[seq_idx]
                    if tid < len(self._token_cache):
                        info = self._token_cache[tid]
                        if info and info.get('binary_op_name') == 'id':
                            res = l_node
                        elif info and 'binary_op_name' in info:
                            op_name = info['binary_op_name']
                            res = Node(op_name, params, children=[l_node, r_node])
                        else:
                            res = l_node
                    else:
                        res = l_node
            
            results[idx] = res

        root_idx = self.tree.get_root_inorder_idx()
        return results[root_idx]

    def _legacy_decode(self, token_sequence: List[int]) -> sp.Expr:
        """
        Decode a token sequence back into a SymPy expression
        
        Args:
            token_sequence: encoded token index sequence
            
        Returns:
            decoded SymPy expression
            
        Decoding steps:
        1. Convert the sequence back to a tree token list (merge leaves into tuples)
        2. Recursively build the expression from the root node
        3. Handle identity passthrough (Binary ID keeps only the left subtree, Unary ID passes through)
        """
        # Step 1: sequence -> tree tokens (merge leaves into 2-token tuples)
        tree_tokens = self._sequence_to_tree_tokens(token_sequence)
        
        # Step 2: recursively build the expression from the root node
        
        try:
            root_idx = self.tree.get_root_inorder_idx()
            expr = self._decode_subtree(root_idx, tree_tokens)
        except ValueError as e:
            return None
        
        return expr
    
    def _sequence_to_tree_tokens(self, token_sequence: List[int]) -> List:
        """
        Convert a token sequence back into a tree token list
        
        Key: leaf nodes occupy 2 positions and must be merged back into tuples
        """
        tree_tokens = []
        seq_idx = 0
        
        for node in self.tree.nodes:
            if node['type'] == NODE_LEAF:
                # Leaf: read 2 tokens and merge them into a tuple
                if seq_idx + 1 < len(token_sequence):
                    pos1_id = token_sequence[seq_idx]
                    pos2_id = token_sequence[seq_idx + 1]
                    pos1 = self.equation_id2word.get(pos1_id, '<UNK>')
                    pos2 = self.equation_id2word.get(pos2_id, '<UNK>')
                    tree_tokens.append((pos1, pos2))
                    seq_idx += 2
                else:
                    tree_tokens.append(('<PAD>', '<PAD>'))
                    seq_idx += 2
            else:
                # Binary/Unary: single token
                if seq_idx < len(token_sequence):
                    token_id = token_sequence[seq_idx]
                    token = self.equation_id2word.get(token_id, '<UNK>')
                    tree_tokens.append(token)
                    seq_idx += 1
                else:
                    tree_tokens.append('<PAD>')
                    seq_idx += 1
        
        return tree_tokens
    
    def _decode_subtree(self, inorder_idx: int, tree_tokens: List) -> sp.Expr:
        """
        Recursively decode a subtree into a SymPy expression
        
        Args:
            inorder_idx: inorder index of the current node
            tree_tokens: tree token list
            
        Returns:
            SymPy expression for the current subtree
        """
        node = self.tree.get_node_by_inorder_idx(inorder_idx)
        if node is None:
            return sp.Integer(0)
        
        token = tree_tokens[inorder_idx]
        node_type = node['type']
        
        # Handle leaf nodes
        if node_type == NODE_LEAF:
            return self._decode_leaf(token)
        
        # Handle Unary nodes
        if node_type == NODE_UNARY:
            child_idx = self._get_unary_child_idx(inorder_idx)
            child_expr = self._decode_subtree(child_idx, tree_tokens)
            
            # Identity passthrough
            if token == '<ID_Unary>':
                return child_expr
            
            # Apply the unary operation
            return self._apply_unary_op(token, child_expr)
        
        # Handle Binary nodes
        if node_type == NODE_BINARY:
            left_idx, right_idx = self._get_binary_child_indices(inorder_idx)
            left_expr = self._decode_subtree(left_idx, tree_tokens)
            right_expr = self._decode_subtree(right_idx, tree_tokens)
            
            # Identity passthrough (keep only the left subtree)
            if token == '<ID_Binary>':
                return left_expr
            
            # Apply the binary operation
            return self._apply_binary_op(token, left_expr, right_expr)
        
        return sp.Integer(0)
    
    def _decode_leaf(self, token: Tuple[str, str]) -> sp.Expr:
        """
        Decode the 2-token format of a leaf node
        
        Args:
            token: (pos1, pos2) tuple
            
        Returns:
            SymPy expression (constant or variable)
        """
        if not isinstance(token, tuple) or len(token) != 2:
            raise ValueError(f"Invalid leaf token tuple: {token}")
        
        pos1, pos2 = token
        
        # Padding node
        if pos1 == '<PAD>' or pos2 == '<PAD>':
            return sp.Integer(0)
        
        # Variable leaf: [+/-, x_0...x_9]
        if pos1 in ['+', '-'] and pos2.startswith('x_'):
            # Extract the variable name
            var_name = pos2
            var = sp.Symbol(var_name)
            
            # Apply the sign
            if pos1 == '-':
                return -var
            else:
                return var
        
        # Constant leaf: [N00-N99 or -N00-N99, E-X...EX]
        if (pos1.startswith('N') or pos1.startswith('-N')) and pos2.startswith('E'):
            try:
                # Check whether this is a negative constant token
                is_negative = pos1.startswith('-N')
                
                # Extract the mantissa
                if is_negative:
                    mantissa_str = pos1[2:]  # remove '-N'
                else:
                    mantissa_str = pos1[1:]  # remove 'N'
                mantissa_int = int(mantissa_str)
                
                # Extract the exponent
                exponent_str = pos2[1:]  # remove 'E'
                exponent_int = int(exponent_str)
                
                # Compute the constant value: mantissa_int × 10^exponent
                value = mantissa_int * (10 ** exponent_int)
                
                # Apply the sign
                if is_negative:
                    value = -value
                
                # Choose an appropriate SymPy type based on the magnitude
                if abs(value - int(value)) < 1e-10:
                    return sp.Integer(int(value))
                else:
                    return sp.Float(value)
            except:
                print(f"Unable to decode constant leaf: {token}")
                raise ValueError(f"Failed to decode constant leaf: {token}")
        
        # Unknown format
        print(f"Unknown leaf format: {token}")
        raise ValueError(f"Unknown leaf token pair: {token}")
    
    def _apply_unary_op(self, op_token: str, arg: sp.Expr) -> sp.Expr:
        """Apply a unary operation"""
        # Validate that this is a legal unary operator
        if op_token not in UNARY_OPS:
            # Unknown operation, return the original expression
            return arg
        
        # Use a dictionary map for unified handling
        func_map = {
            'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
            'arcsin': sp.asin, 'arccos': sp.acos, 'arctan': sp.atan,
            'exp': sp.exp, 'log': sp.log, 
            'sqrt': _sqrt,  # preserve **0.5 format
            'abs': sp.Abs, 'neg': _neg,
            'inv': _inv, 'pow2': _pow2, 'pow3': _pow3,
            '<ID_Unary>': _identity  # identity passthrough
        }
        
        if op_token in func_map:
            return func_map[op_token](arg)
        else:
            return arg
    
    def _apply_binary_op(self, op_token: str, left: sp.Expr, right: sp.Expr) -> sp.Expr:
        """Apply a binary operation"""
        if op_token == 'add':
            return left + right
        elif op_token == 'sub':
            return left - right
        elif op_token == 'mul':
            return left * right
        elif op_token == 'div':
            # Avoid division by zero
            return left / right
        elif op_token == 'pow':
            return left ** right
        else:
            # Unknown operation, return the left expression
            return left

    def _assign_positions(self, node_idx, x_start, x_end, node_positions):
        """Recursively assign node positions so the tree is symmetric"""
        node = self.tree.nodes[node_idx]
        layer = node['layer']
        y_pos = -layer * 1.5
        
        if node['type'] == NODE_LEAF:
            x_pos = (x_start + x_end) / 2
            node_positions[node_idx] = (x_pos, y_pos)
            return x_pos
        elif node['type'] == NODE_UNARY:
            child_layer = layer + 1
            child_index_in_layer = node['index_in_layer']
            # Find the child node index
            for j, n in enumerate(self.tree.nodes):
                if n['layer'] == child_layer and n['index_in_layer'] == child_index_in_layer:
                    child_x = self._assign_positions(j, x_start, x_end, node_positions)
                    node_positions[node_idx] = (child_x, y_pos)
                    return child_x
        elif node['type'] == NODE_BINARY:
            child_layer = layer + 1
            left_index = 2 * node['index_in_layer']
            right_index = 2 * node['index_in_layer'] + 1
            mid = (x_start + x_end) / 2
            # Find the left child
            left_x = None
            for j, n in enumerate(self.tree.nodes):
                if n['layer'] == child_layer and n['index_in_layer'] == left_index:
                    left_x = self._assign_positions(j, x_start, mid, node_positions)
                    break
            # Find the right child
            right_x = None
            for j, n in enumerate(self.tree.nodes):
                if n['layer'] == child_layer and n['index_in_layer'] == right_index:
                    right_x = self._assign_positions(j, mid, x_end, node_positions)
                    break
            if left_x is not None and right_x is not None:
                node_x = (left_x + right_x) / 2
            else:
                node_x = mid
            node_positions[node_idx] = (node_x, y_pos)
            return node_x
        return (x_start + x_end) / 2

    def _get_seq_pos_cache(self):
        """Get sequence position cache via cache manager."""
        return self._cache_manager.get_seq_pos_cache()

    def visualize_relaxed_subtree(
        self,
        topk_probs,
        topk_indices,
        X=None,
        node_idx=None,
        node_layer=None,
        node_index_in_layer=None,
        max_depth=None,
        output_path=None,
        topk_display=3,
    ):
        """Visualization interface (implementation moved to symbolicregression.visualization)."""
        if not torch.is_tensor(topk_probs):
            topk_probs = torch.as_tensor(topk_probs, dtype=torch.float32)
        if not torch.is_tensor(topk_indices):
            topk_indices = torch.as_tensor(topk_indices, dtype=torch.long)
        if topk_probs.dim() != 2 or topk_indices.dim() != 2:
            raise ValueError("visualize_relaxed_subtree expects (seq_len, K) tensors.")
        if topk_probs.shape != topk_indices.shape:
            raise ValueError("topk_probs and topk_indices must share the same shape.")

        if node_idx is None:
            if node_layer is not None and node_index_in_layer is not None:
                node_idx = self.tree.get_inorder_idx_from_layer(node_layer, node_index_in_layer)
            else:
                node_idx = self.tree.get_root_inorder_idx()

        seq_len = topk_probs.size(0)
        logits = topk_probs.new_full((seq_len, self.env.n_words), float('-inf'))
        safe_probs = torch.clamp(topk_probs, min=1e-8)
        logits.scatter_(1, topk_indices, torch.log(safe_probs))

        if output_path is None:
            output_path = Path("imgs/relaxed_subtree.png")
        else:
            output_path = Path(output_path)

        render_relaxed_subtree(
            fex_env=self.env,
            logits=logits.detach().cpu(),
            active_seq_len=min(seq_len, self.tree.sequence_length + 2),
            active_positions=None,
            subtree_root=node_idx,
            output_path=output_path,
            topk=topk_display,
            title=f"Relaxed Subtree root={node_idx}",
        )
        print(f"[FEX] Relaxed subtree saved to: {output_path}")
        return output_path

    def _ensure_vocab_lookup_tables(self, device):
        self._inner_loop_executor._ensure_vocab_lookup_tables(device)

    def _get_sequence_position(self, inorder_idx):

        """
        Convert an inorder index to a sequence position (accounting for the 2-token expansion of leaf nodes)
        
        Key: leaf nodes occupy 2 positions!
        
        Example:
        Inorder: [L0, B0, L1, U0, ...]
        Sequence: [L0_pos1, L0_pos2, B0, L1_pos1, L1_pos2, U0, ...]
        
        Args:
            inorder_idx: node inorder index
        
        Returns:
            start position of the node in the sequence
        """
        seq_pos = 0
        for i in range(inorder_idx):
            node = self.tree.nodes[i]
            if node['type'] == NODE_LEAF:
                seq_pos += 2  # Leaves occupy 2 positions
            else:
                seq_pos += 1  # Non-leaf nodes occupy 1 position
        return seq_pos

    def tree_tokens_to_sequence(self, tree_tokens: List) -> List[int]:
        """
        Public wrapper so external tools can convert tree-token layouts into
        vocabulary IDs without touching private helpers.
        """
        return self._tree_tokens_to_sequence(tree_tokens)

    def sequence_to_tree_tokens(self, token_sequence: List[int]) -> List:
        """
        Public wrapper exposing the inverse transformation used by statistics
        collectors and evaluation tools.
        """
        return self._sequence_to_tree_tokens(token_sequence)


def validate_fex_structure(tree_tokens: List, tree: FEXTree, 
                           vocab: Dict[str, int] = None) -> Tuple[bool, List[str]]:
    """
    Verify that the encoding result satisfies the FEX structural constraints
    
    Args:
        tree_tokens: tree token list (one token per node, leaves are tuples)
        tree: FEXTree structure
        vocab: vocabulary (optional)
        
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    # Check the node count
    if len(tree_tokens) != tree.total_nodes:
        errors.append(f"Token count mismatch: expected {tree.total_nodes}, got {len(tree_tokens)}")
    
    # Check each node
    for i, node in enumerate(tree.nodes):
        if i >= len(tree_tokens):
            errors.append(f"Missing node {i}")
            continue
            
        token = tree_tokens[i]
        node_type = node['type']
        
        if node_type == NODE_BINARY:
            if token not in BINARY_OPS:
                errors.append(
                    f"Node {i} (Binary) token '{token}' is invalid, expected: {BINARY_OPS}"
                )
        elif node_type == NODE_UNARY:
            if token not in UNARY_OPS:
                errors.append(
                    f"Node {i} (Unary) token '{token}' is invalid, expected: {UNARY_OPS}"
                )
        elif node_type == NODE_LEAF:
            # Leaf nodes should be 2-token tuples
            if not isinstance(token, tuple) or len(token) != 2:
                errors.append(
                    f"Node {i} (Leaf) should be a 2-token tuple, but got {type(token).__name__}: {token}"
                )
            else:
                pos1, pos2 = token
                # Check whether this is a padded leaf
                is_padding = (pos1 == '<PAD>' and pos2 == '<PAD>')
                
                if not is_padding:
                    # Check pos1: should be N00-N99 or +/-
                    valid_pos1 = pos1 in LEAF_POS1_MANTISSA or pos1 in LEAF_POS1_SIGN
                    if not valid_pos1:
                        errors.append(f"Node {i} (Leaf) pos1 '{pos1}' is invalid")
                    
                    # Check pos2: should be E-2/E-1/E0 or x_0...x_9
                    valid_pos2 = pos2 in LEAF_POS2_EXPONENT or pos2 in LEAF_POS2_VARIABLE
                    if not valid_pos2:
                        errors.append(f"Node {i} (Leaf) pos2 '{pos2}' is invalid")
    
    return len(errors) == 0, errors


def validate_encoding(encoder: FixedTreeEncoder, expr: sp.Expr) -> Tuple[bool, List[str]]:
    """Verify that the encoder correctly encodes an expression"""
    tree_tokens = [None] * encoder.tree.total_nodes
    root_idx = encoder.tree.get_root_inorder_idx()
    encoder._fill_tree_balanced(expr, root_idx, tree_tokens)
    encoder._fill_identity(tree_tokens)
    
    return validate_fex_structure(tree_tokens, encoder.tree, encoder.equation_word2id)
