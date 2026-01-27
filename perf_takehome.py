"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        # Bundle accumulator for VLIW packing
        self.current_bundle = defaultdict(list)

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def emit_bundle(self):
        """Emit the current bundle and reset it"""
        if self.current_bundle:
            self.instrs.append(dict(self.current_bundle))
            self.current_bundle = defaultdict(list)

    def bundle_add(self, engine, slot):
        """Add an operation to the current bundle, emit if slot limit reached"""
        if len(self.current_bundle[engine]) >= SLOT_LIMITS[engine]:
            self.emit_bundle()
        self.current_bundle[engine].append(slot)

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def alloc_vec(self, name=None):
        """Allocate a vector register (VLEN consecutive scratch locations)"""
        return self.alloc_scratch(name, VLEN)

    def scratch_const_vec(self, val, name=None):
        """Allocate a vector constant (broadcast scalar to all lanes)"""
        key = ("vec", val)
        if key not in self.const_map:
            scalar_addr = self.scratch_const(val)
            vec_addr = self.alloc_vec(name)
            self.add("valu", ("vbroadcast", vec_addr, scalar_addr))
            self.const_map[key] = vec_addr
        return self.const_map[key]

    def build_hash_vec_bundled(self, v_val, v_tmp1, v_tmp2):
        """Build vectorized hash computation with bundling"""
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v_const1 = self.scratch_const_vec(val1)
            v_const3 = self.scratch_const_vec(val3)
            # tmp1 = val op1 const1 and tmp2 = val op3 const3 can be bundled (both read v_val)
            self.bundle_add("valu", (op1, v_tmp1, v_val, v_const1))
            self.bundle_add("valu", (op3, v_tmp2, v_val, v_const3))
            self.emit_bundle()
            # val = tmp1 op2 tmp2 (depends on both above)
            self.bundle_add("valu", (op2, v_val, v_tmp1, v_tmp2))
            self.emit_bundle()

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Vectorized implementation with group processing.
        Processes multiple blocks together in phases for better slot utilization.
        """
        # Scalar temps
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")

        # Scratch space addresses for memory layout info
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        # Scalar constants
        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Vector constants
        v_zero = self.scratch_const_vec(0, "v_zero")
        v_one = self.scratch_const_vec(1, "v_one")
        v_two = self.scratch_const_vec(2, "v_two")
        v_n_nodes = self.alloc_vec("v_n_nodes")
        self.add("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))
        v_forest_p = self.alloc_vec("v_forest_p")
        self.add("valu", ("vbroadcast", v_forest_p, self.scratch["forest_values_p"]))

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting optimized processing"))

        # Number of vector blocks
        n_blocks = batch_size // VLEN  # 32 blocks for batch_size=256

        # Preload shallow tree nodes (levels 0-3, nodes 0-14)
        # This allows us to use vselect instead of memory loads for early rounds
        N_PRELOAD = 15  # nodes 0-14
        preloaded_nodes = []
        for node_idx in range(N_PRELOAD):
            v_node = self.alloc_vec(f"preload_{node_idx}")
            # Load the node value and broadcast to vector
            s_node_addr = self.alloc_scratch()
            self.add("alu", ("+", s_node_addr, self.scratch["forest_values_p"], self.scratch_const(node_idx)))
            s_node_val = self.alloc_scratch()
            self.add("load", ("load", s_node_val, s_node_addr))
            self.add("valu", ("vbroadcast", v_node, s_node_val))
            preloaded_nodes.append(v_node)

        # Process all blocks together for maximum parallelism
        N_GROUP = n_blocks  # 32 blocks

        # Allocate vector registers for each block in the group
        group_v_idx = [self.alloc_vec(f"g_idx_{i}") for i in range(N_GROUP)]
        group_v_val = [self.alloc_vec(f"g_val_{i}") for i in range(N_GROUP)]
        group_v_addr = [self.alloc_vec(f"g_addr_{i}") for i in range(N_GROUP)]
        group_v_node = [self.alloc_vec(f"g_node_{i}") for i in range(N_GROUP)]

        # Scalar base address registers for each block
        group_s_idx_base = [self.alloc_scratch(f"s_idx_{i}") for i in range(N_GROUP)]
        group_s_val_base = [self.alloc_scratch(f"s_val_{i}") for i in range(N_GROUP)]

        # Precompute block offset constants
        block_offsets = [self.scratch_const(b * VLEN) for b in range(n_blocks)]

        # ROUND FUSION - process all blocks together
        group_size = N_GROUP

        # Phase 1: Compute all base addresses (ALU ops - can bundle up to 12)
        for b in range(group_size):
            block_offset_const = self.scratch_const(b * VLEN)
            self.bundle_add("alu", ("+", group_s_idx_base[b], self.scratch["inp_indices_p"], block_offset_const))
            self.bundle_add("alu", ("+", group_s_val_base[b], self.scratch["inp_values_p"], block_offset_const))
        self.emit_bundle()

        # Phase 2: Load all indices and values ONCE (2 loads per cycle)
        for b in range(group_size):
            self.bundle_add("load", ("vload", group_v_idx[b], group_s_idx_base[b]))
            self.bundle_add("load", ("vload", group_v_val[b], group_s_val_base[b]))
        self.emit_bundle()

        # Reuse vectors for temps/cond/child throughout all rounds
        group_v_tmp1 = group_v_addr
        group_v_tmp2 = group_v_node
        group_v_cond = group_v_addr
        group_v_child = group_v_node

        # Process ALL rounds without storing back
        for round_num in range(rounds):
            # PHASE 3-4: Get tree node values
            # Optimize shallow levels with preloaded nodes + vselect
            if round_num == 0:
                # All indices are 0, use preloaded node 0
                for b in range(group_size):
                    self.bundle_add("valu", ("+", group_v_node[b], preloaded_nodes[0], v_zero))
                self.emit_bundle()
            elif round_num == 1:
                # Indices are 1 or 2
                # idx & 1: gives 1 for idx=1, 0 for idx=2
                for b in range(group_size):
                    self.bundle_add("valu", ("&", group_v_addr[b], group_v_idx[b], v_one))
                self.emit_bundle()
                for b in range(group_size):
                    self.bundle_add("flow", ("vselect", group_v_node[b], group_v_addr[b], preloaded_nodes[1], preloaded_nodes[2]))
                    self.emit_bundle()
            else:
                # Standard memory load path for deeper levels
                # Compute all tree addresses
                for b in range(group_size):
                    self.bundle_add("valu", ("+", group_v_addr[b], v_forest_p, group_v_idx[b]))
                self.emit_bundle()

                # Gather all tree node values
                for lane in range(VLEN):
                    for b in range(group_size):
                        self.bundle_add("load", ("load_offset", group_v_node[b], group_v_addr[b], lane))
                    self.emit_bundle()

            # PHASE 5: XOR node values
            for b in range(group_size):
                self.bundle_add("valu", ("^", group_v_val[b], group_v_val[b], group_v_node[b]))
            self.emit_bundle()

            # PHASE 6: Hash computation
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                if op1 == "+" and op2 == "+" and op3 == "<<":
                    multiplier = 1 + (1 << val3)
                    v_mult = self.scratch_const_vec(multiplier)
                    v_const1 = self.scratch_const_vec(val1)
                    for b in range(group_size):
                        self.bundle_add("valu", ("multiply_add", group_v_val[b], group_v_val[b], v_mult, v_const1))
                    self.emit_bundle()
                else:
                    v_const1 = self.scratch_const_vec(val1)
                    v_const3 = self.scratch_const_vec(val3)
                    for b in range(group_size):
                        self.bundle_add("valu", (op1, group_v_tmp1[b], group_v_val[b], v_const1))
                        self.bundle_add("valu", (op3, group_v_tmp2[b], group_v_val[b], v_const3))
                    self.emit_bundle()
                    for b in range(group_size):
                        self.bundle_add("valu", (op2, group_v_val[b], group_v_tmp1[b], group_v_tmp2[b]))
                    self.emit_bundle()

            # PHASE 7: Child selection and index update
            for b in range(group_size):
                self.bundle_add("valu", ("&", group_v_cond[b], group_v_val[b], v_one))
            self.emit_bundle()

            for b in range(group_size):
                self.bundle_add("valu", ("+", group_v_child[b], v_one, group_v_cond[b]))
            self.emit_bundle()

            for b in range(group_size):
                self.bundle_add("valu", ("multiply_add", group_v_idx[b], group_v_idx[b], v_two, group_v_child[b]))
            self.emit_bundle()

            for b in range(group_size):
                self.bundle_add("valu", ("<", group_v_cond[b], group_v_idx[b], v_n_nodes))
            self.emit_bundle()

            for b in range(group_size):
                self.bundle_add("valu", ("*", group_v_idx[b], group_v_idx[b], group_v_cond[b]))
            self.emit_bundle()

        # Phase 8: Store all results ONCE after all rounds
        for b in range(group_size):
            self.bundle_add("store", ("vstore", group_s_idx_base[b], group_v_idx[b]))
            self.bundle_add("store", ("vstore", group_s_val_base[b], group_v_val[b]))
        self.emit_bundle()

        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
