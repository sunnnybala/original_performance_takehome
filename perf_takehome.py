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

# MODIFICATION: This version stores both values AND indices to memory.
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

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def get_slot_dest(self, slot):
        """Get destination address of a slot, or None for stores/control."""
        if slot[0] in ("store", "vstore"):
            return None
        if slot[0] in ("pause", "halt", "jump", "cond_jump"):
            return None
        # For most ops, dest is slot[1]
        return slot[1]

    def get_slot_reads(self, engine, slot):
        """Get set of addresses read by a slot."""
        reads = set()
        if engine == "store":
            if slot[0] == "store":
                reads.add(slot[1])  # addr
                reads.add(slot[2])  # src
            elif slot[0] == "vstore":
                reads.add(slot[1])  # addr
                for i in range(VLEN):
                    reads.add(slot[2] + i)  # src vector
        elif engine == "load":
            if slot[0] == "load":
                reads.add(slot[2])  # addr
            elif slot[0] == "vload":
                reads.add(slot[2])  # addr (scalar)
            # const has no reads
        elif engine == "valu":
            if slot[0] == "vbroadcast":
                reads.add(slot[2])  # scalar src
            elif slot[0] == "multiply_add":
                for i in range(VLEN):
                    reads.add(slot[2] + i)
                    reads.add(slot[3] + i)
                    reads.add(slot[4] + i)
            else:
                # Binary vector op
                for i in range(VLEN):
                    reads.add(slot[2] + i)
                    reads.add(slot[3] + i)
        elif engine == "alu":
            reads.add(slot[2])
            reads.add(slot[3])
        elif engine == "flow":
            if slot[0] == "select":
                reads.add(slot[2])  # cond
                reads.add(slot[3])  # a
                reads.add(slot[4])  # b
            elif slot[0] == "vselect":
                for i in range(VLEN):
                    reads.add(slot[2] + i)  # cond
                    reads.add(slot[3] + i)  # a
                    reads.add(slot[4] + i)  # b
        return reads

    def get_slot_writes(self, engine, slot):
        """Get set of addresses written by a slot."""
        writes = set()
        dest = self.get_slot_dest(slot)
        if dest is None:
            return writes
        if engine == "valu" or (engine == "flow" and slot[0] == "vselect"):
            for i in range(VLEN):
                writes.add(dest + i)
        elif engine == "load" and slot[0] == "vload":
            for i in range(VLEN):
                writes.add(dest + i)
        else:
            writes.add(dest)
        return writes

    def _dead_code_elimination(self, slots):
        """Remove operations whose results are never used."""
        if not slots:
            return slots

        n = len(slots)
        op_reads = [self.get_slot_reads(e, s) for e, s in slots]
        op_writes = [self.get_slot_writes(e, s) for e, s in slots]

        # Start with side-effect ops as live (stores, flow control)
        live = [False] * n
        for i, (engine, slot) in enumerate(slots):
            if engine == "store":
                live[i] = True
            elif engine == "flow" and slot[0] in ("pause", "halt", "jump", "cond_jump"):
                live[i] = True
            elif engine == "debug":
                live[i] = True

        # Backward pass: mark ops as live if their writes are read by a live op
        changed = True
        while changed:
            changed = False
            for i in range(n - 1, -1, -1):
                if live[i]:
                    # Mark all ops that write to addresses we read as live
                    for addr in op_reads[i]:
                        for j in range(i - 1, -1, -1):
                            if not live[j] and addr in op_writes[j]:
                                live[j] = True
                                changed = True
                                break  # Found the producer, don't look further back

        # Filter to only live ops
        return [slots[i] for i in range(n) if live[i]]

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False,
              beam_width: int = 2, beam_cycles: int = 0, beam_bundles: int = 3):
        """Pack slots into instruction bundles using list scheduling with beam search.

        For the first beam_cycles cycles, maintains beam_width candidate schedules,
        exploring beam_bundles different bundle choices at each step.
        """
        if not slots:
            return []

        # Dead code elimination disabled - breaks correctness with wavefront scheduling
        # The issue is that DCE doesn't see implicit dependencies between wavefront rounds
        # slots = self._dead_code_elimination(slots)

        n = len(slots)
        if n == 0:
            return []

        # Precompute reads/writes for each op
        op_reads = [self.get_slot_reads(e, s) for e, s in slots]
        op_writes = [self.get_slot_writes(e, s) for e, s in slots]

        # Build dependency graph
        # strict_deps[i] = ops that must complete BEFORE cycle containing i (RAW, WAW)
        # weak_deps[i] = ops that must be scheduled in same or earlier cycle (WAR)
        strict_deps = [set() for _ in range(n)]
        weak_deps = [set() for _ in range(n)]

        last_write = {}  # addr -> op index
        last_read = {}   # addr -> list of op indices

        for i in range(n):
            reads = op_reads[i]
            writes = op_writes[i]

            # RAW: i reads addr, some earlier op j wrote addr -> strict dep
            for addr in reads:
                if addr in last_write:
                    strict_deps[i].add(last_write[addr])

            # WAW: i writes addr, some earlier op j wrote addr -> strict dep
            for addr in writes:
                if addr in last_write:
                    strict_deps[i].add(last_write[addr])

            # WAR: i writes addr, some earlier op j read addr -> weak dep
            # (can be same cycle since reads before writes)
            for addr in writes:
                if addr in last_read:
                    for j in last_read[addr]:
                        if j < i:
                            weak_deps[i].add(j)

            # Update tracking
            for addr in writes:
                last_write[addr] = i
            for addr in reads:
                if addr not in last_read:
                    last_read[addr] = []
                last_read[addr].append(i)

        # Compute reverse dependencies for efficient updates
        strict_reverse = [set() for _ in range(n)]
        weak_reverse = [set() for _ in range(n)]
        for i in range(n):
            for j in strict_deps[i]:
                strict_reverse[j].add(i)
            for j in weak_deps[i]:
                weak_reverse[j].add(i)

        # Compute criticality (longest path to any sink)
        # Sinks are stores and flow control (pause, halt)
        criticality = [0] * n
        visited = [False] * n

        def compute_criticality(i):
            if visited[i]:
                return criticality[i]
            visited[i] = True
            engine = slots[i][0]
            # Sink nodes have criticality 0
            if engine == "store" or (engine == "flow" and slots[i][1][0] in ("pause", "halt")):
                criticality[i] = 0
            else:
                # Max of successors + 1
                max_succ = -1
                for j in strict_reverse[i]:
                    max_succ = max(max_succ, compute_criticality(j))
                criticality[i] = max_succ + 1 if max_succ >= 0 else 0
            return criticality[i]

        for i in range(n):
            compute_criticality(i)

        # Compute dist_to_load: minimum distance to any load op
        # This helps prioritize ops that are on the path to loads
        dist_to_load = [float('inf')] * n
        from collections import deque

        # Start BFS from all load ops
        queue = deque()
        for i in range(n):
            if slots[i][0] == "load":
                dist_to_load[i] = 0
                queue.append(i)

        # BFS backward through dependencies
        while queue:
            i = queue.popleft()
            # For each op j that i depends on (j -> i in dep graph)
            for j in strict_deps[i]:
                new_dist = dist_to_load[i] + 1
                if new_dist < dist_to_load[j]:
                    dist_to_load[j] = new_dist
                    queue.append(j)

        # Cap at reasonable max
        for i in range(n):
            if dist_to_load[i] == float('inf'):
                dist_to_load[i] = 1000

        def priority(i):
            """Priority function for sorting ready ops."""
            engine = slots[i][0]
            is_load = 1 if engine == "load" else 0
            is_store = 1 if engine == "store" else 0
            is_valu = 1 if engine == "valu" else 0
            # Check if this op unblocks any loads
            unblocks_load = 0
            for j in strict_reverse[i]:
                if slots[j][0] == "load":
                    unblocks_load = 1
                    break
            # Count successors
            n_successors = len(strict_reverse[i])
            return (-is_load, -is_store, dist_to_load[i], -unblocks_load, -criticality[i], -is_valu, -n_successors, i)

        def build_bundle(ready, scheduled, scheduled_this_cycle_set, skip_indices=None):
            """Build a bundle from ready ops, optionally skipping some indices."""
            if skip_indices is None:
                skip_indices = set()

            current_bundle = defaultdict(list)
            current_writes = set()
            current_reads = set()
            scheduled_this_cycle = []

            for i in ready:
                if i in skip_indices:
                    continue

                engine, slot = slots[i]
                reads = op_reads[i]
                writes = op_writes[i]

                # Check resource limit
                if len(current_bundle[engine]) >= SLOT_LIMITS[engine]:
                    continue

                # Check RAW within bundle (read depends on write this cycle)
                if reads & current_writes:
                    continue

                # Check WAW within bundle
                if writes & current_writes:
                    continue

                # Check weak deps: all weak deps must be scheduled (can be this cycle)
                weak_ok = True
                for j in weak_deps[i]:
                    if not scheduled[j] and j not in scheduled_this_cycle_set and j not in scheduled_this_cycle:
                        weak_ok = False
                        break
                if not weak_ok:
                    continue

                # Can schedule this op
                current_bundle[engine].append(slot)
                current_writes.update(writes)
                current_reads.update(reads)
                scheduled_this_cycle.append(i)

            return current_bundle, scheduled_this_cycle

        def score_state(scheduled, strict_remaining, instrs_so_far):
            """Score a partial schedule. Higher is better."""
            # Count loads scheduled (important to start data flowing)
            loads_scheduled = sum(1 for i in range(n) if scheduled[i] and slots[i][0] == "load")
            # Count total ops scheduled
            ops_scheduled = sum(scheduled)
            # Count flow ops (scarce resource)
            flow_scheduled = sum(1 for i in range(n) if scheduled[i] and slots[i][0] == "flow")
            # Sum of dist_to_load for scheduled ops (lower is better)
            dtl_sum = sum(dist_to_load[i] for i in range(n) if scheduled[i])

            # Score: prioritize loads, then total ops, penalize flow usage and dist_to_load
            return loads_scheduled * 100 + ops_scheduled * 10 - flow_scheduled * 5 - dtl_sum * 0.1

        # Beam search state: (scheduled, strict_remaining, weak_remaining, instrs)
        initial_state = (
            [False] * n,
            [len(d) for d in strict_deps],
            [len(d) for d in weak_deps],
            []
        )

        beam = [initial_state]

        cycle = 0
        while beam:
            # Check if best state is done
            best_scheduled = beam[0][0]
            if all(best_scheduled):
                break

            new_beam = []

            for state in beam:
                scheduled, strict_remaining, weak_remaining, instrs = state

                # Find ready ops for this state
                ready = [i for i in range(n) if not scheduled[i] and strict_remaining[i] == 0]

                if not ready:
                    if all(scheduled):
                        new_beam.append(state)
                    continue

                # Sort by priority
                ready.sort(key=priority)

                if cycle < beam_cycles:
                    # Beam search: explore multiple bundle choices
                    # Generate beam_bundles different bundles by skipping different ops
                    bundles_to_try = []

                    # Bundle 0: standard greedy
                    bundle, sched = build_bundle(ready, scheduled, set())
                    if sched:
                        bundles_to_try.append((bundle, sched))

                    # Bundle 1+: skip some high-priority ops to explore alternatives
                    for skip_count in range(1, min(beam_bundles, len(ready))):
                        # Skip the first skip_count ops
                        skip_set = set(ready[:skip_count])
                        bundle, sched = build_bundle(ready, scheduled, set(), skip_set)
                        if sched and (bundle, sched) not in bundles_to_try:
                            bundles_to_try.append((bundle, sched))

                    # Create new states for each bundle choice
                    for bundle, sched in bundles_to_try:
                        new_scheduled = scheduled.copy()
                        new_strict = strict_remaining.copy()
                        new_weak = weak_remaining.copy()
                        new_instrs = instrs + [dict(bundle)]

                        for i in sched:
                            new_scheduled[i] = True
                            for j in strict_reverse[i]:
                                new_strict[j] -= 1
                            for j in weak_reverse[i]:
                                new_weak[j] -= 1

                        new_beam.append((new_scheduled, new_strict, new_weak, new_instrs))
                else:
                    # After beam_cycles, use greedy
                    bundle, sched = build_bundle(ready, scheduled, set())
                    if sched:
                        new_scheduled = scheduled.copy()
                        new_strict = strict_remaining.copy()
                        new_weak = weak_remaining.copy()
                        new_instrs = instrs + [dict(bundle)]

                        for i in sched:
                            new_scheduled[i] = True
                            for j in strict_reverse[i]:
                                new_strict[j] -= 1
                            for j in weak_reverse[i]:
                                new_weak[j] -= 1

                        new_beam.append((new_scheduled, new_strict, new_weak, new_instrs))

            if not new_beam:
                break

            # Score and keep top beam_width states
            if cycle < beam_cycles:
                new_beam.sort(key=lambda s: -score_state(s[0], s[1], s[3]))
                beam = new_beam[:beam_width]
            else:
                beam = new_beam

            cycle += 1

        # Return instructions from best state
        if beam:
            return beam[0][3]
        return []

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

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

    def build_vhash_fused(self, val_vec, tmp1_vec, tmp2_vec, const_vecs, mult_vecs):
        """
        Optimized vector hash with multiply_add fusion.

        HASH_STAGES:
        Stage 0: (a + C1) + (a << 12) = a * 4097 + C1  -> multiply_add
        Stage 1: (a ^ C2) ^ (a >> 19)                  -> 3 ops
        Stage 2: (a + C3) + (a << 5) = a * 33 + C3     -> multiply_add
        Stage 3: (a + C4) ^ (a << 9)                   -> 3 ops
        Stage 4: (a + C5) + (a << 3) = a * 9 + C5      -> multiply_add
        Stage 5: (a ^ C6) ^ (a >> 16)                  -> 3 ops
        """
        slots = []

        # Stage 0: a * 4097 + C1 (4097 = 1 + 4096 = 1 + 2^12)
        slots.append(("valu", ("multiply_add", val_vec, val_vec, mult_vecs[4097], const_vecs[0x7ED55D16])))

        # Stage 1: (a ^ C2) ^ (a >> 19)
        slots.append(("valu", ("^", tmp1_vec, val_vec, const_vecs[0xC761C23C])))
        slots.append(("valu", (">>", tmp2_vec, val_vec, const_vecs[19])))
        slots.append(("valu", ("^", val_vec, tmp1_vec, tmp2_vec)))

        # Stage 2: a * 33 + C3 (33 = 1 + 32 = 1 + 2^5)
        slots.append(("valu", ("multiply_add", val_vec, val_vec, mult_vecs[33], const_vecs[0x165667B1])))

        # Stage 3: (a + C4) ^ (a << 9)
        slots.append(("valu", ("+", tmp1_vec, val_vec, const_vecs[0xD3A2646C])))
        slots.append(("valu", ("<<", tmp2_vec, val_vec, const_vecs[9])))
        slots.append(("valu", ("^", val_vec, tmp1_vec, tmp2_vec)))

        # Stage 4: a * 9 + C5 (9 = 1 + 8 = 1 + 2^3)
        slots.append(("valu", ("multiply_add", val_vec, val_vec, mult_vecs[9], const_vecs[0xFD7046C5])))

        # Stage 5: (a ^ C6) ^ (a >> 16)
        slots.append(("valu", ("^", tmp1_vec, val_vec, const_vecs[0xB55A4F09])))
        slots.append(("valu", (">>", tmp2_vec, val_vec, const_vecs[16])))
        slots.append(("valu", ("^", val_vec, tmp1_vec, tmp2_vec)))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Optimized SIMD implementation with depth preloading.
        - Phase 1: SIMD vectorization + scratch residency
        - Phase 2: Hash optimization with multiply_add fusion
        - Phase 3: Preload nodes for depths 0-3, eliminate gathers
        - MODIFICATION: Now stores both values AND indices at the end
        """
        n_tiles = batch_size // VLEN  # 256/8 = 32 tiles
        max_depth = forest_height  # Tree has depths 0 to forest_height

        # Scalar temps
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        tmp_addr = self.alloc_scratch("tmp_addr")

        # Scratch space addresses for init vars
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

        # Allocate vector scratch regions
        val_tiles_base = self.alloc_scratch("val_tiles", n_tiles * VLEN)
        # Pointer-form: store ptr = forest_values_p + idx instead of idx
        # This eliminates address computation during gathers
        ptr_tiles_base = self.alloc_scratch("ptr_tiles", n_tiles * VLEN)
        node_val_vec = self.alloc_scratch("node_val_vec", VLEN)
        # Per-tile temp vectors for stage-major hash interleaving
        tmp1_tiles_base = self.alloc_scratch("tmp1_tiles", n_tiles * VLEN)
        tmp2_tiles_base = self.alloc_scratch("tmp2_tiles", n_tiles * VLEN)
        bit_vec = self.alloc_scratch("bit_vec", VLEN)
        cmp_vec = self.alloc_scratch("cmp_vec", VLEN)

        # Batch all init operations for optimal packing
        init_ops = []

        # Precompute constant vectors for hash
        hash_consts = {
            0x7ED55D16, 0xC761C23C, 0x165667B1, 0xD3A2646C, 0xFD7046C5, 0xB55A4F09,
            19, 9, 16
        }
        const_vecs = {}
        const_scalars = {}  # Pre-create scalar consts
        for c in hash_consts:
            vec_addr = self.alloc_scratch(f"const_vec_{c}", VLEN)
            const_vecs[c] = vec_addr
            const_scalars[c] = self.scratch_const(c)

        mult_vecs = {}
        mult_scalars = {}
        for m in [4097, 33, 9]:
            vec_addr = self.alloc_scratch(f"mult_vec_{m}", VLEN)
            mult_vecs[m] = vec_addr
            mult_scalars[m] = self.scratch_const(m)

        zero_vec = self.alloc_scratch("zero_vec", VLEN)
        one_vec = self.alloc_scratch("one_vec", VLEN)
        two_vec = self.alloc_scratch("two_vec", VLEN)
        n_nodes_vec = self.alloc_scratch("n_nodes_vec", VLEN)

        # Preload nodes for depths 0-3 (nodes 0-14)
        preloaded_node_scalars = []
        node_addr_temps = []  # Separate address temps for each node load
        for node_idx in range(15):
            node_scalar = self.alloc_scratch(f"node_{node_idx}")
            preloaded_node_scalars.append(node_scalar)
            addr_tmp = self.alloc_scratch(f"node_addr_{node_idx}")
            node_addr_temps.append(addr_tmp)

        # Broadcast preloaded nodes to vectors
        preloaded_node_vecs = []
        for i in range(15):
            vec_addr = self.alloc_scratch(f"node_vec_{i}", VLEN)
            preloaded_node_vecs.append(vec_addr)

        # Diff vectors
        depth1_diff_vec = self.alloc_scratch("depth1_diff_vec", VLEN)
        depth2_diff34_vec = self.alloc_scratch("depth2_diff34_vec", VLEN)
        depth2_diff56_vec = self.alloc_scratch("depth2_diff56_vec", VLEN)
        depth3_diff_vecs = []
        for i in range(4):
            base = 7 + 2*i
            diff_vec = self.alloc_scratch(f"depth3_diff_{base}_{base+1}_vec", VLEN)
            depth3_diff_vecs.append(diff_vec)

        # Temp vectors for depth selection
        sel_tmp1 = self.alloc_scratch("sel_tmp1", VLEN)
        sel_tmp2 = self.alloc_scratch("sel_tmp2", VLEN)
        sel_tmp3 = self.alloc_scratch("sel_tmp3", VLEN)
        sel_tmp4 = self.alloc_scratch("sel_tmp4", VLEN)

        three_const = self.scratch_const(3)
        seven_const = self.scratch_const(7)

        # Pointer-form offset vectors
        # For depth 0: ptr = (fvp + 1) + bit
        # For depth > 0 with vselect: ptr = 2*ptr + addend, where addend = (1-fvp) if bit=0, (2-fvp) if bit=1
        ptr_offset0_scalar = self.alloc_scratch("ptr_offset0_scalar")  # 1 - forest_values_p (addend when bit=0)
        ptr_offset0_vec = self.alloc_scratch("ptr_offset0_vec", VLEN)
        ptr_offset1_scalar = self.alloc_scratch("ptr_offset1_scalar")  # 2 - forest_values_p (addend when bit=1)
        ptr_offset1_vec = self.alloc_scratch("ptr_offset1_vec", VLEN)
        fvp_plus_1_scalar = self.alloc_scratch("fvp_plus_1_scalar")  # forest_values_p + 1
        fvp_plus_1_vec = self.alloc_scratch("fvp_plus_1_vec", VLEN)
        fvp_plus_3_scalar = self.alloc_scratch("fvp_plus_3_scalar")  # forest_values_p + 3 (for depth 2)
        fvp_plus_3_vec = self.alloc_scratch("fvp_plus_3_vec", VLEN)
        fvp_plus_7_scalar = self.alloc_scratch("fvp_plus_7_scalar")  # forest_values_p + 7 (for depth 3)
        fvp_plus_7_vec = self.alloc_scratch("fvp_plus_7_vec", VLEN)
        fvp_vec = self.alloc_scratch("fvp_vec", VLEN)  # forest_values_p broadcast

        # Now emit all init ops in batched order
        # Phase 1: All broadcasts that don't depend on loads (can run in parallel)
        for c in hash_consts:
            init_ops.append(("valu", ("vbroadcast", const_vecs[c], const_scalars[c])))
        for m in [4097, 33, 9]:
            init_ops.append(("valu", ("vbroadcast", mult_vecs[m], mult_scalars[m])))
        init_ops.append(("valu", ("vbroadcast", zero_vec, zero_const)))
        init_ops.append(("valu", ("vbroadcast", one_vec, one_const)))
        init_ops.append(("valu", ("vbroadcast", two_vec, two_const)))
        init_ops.append(("valu", ("vbroadcast", n_nodes_vec, self.scratch["n_nodes"])))

        # Pointer-form offset computations (depend on forest_values_p being loaded)
        # ptr_offset0 = 1 - forest_values_p (addend when bit=0)
        init_ops.append(("alu", ("-", ptr_offset0_scalar, one_const, self.scratch["forest_values_p"])))
        # ptr_offset1 = 2 - forest_values_p (addend when bit=1)
        init_ops.append(("alu", ("-", ptr_offset1_scalar, two_const, self.scratch["forest_values_p"])))
        # fvp_plus_1 = forest_values_p + 1 (for depth 0 update and depth 1)
        init_ops.append(("alu", ("+", fvp_plus_1_scalar, self.scratch["forest_values_p"], one_const)))
        # fvp_plus_3 = forest_values_p + 3 (for depth 2)
        init_ops.append(("alu", ("+", fvp_plus_3_scalar, self.scratch["forest_values_p"], three_const)))
        # fvp_plus_7 = forest_values_p + 7 (for depth 3)
        init_ops.append(("alu", ("+", fvp_plus_7_scalar, self.scratch["forest_values_p"], seven_const)))
        # Broadcast these offsets
        init_ops.append(("valu", ("vbroadcast", ptr_offset0_vec, ptr_offset0_scalar)))
        init_ops.append(("valu", ("vbroadcast", ptr_offset1_vec, ptr_offset1_scalar)))
        init_ops.append(("valu", ("vbroadcast", fvp_plus_1_vec, fvp_plus_1_scalar)))
        init_ops.append(("valu", ("vbroadcast", fvp_plus_3_vec, fvp_plus_3_scalar)))
        init_ops.append(("valu", ("vbroadcast", fvp_plus_7_vec, fvp_plus_7_scalar)))
        init_ops.append(("valu", ("vbroadcast", fvp_vec, self.scratch["forest_values_p"])))

        # Phase 2: Node address computations (can be parallel with broadcasts)
        for node_idx in range(15):
            idx_const = self.scratch_const(node_idx)
            init_ops.append(("alu", ("+", node_addr_temps[node_idx], self.scratch["forest_values_p"], idx_const)))

        # Phase 3: Node loads (depend on addresses)
        for node_idx in range(15):
            init_ops.append(("load", ("load", preloaded_node_scalars[node_idx], node_addr_temps[node_idx])))

        # Phase 4: Node broadcasts (depend on loads)
        for i in range(15):
            init_ops.append(("valu", ("vbroadcast", preloaded_node_vecs[i], preloaded_node_scalars[i])))

        # Phase 5: Diff computations (depend on node broadcasts)
        init_ops.append(("valu", ("-", depth1_diff_vec, preloaded_node_vecs[2], preloaded_node_vecs[1])))
        init_ops.append(("valu", ("-", depth2_diff34_vec, preloaded_node_vecs[4], preloaded_node_vecs[3])))
        init_ops.append(("valu", ("-", depth2_diff56_vec, preloaded_node_vecs[6], preloaded_node_vecs[5])))
        for i in range(4):
            base = 7 + 2*i
            init_ops.append(("valu", ("-", depth3_diff_vecs[i], preloaded_node_vecs[base+1], preloaded_node_vecs[base])))

        # Phase 6: Interleave input value loads with pointer initialization
        # Pointer-form: ptr = forest_values_p + 0 = forest_values_p at start
        for tile in range(n_tiles):
            tile_offset = tile * VLEN
            # Input address compute
            offset_const = self.scratch_const(tile_offset)
            init_ops.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], offset_const)))
            # Pointer init: broadcast forest_values_p to this tile's ptr vector
            init_ops.append(("valu", ("vbroadcast", ptr_tiles_base + tile_offset, self.scratch["forest_values_p"])))
            # Input vload
            init_ops.append(("load", ("vload", val_tiles_base + tile_offset, tmp_addr)))

        # Build batched init instructions
        init_instrs = self.build(init_ops)
        self.instrs.extend(init_instrs)

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting loop"))

        body = []

        def get_depth_for_round(r):
            """Compute which tree depth we're reading from in round r."""
            # All start at depth 0 (root)
            # After reading, we move to next depth
            # At depth max_depth (leaves), we wrap to depth 0
            return r % (max_depth + 1)

        # Wavefront scheduling: divide tiles into groups that are staggered by round
        # This creates overlapping phases for better resource utilization
        wavefront_groups = 8  # Optimal for this code structure (tested 5-20)
        tiles_per_group = (n_tiles + wavefront_groups - 1) // wavefront_groups

        def get_group_tiles(group_idx):
            start = group_idx * tiles_per_group
            end = min(start + tiles_per_group, n_tiles)
            return list(range(start, end))

        # For each wave, emit operations for all groups at their current round
        for wave in range(rounds + wavefront_groups - 1):
            for group_idx in range(wavefront_groups):
                round_num = wave - group_idx
                if round_num < 0 or round_num >= rounds:
                    continue

                tiles = get_group_tiles(group_idx)
                if not tiles:
                    continue

                depth = get_depth_for_round(round_num)
                is_last_round = (round_num == rounds - 1)
                is_leaf_depth = (depth == max_depth)

                # Emit operations for this group at this round
                # Node selection and XOR
                if depth == 0:
                    for tile in tiles:
                        val_addr = val_tiles_base + tile * VLEN
                        body.append(("valu", ("^", val_addr, val_addr, preloaded_node_vecs[0])))
                elif depth >= 4:
                    # Emit all loads first, then all XORs (scheduler handles interleaving)
                    # Hash is done in the shared section below
                    for tile in tiles:
                        ptr_addr = ptr_tiles_base + tile * VLEN
                        for lane in range(VLEN):
                            body.append(("load", ("load", tmp1_tiles_base + tile * VLEN + lane, ptr_addr + lane)))
                    for tile in tiles:
                        val_addr = val_tiles_base + tile * VLEN
                        node_addr = tmp1_tiles_base + tile * VLEN
                        body.append(("valu", ("^", val_addr, val_addr, node_addr)))
                elif depth == 1:
                    # Use vselect for 1-bit lookup (1 VALU + 1 FLOW instead of 2 VALU)
                    for tile in tiles:
                        tile_offset = tile * VLEN
                        ptr_addr = ptr_tiles_base + tile_offset
                        node_dest = tmp1_tiles_base + tile_offset
                        # bit = ptr - fvp_plus_1 (gives 0 or 1)
                        body.append(("valu", ("-", cmp_vec, ptr_addr, fvp_plus_1_vec)))
                        body.append(("flow", ("vselect", node_dest, cmp_vec, preloaded_node_vecs[2], preloaded_node_vecs[1])))
                    for tile in tiles:
                        val_addr = val_tiles_base + tile * VLEN
                        node_addr = tmp1_tiles_base + tile * VLEN
                        body.append(("valu", ("^", val_addr, val_addr, node_addr)))
                elif depth == 2:
                    # Use vselect for 2-bit lookup (4 VALU + 3 FLOW instead of 8 VALU)
                    for tile in tiles:
                        tile_offset = tile * VLEN
                        ptr_addr = ptr_tiles_base + tile_offset
                        node_dest = tmp1_tiles_base + tile_offset
                        # Extract 2-bit index: bit0, bit1
                        body.append(("valu", ("-", cmp_vec, ptr_addr, fvp_plus_3_vec)))
                        body.append(("valu", ("&", bit_vec, cmp_vec, one_vec)))
                        body.append(("valu", (">>", sel_tmp1, cmp_vec, one_vec)))
                        body.append(("valu", ("&", sel_tmp1, sel_tmp1, one_vec)))
                        # Level 1: select by bit0 -> pairs (3,4) and (5,6)
                        body.append(("flow", ("vselect", sel_tmp2, bit_vec, preloaded_node_vecs[4], preloaded_node_vecs[3])))
                        body.append(("flow", ("vselect", sel_tmp3, bit_vec, preloaded_node_vecs[6], preloaded_node_vecs[5])))
                        # Level 2: select by bit1
                        body.append(("flow", ("vselect", node_dest, sel_tmp1, sel_tmp3, sel_tmp2)))
                    for tile in tiles:
                        val_addr = val_tiles_base + tile * VLEN
                        node_addr = tmp1_tiles_base + tile * VLEN
                        body.append(("valu", ("^", val_addr, val_addr, node_addr)))
                elif depth == 3:
                    # Use vselect tree for 3-bit lookup (6 VALU + 7 FLOW instead of 10 VALU + 3 FLOW)
                    for tile in tiles:
                        tile_offset = tile * VLEN
                        ptr_addr = ptr_tiles_base + tile_offset
                        node_dest = tmp1_tiles_base + tile_offset
                        # Extract 3-bit index
                        body.append(("valu", ("-", cmp_vec, ptr_addr, fvp_plus_7_vec)))
                        body.append(("valu", ("&", bit_vec, cmp_vec, one_vec)))
                        body.append(("valu", (">>", sel_tmp1, cmp_vec, one_vec)))
                        body.append(("valu", ("&", sel_tmp1, sel_tmp1, one_vec)))
                        body.append(("valu", (">>", sel_tmp2, cmp_vec, two_vec)))
                        body.append(("valu", ("&", sel_tmp2, sel_tmp2, one_vec)))
                        # Level 1: select pairs by bit0 (4 vselect)
                        body.append(("flow", ("vselect", sel_tmp3, bit_vec, preloaded_node_vecs[8], preloaded_node_vecs[7])))
                        body.append(("flow", ("vselect", sel_tmp4, bit_vec, preloaded_node_vecs[10], preloaded_node_vecs[9])))
                        body.append(("flow", ("vselect", cmp_vec, bit_vec, preloaded_node_vecs[12], preloaded_node_vecs[11])))
                        body.append(("flow", ("vselect", bit_vec, bit_vec, preloaded_node_vecs[14], preloaded_node_vecs[13])))
                        # Level 2: select groups by bit1 (2 vselect)
                        body.append(("flow", ("vselect", node_val_vec, sel_tmp1, sel_tmp4, sel_tmp3)))
                        body.append(("flow", ("vselect", sel_tmp3, sel_tmp1, bit_vec, cmp_vec)))
                        # Level 3: final select by bit2
                        body.append(("flow", ("vselect", node_dest, sel_tmp2, sel_tmp3, node_val_vec)))
                    for tile in tiles:
                        val_addr = val_tiles_base + tile * VLEN
                        node_addr = tmp1_tiles_base + tile * VLEN
                        body.append(("valu", ("^", val_addr, val_addr, node_addr)))

                # Hash for this group
                for tile in tiles:
                    val_addr = val_tiles_base + tile * VLEN
                    body.append(("valu", ("multiply_add", val_addr, val_addr, mult_vecs[4097], const_vecs[0x7ED55D16])))
                for tile in tiles:
                    val_addr = val_tiles_base + tile * VLEN
                    tmp2_addr = tmp2_tiles_base + tile * VLEN
                    body.append(("valu", (">>", tmp2_addr, val_addr, const_vecs[19])))
                c2_scalar = const_scalars[0xC761C23C]
                for tile in tiles:
                    val_addr = val_tiles_base + tile * VLEN
                    for lane in range(VLEN):
                        body.append(("alu", ("^", val_addr + lane, val_addr + lane, c2_scalar)))
                for tile in tiles:
                    val_addr = val_tiles_base + tile * VLEN
                    tmp2_addr = tmp2_tiles_base + tile * VLEN
                    body.append(("valu", ("^", val_addr, val_addr, tmp2_addr)))
                for tile in tiles:
                    val_addr = val_tiles_base + tile * VLEN
                    body.append(("valu", ("multiply_add", val_addr, val_addr, mult_vecs[33], const_vecs[0x165667B1])))
                for tile in tiles:
                    val_addr = val_tiles_base + tile * VLEN
                    tmp2_addr = tmp2_tiles_base + tile * VLEN
                    body.append(("valu", ("<<", tmp2_addr, val_addr, const_vecs[9])))
                c4_scalar = const_scalars[0xD3A2646C]
                for tile in tiles:
                    val_addr = val_tiles_base + tile * VLEN
                    for lane in range(VLEN):
                        body.append(("alu", ("+", val_addr + lane, val_addr + lane, c4_scalar)))
                for tile in tiles:
                    val_addr = val_tiles_base + tile * VLEN
                    tmp2_addr = tmp2_tiles_base + tile * VLEN
                    body.append(("valu", ("^", val_addr, val_addr, tmp2_addr)))
                for tile in tiles:
                    val_addr = val_tiles_base + tile * VLEN
                    body.append(("valu", ("multiply_add", val_addr, val_addr, mult_vecs[9], const_vecs[0xFD7046C5])))
                for tile in tiles:
                    val_addr = val_tiles_base + tile * VLEN
                    tmp2_addr = tmp2_tiles_base + tile * VLEN
                    body.append(("valu", (">>", tmp2_addr, val_addr, const_vecs[16])))
                c6_scalar = const_scalars[0xB55A4F09]
                for tile in tiles:
                    val_addr = val_tiles_base + tile * VLEN
                    for lane in range(VLEN):
                        body.append(("alu", ("^", val_addr + lane, val_addr + lane, c6_scalar)))
                for tile in tiles:
                    val_addr = val_tiles_base + tile * VLEN
                    tmp2_addr = tmp2_tiles_base + tile * VLEN
                    body.append(("valu", ("^", val_addr, val_addr, tmp2_addr)))

                # Pointer update for this group
                # NOTE: We need to compute pointer even in last round to store the final indices
                if is_leaf_depth:
                    for tile in tiles:
                        ptr_addr = ptr_tiles_base + tile * VLEN
                        body.append(("valu", ("+", ptr_addr, fvp_vec, zero_vec)))
                elif depth == 0:
                    for tile in tiles:
                        val_addr = val_tiles_base + tile * VLEN
                        ptr_addr = ptr_tiles_base + tile * VLEN
                        body.append(("valu", ("&", ptr_addr, val_addr, one_vec)))
                    for tile in tiles:
                        ptr_addr = ptr_tiles_base + tile * VLEN
                        body.append(("valu", ("+", ptr_addr, ptr_addr, fvp_plus_1_vec)))
                else:
                    # Pointer update with vselect: trades 1 VALU for 1 FLOW
                    # Old: bit = val & 1 (VALU), ptr = 2*ptr + offset0 (VALU), ptr += bit (VALU) = 3 VALU
                    # New: bit = val & 1 (VALU), addend = vselect(bit, offset1, offset0) (FLOW),
                    #      ptr = 2*ptr + addend (VALU) = 2 VALU + 1 FLOW
                    for tile in tiles:
                        val_addr = val_tiles_base + tile * VLEN
                        tmp1_addr = tmp1_tiles_base + tile * VLEN
                        body.append(("valu", ("&", tmp1_addr, val_addr, one_vec)))  # bit = val & 1
                    for tile in tiles:
                        tmp1_addr = tmp1_tiles_base + tile * VLEN
                        # vselect: if bit then offset1 else offset0 (read bit first, then write addend)
                        body.append(("flow", ("vselect", tmp1_addr, tmp1_addr, ptr_offset1_vec, ptr_offset0_vec)))
                    for tile in tiles:
                        ptr_addr = ptr_tiles_base + tile * VLEN
                        tmp1_addr = tmp1_tiles_base + tile * VLEN
                        body.append(("valu", ("multiply_add", ptr_addr, ptr_addr, two_vec, tmp1_addr)))

                # Stores for last round - NOW STORES BOTH VALUES AND INDICES
                if is_last_round:
                    # Split node_addr_temps: first half for value addresses, second half for index addresses
                    batch_size_store = len(node_addr_temps) // 2  # 7 temps for values, 7 for indices
                    val_addr_temps = node_addr_temps[:batch_size_store]
                    idx_addr_temps = node_addr_temps[batch_size_store:batch_size_store*2]

                    for batch_start in range(0, len(tiles), batch_size_store):
                        batch_end = min(batch_start + batch_size_store, len(tiles))
                        batch_tiles = tiles[batch_start:batch_end]

                        # Phase 1: Compute indices (idx = ptr - fvp) into tmp2_tiles
                        # Use tmp2_tiles since it's available after hash computation
                        for i, tile in enumerate(batch_tiles):
                            tile_offset = tile * VLEN
                            ptr_addr = ptr_tiles_base + tile_offset
                            idx_dest = tmp2_tiles_base + tile_offset
                            body.append(("valu", ("-", idx_dest, ptr_addr, fvp_vec)))

                        # Phase 2: Compute value store addresses
                        for i, tile in enumerate(batch_tiles):
                            tile_offset = tile * VLEN
                            offset_const = self.scratch_const(tile_offset)
                            body.append(("alu", ("+", val_addr_temps[i], self.scratch["inp_values_p"], offset_const)))

                        # Phase 3: Compute index store addresses (parallel with Phase 2)
                        for i, tile in enumerate(batch_tiles):
                            tile_offset = tile * VLEN
                            offset_const = self.scratch_const(tile_offset)
                            body.append(("alu", ("+", idx_addr_temps[i], self.scratch["inp_indices_p"], offset_const)))

                        # Phase 4: Store values
                        for i, tile in enumerate(batch_tiles):
                            tile_offset = tile * VLEN
                            body.append(("store", ("vstore", val_addr_temps[i], val_tiles_base + tile_offset)))

                        # Phase 5: Store indices
                        for i, tile in enumerate(batch_tiles):
                            tile_offset = tile * VLEN
                            body.append(("store", ("vstore", idx_addr_temps[i], tmp2_tiles_base + tile_offset)))

        # Remove old loop since we've replaced it with wavefront
        # The old 'for round_num in range(rounds):' loop has been replaced above
        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)
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
        ), f"Incorrect output values on round {i}"

        # NOW ALSO CHECK INDICES
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        assert (
            machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)]
            == ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)]
        ), f"Incorrect output indices on round {i}"

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
#    python perf_takehome_indices.py
# To run a specific test:
#    python perf_takehome_indices.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome_indices.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
