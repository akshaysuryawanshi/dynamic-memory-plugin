"""InnoDB-style tail withdrawal on the pinned, single-group block pool."""
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlock


def referenced_block_count(pool):
    """Return capacity occupied by live references, including block zero.

    The null block is permanently removed from the free queue and its refcount
    is intentionally not maintained by vLLM. Counting total minus free treats
    that reserved block as occupied, matching a target that also includes it.
    """
    free = pool.get_num_free_blocks()
    # During tail withdrawal, zero-ref blocks in ``withdrawn`` have already
    # left the free queue. They are withheld capacity, not live references.
    withheld = len(getattr(pool, "withdrawn", ()))
    if not 0 <= free + withheld < pool.num_gpu_blocks:
        raise RuntimeError("Block-pool free count is inconsistent with capacity")
    return pool.num_gpu_blocks - free - withheld


class WithdrawingBlockPool(BlockPool):
    """Retain block identities and live references while retiring a tail.

    The pool remains usable throughout withdrawal. Retiring cached blocks
    disappear from prefix lookup immediately, and retiring free blocks are
    held out of the allocator until commit or cancellation.
    """

    @classmethod
    def adopt(cls, pool):
        if isinstance(pool, cls):
            return pool
        if type(pool) is not BlockPool or pool._reuse_watchers:
            raise ValueError("Withdrawal needs an ordinary pool without external readers")
        pool.__class__ = cls
        pool.retire_target = None
        pool.withdrawn = {}
        return pool

    def begin_withdrawal(self, target):
        if self.retire_target is not None or not 1 < target < self.num_gpu_blocks:
            raise ValueError("Invalid or overlapping tail withdrawal")
        self.retire_target = target
        for block in self.blocks[target:]:
            self._maybe_evict_cached_block(block)
            if block.ref_cnt == 0:
                self._withdraw_free(block)

    def _withdraw_free(self, block):
        if block.block_id not in self.withdrawn:
            self.free_block_queue.remove(block)
            self.withdrawn[block.block_id] = block

    def _insert_block_hash(self, key, block, num_tokens):
        if self.retire_target is not None and block.block_id >= self.retire_target:
            return
        super()._insert_block_hash(key, block, num_tokens)

    def free_blocks(self, ordered_blocks):
        blocks = tuple(ordered_blocks)
        super().free_blocks(blocks)
        if self.retire_target is not None:
            for block in blocks:
                if block.pool is self and block.block_id >= self.retire_target and block.ref_cnt == 0:
                    self._withdraw_free(block)

    def ready(self):
        return self.retire_target is None or len(self.withdrawn) == self.num_gpu_blocks - self.retire_target

    def release_adapter(self):
        """Remove transaction-only overrides from the normal scheduling path."""
        if self.retire_target is not None or self.withdrawn or self._reuse_watchers:
            raise RuntimeError("Cannot detach an unfinished withdrawal")
        del self.retire_target, self.withdrawn
        self.__class__ = BlockPool

    def cancel_withdrawal(self):
        self.free_block_queue.append_n(list(self.withdrawn.values()))
        self.withdrawn.clear()
        self.retire_target = None

    def commit_capacity(self, target):
        if target < self.num_gpu_blocks:
            if self.retire_target != target or not self.ready():
                raise RuntimeError("Tail still has live references")
            del self.blocks[target:]
            self.withdrawn.clear()
            self.retire_target = None
        elif target > self.num_gpu_blocks:
            added = [KVCacheBlock(i, pool=self) for i in range(self.num_gpu_blocks, target)]
            self.blocks.extend(added)
            self.free_block_queue.append_n(added)
        self.num_gpu_blocks = target
