import os
import torch

class LayerToCache:
    def __init__(self, layer_num, cache_num) -> None:
        r'''
            args:
                layer_num: the number of layers that managed by the manager
                cache_num: the number of layers to be offloaded, 
                           the allocated buffer is `layer_num - cache_num`
        '''
        self.layer_num = layer_num
        self.cache_num = cache_num
        self.validate()

    def __call__(self, layer_num):
        r'''
            return: 
                cache_idx: the index of cache that the layer is offloaded to
                offload:   whether the layer needs to be offloaded, 
                           in other words, whether the layer is sharing the cache with other layers
        '''
        raise NotImplementedError('')

    def validate(self):
        raise NotImplementedError('')

class RingCacheOrder2(LayerToCache):
    def __init__(self, layer_num, cache_num) -> None:
        self.half_layer_num = layer_num // 2
        self.prefetch = lambda x: (x + self.half_layer_num) % self.layer_num
        super().__init__(layer_num, cache_num)

    def validate(self):
        results = []
        for i in range(self.layer_num):
            results.append(self(i))
        layer_nums = [i[0] for i in results]
        cache_nums = [i[1] for i in results]
        assert len(set(layer_nums)) == self.layer_num - self.cache_num
        assert max(layer_nums) == self.layer_num - self.cache_num - 1
        assert sum(cache_nums) == self.cache_num * 2

        layer_to_offload = [i[0] for i in results if i[1]]
        for i in layer_to_offload:
            assert i == self.prefetch(self.prefetch(
                i)), f'the next of {i} is {self.prefetch(i)}, but the next of {self.prefetch(i)} is {self.prefetch(self.prefetch(i))}'

    def __call__(self, layer_num):
        if layer_num < self.cache_num:
            return layer_num, True
        elif layer_num < self.half_layer_num:
            return layer_num, False
        elif layer_num < self.half_layer_num + self.cache_num:
            return layer_num - self.half_layer_num, True
        else:
            return layer_num - self.cache_num, False

class MemoryOffloader:
    protected_attr = ['key_buffer', 'value_buffer']
    def __init__(self, mem_manager_cls, mem_manager_cfg, cache_num, layer_cache_mapping=RingCacheOrder2):
        self.cache_num = cache_num
        assert 'layer_num' in mem_manager_cfg
        self.layer_num = mem_manager_cfg['layer_num']
        mem_manager_cfg['layer_num'] = self.layer_num - cache_num
        self.mem_manager = mem_manager_cls(**mem_manager_cfg)
        self.key_shape = self.mem_manager.key_buffer[0].shape
        self.value_shape = self.mem_manager.value_buffer[0].shape
        self.layer_to_cache = layer_cache_mapping(self.layer_num, self.cache_num)
        self.dtype = self.mem_manager.key_buffer[0].dtype

        self.__enable = (self.cache_num > 0)

        self._allocate_cpu_buffer(self.key_shape, self.value_shape, self.dtype, self.layer_num)
        self._allocate_cuda_context(self.layer_num, self.cache_num)

    def __getattribute__(self, name):
        if name.startswith('_') is False and name not in MemoryOffloader.protected_attr:
            try:
                attr = super().__getattribute__('mem_manager')
                return getattr(attr, name)
            except AttributeError:
                pass
        try:
            return super().__getattribute__(name)
        except AttributeError:
            raise AttributeError(f"'OffloadWrapper' and its mem_manager have no attribute '{name}'")

    @property
    def key_buffer(self):
        return lambda idx: self.get_key_buffer(idx)
    
    @property
    def value_buffer(self):
        return lambda idx: self.get_value_buffer(idx)

    @property
    def enable(self):
        return self.__enable

    def _allocate_cpu_buffer(self, key_shape, value_shape, dtype, layer_num):
        if not self.enable: return
        self.key_buffer_cpu = [torch.empty(
            key_shape, dtype=dtype, device="cpu", pin_memory=True) for _ in range(layer_num)]
        self.value_buffer_cpu = [torch.empty(
            value_shape, dtype=dtype, device="cpu", pin_memory=True) for _ in range(layer_num)]
        
    def _allocate_cuda_context(self, layer_num, cache_num):
        if not self.enable: return
        self.key_event = [[] for _ in range(layer_num - cache_num)]
        self.value_event = [[] for _ in range(layer_num - cache_num)]
        self.offload_stream = torch.cuda.Stream()
    
    def check_key_cpu(self):
        torch.cuda.synchronize()
        res = []
        for i in self.key_buffer_cpu:
            res.append(i[:5].norm(2).item())
    
    def check_value_cpu(self):
        torch.cuda.synchronize()
        res = []
        for i in self.value_buffer_cpu:
            res.append(i[:5].norm(2).item())

    def wait_key(self, layer_num):
        if not self.enable: return
        cache_idx, is_offload = self.layer_to_cache(layer_num)
        if is_offload:
            for event in self.key_event[cache_idx]:
                event.synchronize()
            self.key_event[cache_idx].clear()
    
    def wait_value(self, layer_num):
        if not self.enable: return
        cache_idx, is_offload = self.layer_to_cache(layer_num)
        if is_offload:
            for event in self.value_event[cache_idx]:
                event.synchronize()
            self.value_event[cache_idx].clear()
    
    def try_offload_key(self, layer_num, idx):
        if not self.enable: return
        cache_idx, is_offload = self.layer_to_cache(layer_num)
        index = slice(idx + 1)
        if is_offload:
            # torch.cuda.synchronize()
            ds = torch.cuda.default_stream()
            self.offload_stream.wait_stream(ds)
            next_layer_num = self.layer_to_cache.prefetch(layer_num)
            assert self.key_event[cache_idx] == [], f'cache_idx {cache_idx} key_event is not empty'
            with torch.cuda.stream(self.offload_stream):
                self.key_buffer_cpu[layer_num][index].copy_(
                    self.mem_manager.key_buffer[cache_idx][index], non_blocking=True)
                self.mem_manager.key_buffer[cache_idx][index].copy_(
                    self.key_buffer_cpu[next_layer_num][index], non_blocking=True)
                self.key_event[cache_idx].append(
                    self.offload_stream.record_event())

    def try_offload_value(self, layer_num, idx):
        if not self.enable: return
        cache_idx, is_offload = self.layer_to_cache(layer_num)
        index = slice(idx + 1)
        if is_offload:
            # torch.cuda.synchronize()
            ds = torch.cuda.default_stream()
            self.offload_stream.wait_stream(ds)
            next_layer_num = self.layer_to_cache.prefetch(layer_num)
            assert self.value_event[cache_idx] == [], f'cache_idx {cache_idx} value_event is not empty'
            with torch.cuda.stream(self.offload_stream):
                self.value_buffer_cpu[layer_num][index].copy_(
                    self.mem_manager.value_buffer[cache_idx][index], non_blocking=True)
                self.mem_manager.value_buffer[cache_idx][index].copy_(
                    self.value_buffer_cpu[next_layer_num][index], non_blocking=True)
                self.value_event[cache_idx].append(
                    self.offload_stream.record_event())

    
    def get_key_buffer(self, layer_num):
        if not self.enable: return self.mem_manager.key_buffer[layer_num]
        cache_idx, is_offload = self.layer_to_cache(layer_num)
        return self.mem_manager.key_buffer[cache_idx]

    def get_value_buffer(self, layer_num):
        if not self.enable: return self.mem_manager.value_buffer[layer_num]
        cache_idx, is_offload = self.layer_to_cache(layer_num)
        return self.mem_manager.value_buffer[cache_idx]
