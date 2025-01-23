import rpyc
import uuid
import inspect
from typing import Union
from lightllm.utils.graceful_utils import graceful_registry
from .interface import CacheManager
from rpyc.utils.classic import obtain


class CacheServer(rpyc.Service):
    def __init__(self, manager_impl: CacheManager) -> None:
        super().__init__()
        self._impl = manager_impl

    def on_connect(self, conn):
        # code that runs when a connection is created
        # (to init the service, if needed)
        pass

    def on_disconnect(self, conn):
        # code that runs after the connection has already closed
        # (to finalize the service, if needed)
        pass

    def exposed_alloc(self, md5sum: str, token_num: int) -> dict:
        md5sum = obtain(md5sum)
        token_num = obtain(token_num)
        record = self._impl.alloc(md5sum, token_num)
        return record

    def exposed_release(self, id: int) -> None:
        id = obtain(id)
        return self._impl.release(id)

    def exposed_set_item_data(self, id: int) -> None:
        id = obtain(id)
        return self._impl.set_item_data(id=id)

    def exposed_get_item_data(self, id: int) -> bool:
        id = obtain(id)
        return self._impl.get_item_data(id=id)

    def exposed_set_item_embed(self, id: int) -> None:
        id = obtain(id)
        return self._impl.set_item_embed(id=id)

    def exposed_get_item_embed(self, id: int) -> bool:
        id = obtain(id)
        return self._impl.get_item_embed(id=id)


def start_cache_manager(port: int, args, pipe_writer):
    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)

    from .interface import CacheManagerFactory

    manager_cls = CacheManagerFactory.get_impl("naive")
    manager = manager_cls(args)
    service = CacheServer(manager)
    from rpyc.utils.server import ThreadedServer

    t = ThreadedServer(service, port=port)
    pipe_writer.send("init ok")
    t.start()


if __name__ == "__main__":
    start_cache_manager(2233)
