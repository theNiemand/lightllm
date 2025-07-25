import time
from lightllm.utils.log_utils import init_logger
from .batch import Batch

logger = init_logger(__name__)


class Stats:
    def __init__(self, log_status, log_stats_interval) -> None:
        self.log_stats = log_status
        self.log_stats_interval = log_stats_interval
        self.last_log_time = time.time()
        self.all_tokens = 0
        self.output_tokens = 0
        self.prompt_tokens = 0
        return

    def count_prompt_tokens(self, run_batch: Batch):
        if self.log_stats and run_batch is not None:
            tokens = run_batch.input_tokens()
            self.prompt_tokens += tokens
            self.all_tokens += tokens
        return

    def count_output_tokens(self, run_batch: Batch):
        if self.log_stats and run_batch is not None:
            tokens = len(run_batch.reqs)
            self.output_tokens += tokens
            self.all_tokens += tokens
        return

    def print_stats(self):
        if not self.log_stats:
            return

        now = time.time()
        if now - self.last_log_time > self.log_stats_interval:
            logger.debug(
                f"Avg tokens(prompt+generate) throughput: {self.all_tokens/(now-self.last_log_time):8.3f} tokens/s\n"
                f"Avg prompt tokens throughput:           {self.prompt_tokens/(now-self.last_log_time):8.3f} tokens/s\n"
                f"Avg generate tokens throughput:         {self.output_tokens/(now-self.last_log_time):8.3f} tokens/s"
            )
            self.all_tokens = 0
            self.output_tokens = 0
            self.prompt_tokens = 0
            self.last_log_time = now
        return
