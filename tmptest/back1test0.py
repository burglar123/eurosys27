import json
import time
from nano_pearl import PEARLConfig, PEARLEngine
from nano_pearl.layers.sampler import SamplingParams


def main():
    draft_model_path = "/root/autodl-tmp/models/Qwen3-0.6B"
    target_model_path = "/root/autodl-tmp/models/Qwen3-8B"

    config = PEARLConfig(
        draft_model_path=draft_model_path,
        target_model_path=target_model_path,
        draft_tensor_parallel_size=1,
        target_tensor_parallel_size=1,
        max_num_batched_tokens=8192,
        max_num_seqs=128,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        gamma=4,
    )

    engine = PEARLEngine(config)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=32,
        ignore_eos=False,
    )

    now = time.time()

    engine.add_request(
        "Explain speculative decoding briefly.",
        sampling_params,
        request_id="req-0001",
        arrival_ts=now,
        slo_tpot_ms=50.0,
        slo_class="normal",
        per_request_gamma=4,
    )

    output_text, num_tokens, num_acc_tokens, elapsed_time = engine.generate()

    print("output_text:", output_text)
    print("num_tokens:", num_tokens)
    print("num_acc_tokens:", num_acc_tokens)
    print("elapsed_time:", elapsed_time)

    engine.dump_traces_json("trace.json")
    print("trace dumped to trace.json")

    engine.exit()


if __name__ == "__main__":
    main()