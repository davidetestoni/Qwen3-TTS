# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel


def main():
    device = "cuda:0"
    MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

    tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation=None,
        # attn_implementation="flash_attention_2",
    )

    # -------- Single --------
    torch.cuda.synchronize()
    t0 = time.time()

    wavs, sr = tts.generate_voice_design(
        text="H-hey! You dropped your... uh... calculus notebook? I mean, I think it's yours? Maybe?",
        language="English",
        instruct="Male, 17 years old, tenor range, gaining confidence - deeper breath support now, though vowels still tighten when nervous",
    )

    torch.cuda.synchronize()
    t1 = time.time()
    print(f"[VoiceDesign Single] time: {t1 - t0:.3f}s")

    sf.write("qwen3_tts_test_voice_design_single.wav", wavs[0], sr)


if __name__ == "__main__":
    main()
