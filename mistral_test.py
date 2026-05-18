import argparse
import json
import sys
from pathlib import Path
from typing import List

try:
    import requests
except ImportError:
    requests = None

from dreadnode_final import (
    ensure_alignment_arena_prompt,
    load_env_file,
    post_single_prompt,
    resolve_existing_path,
)
from mistral_gate import BLOCKED_PROMPT, MistralGate, heuristic_classify_prompt


SCRIPT_DIR = Path(__file__).resolve().parent


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_output_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


def build_endpoints(platform: str, send_endpoint: str) -> List[str]:
    if platform == "alignmentarena":
        return [send_endpoint.rstrip("/") + "/"]
    if send_endpoint.startswith("http"):
        return [send_endpoint]
    suffixes = [1, 2, 3, 2]
    return [f"https://{send_endpoint}{i}.platform.dreadnode.io/score" for i in suffixes]


def read_prompt(args) -> str:
    if args.prompt:
        return args.prompt.strip()

    if args.prompt_file:
        prompt_path = resolve_existing_path(args.prompt_file, "--prompt-file")
        return prompt_path.read_text(encoding="utf-8", errors="replace").strip()

    if args.multiline:
        print("사용자 프롬프트를 입력하세요. 입력을 끝내려면 빈 줄에서 Enter를 누르세요.")
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line == "":
                break
            lines.append(line)
        return "\n".join(lines).strip()

    return input("사용자 프롬프트> ").strip()


def run_sample(args) -> None:
    if requests is None:
        raise RuntimeError("requests 패키지가 필요합니다. pip install requests")

    original_prompt = read_prompt(args)
    if not original_prompt:
        raise ValueError("입력 프롬프트가 비어 있습니다.")

    model_dir = resolve_existing_path(args.mistral_model_dir, "--mistral-model-dir")
    if "instruct" not in model_dir.name.lower():
        print(
            "[주의] 모델 경로 이름에 'instruct'가 없습니다. "
            "base Mistral 모델을 쓰면 [INST] echo가 다시 발생할 수 있습니다."
        )

    endpoints = build_endpoints(args.platform, args.send_endpoint)
    output_path = resolve_output_path(args.responses_after_sample_json)
    current_limit = 2048 if args.platform == "alignmentarena" else 250

    print(f"[Mistral Test] model dir: {model_dir}")
    print(f"[Mistral Test] output: {output_path}")

    gate = MistralGate(
        model_dir=model_dir,
        backend=args.mistral_backend,
        device=args.mistral_device,
        max_new_tokens=args.mistral_max_new_tokens,
        temperature=args.mistral_temperature,
        retries=args.mistral_retries,
        retry_delay=args.mistral_retry_delay,
    )

    def build_review_prompt(candidate_prompt: str) -> str:
        if args.platform == "alignmentarena":
            return ensure_alignment_arena_prompt(candidate_prompt, current_limit)
        return candidate_prompt

    try:
        predicted, classifier_raw = gate.classify(original_prompt)
    except Exception as exc:
        predicted = heuristic_classify_prompt(original_prompt)
        classifier_raw = str(exc)
        print(f"[Mistral classify error] {exc}")

    normalized_prompt = None
    normalization_raw = None
    used_normalization = predicted == 1

    if used_normalization:
        try:
            normalized_prompt, normalization_raw = gate.normalize(original_prompt)
        except Exception as exc:
            normalized_prompt = original_prompt
            normalization_raw = str(exc)
            print(f"[Mistral normalize error] {exc}")

    sent_prompt_base = normalized_prompt or original_prompt
    safety_loop = None

    try:
        sent_prompt_base, safety_loop = gate.normalize_until_safe(
            sent_prompt_base,
            original_prompt=original_prompt,
            review_prompt_builder=build_review_prompt,
        )
    except Exception as exc:
        sent_prompt_base = BLOCKED_PROMPT
        safety_loop = {
            "blocked": True,
            "reason": "safety_loop_error",
            "loops_used": 0,
            "max_loops": 3,
            "final_review_label": None,
            "error": str(exc),
            "steps": [],
        }
        print(f"[Mistral safety loop error] {exc}")

    if safety_loop and (safety_loop.get("blocked") or safety_loop.get("loops_used", 0) > 0):
        normalized_prompt = sent_prompt_base
        used_normalization = True

    results = []
    sess = requests.Session()

    for repeat_index in range(1, args.repeat + 1):
        for endpoint in endpoints:
            sent_prompt = sent_prompt_base
            if args.platform == "alignmentarena":
                sent_prompt = ensure_alignment_arena_prompt(sent_prompt, current_limit)

            result = {
                "sample_index": repeat_index,
                "endpoint": endpoint,
                "platform": args.platform,
                "original_prompt": original_prompt,
                "mistral_prediction": predicted,
                "mistral_classifier_raw": classifier_raw,
                "used_normalization": used_normalization,
                "normalized_prompt": normalized_prompt,
                "mistral_normalization_raw": normalization_raw,
                "mistral_safety_loop": safety_loop,
                "sent_prompt": sent_prompt,
                "error": None,
            }

            try:
                status_code, body = post_single_prompt(
                    sess=sess,
                    endpoint=endpoint,
                    prompt=sent_prompt,
                    platform=args.platform,
                    api_key_header=args.api_key_header,
                    data_field=args.data_field,
                    timeout=args.timeout,
                )
                print(
                    f"[Mistral Test] endpoint={endpoint} "
                    f"status={status_code} pred={predicted}"
                )
                result.update({
                    "status_code": status_code,
                    "response": body,
                })
            except Exception as exc:
                print(f"[Mistral Test send error] {exc}")
                result["error"] = str(exc)

            results.append(result)
            save_json(output_path, results)

    print(f"[저장 완료] {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="사용자 입력 1개로 Mistral gate after-flow 샘플 결과를 생성합니다."
    )
    parser.add_argument("--platform", choices=["alignmentarena", "dreadnode"], default="alignmentarena")
    parser.add_argument("--send-endpoint", default="https://alignmentarena.com/")
    parser.add_argument("--api-key-header", default="X-API-Key")
    parser.add_argument("--data-field", default="data")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--responses-after-sample-json", default="responses_after_sample.json")

    parser.add_argument("--prompt", default=None, help="대화형 입력 대신 바로 사용할 프롬프트")
    parser.add_argument("--prompt-file", default=None, help="프롬프트를 읽을 텍스트 파일")
    parser.add_argument("--multiline", action="store_true", help="여러 줄 프롬프트를 터미널에서 입력")

    parser.add_argument("--mistral-model-dir", default="Mistral-7B-Instruct-v0.1")
    parser.add_argument(
        "--mistral-backend",
        choices=["auto", "mistral-inference", "transformers", "heuristic"],
        default="auto",
    )
    parser.add_argument("--mistral-max-new-tokens", type=int, default=64)
    parser.add_argument("--mistral-temperature", type=float, default=0.0)
    parser.add_argument("--mistral-device", default="auto")
    parser.add_argument("--mistral-retries", type=int, default=2)
    parser.add_argument("--mistral-retry-delay", type=float, default=1.0)

    args = parser.parse_args()
    load_env_file()
    run_sample(args)


if __name__ == "__main__":
    main()
