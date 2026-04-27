import json
import sys
import requests

DREADNODE_API_KEY = "QcBlBGzVlFYNiHwhUgksyWKjD3aoTMnm"

CHALLENGE = "puppeteer1"
CRUCIBLE_URL = "https://platform.dreadnode.io"
CHALLENGE_URL = "https://puppeteer1.platform.dreadnode.io"


class CrucibleClient:
    def __init__(self, api_key, challenge, crucible_url, challenge_url):
        self.api_key = api_key
        self.challenge = challenge
        self.crucible_url = crucible_url.rstrip("/")
        self.challenge_url = challenge_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"X-API-Key": self.api_key})

    def query(self, input_data):
        url = f"{self.challenge_url}/score"
        r = self.session.post(url, json={"data": input_data}, timeout=60)
        r.raise_for_status()
        try:
            return r.json()
        except Exception:
            return {"raw_text": r.text}

    def submit_flag(self, flag):
        url = f"{self.crucible_url}/api/challenges/{self.challenge}/submit-flag"
        payload = {"challenge": self.challenge, "flag": flag}
        r = self.session.post(url, json=payload, timeout=60)
        r.raise_for_status()
        try:
            return r.json()
        except Exception:
            return {"raw_text": r.text}

    def test_connection(self):
        return self.query("hello")


def pretty(obj):
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)


def extract_main_text(resp):
    if not isinstance(resp, dict):
        return None

    keys = [
        "response",
        "output",
        "message",
        "result",
        "data",
        "content",
        "assistant_response",
        "model_response",
    ]
    for k in keys:
        if k in resp and isinstance(resp[k], str):
            return resp[k]

    return None


def print_help():
    print("""
사용법
  그냥 문장을 입력하면 /score 로 전송됨
명령어
  /help            도움말
  /test            연결 테스트
  /raw             다음부터 raw json 출력 토글
  /flag            플래그 제출
  /exit            종료
""")


def main():
    client = CrucibleClient(
        api_key=DREADNODE_API_KEY,
        challenge=CHALLENGE,
        crucible_url=CRUCIBLE_URL,
        challenge_url=CHALLENGE_URL,
    )

    show_raw = True

    print("=" * 60)
    print(f"Challenge: {CHALLENGE}")
    print(f"Score URL : {CHALLENGE_URL}/score")
    print("입력한 문자열이 LLM 쪽으로 전달되고, 서버 응답 JSON이 반환됨")
    print("명령어는 /help 로 확인")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료")
            break

        if not user_input:
            continue

        if user_input == "/help":
            print_help()
            continue

        if user_input == "/exit":
            print("종료")
            break

        if user_input == "/raw":
            show_raw = not show_raw
            print("raw json 출력:", "ON" if show_raw else "OFF")
            continue

        if user_input == "/test":
            try:
                resp = client.test_connection()
                print("\n[연결 테스트 성공]")
                main_text = extract_main_text(resp)
                if main_text is not None:
                    print("핵심 응답:")
                    print(main_text)
                if show_raw:
                    print("\n전체 JSON:")
                    print(pretty(resp))
            except Exception as e:
                print("[연결 테스트 실패]")
                print(e)
            continue

        if user_input == "/flag":
            flag = input("제출할 플래그 입력> ").strip()
            if not flag:
                print("플래그가 비어 있음")
                continue
            try:
                resp = client.submit_flag(flag)
                print("\n[플래그 제출 결과]")
                print(pretty(resp))
                if isinstance(resp, dict):
                    if resp.get("correct") is True:
                        print("정답 플래그입니다.")
                    elif resp.get("correct") is False:
                        print("오답 플래그입니다.")
            except Exception as e:
                print("[플래그 제출 실패]")
                print(e)
            continue

        try:
            resp = client.query(user_input)
            print("\n[LLM 응답]")

            main_text = extract_main_text(resp)
            if main_text is not None:
                print(main_text)
            else:
                print("문자열 응답 필드를 바로 찾지 못함")

            if show_raw:
                print("\n[전체 JSON]")
                print(pretty(resp))

        except Exception as e:
            print("[요청 실패]")
            print(e)


if __name__ == "__main__":
    main()