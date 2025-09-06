from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import os
import json
import requests
import subprocess
import threading
from io import TextIOBase

# BASE_URL = "https://31pwr5t6ij.execute-api.eu-west-2.amazonaws.com"
BASE_URL = "http://127.0.0.1:8009"
ALPHABET = "012345"  # door labels

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CPP_EXE = os.path.join(ROOT_DIR, 'vs', 'solver', 'bin', 'Release', 'solver.exe')

# ---------- API Client ----------

class AedificiumClient:
    def __init__(self, base_url: str = BASE_URL, team_id: str = 'ignored'):
        self.base_url = base_url.rstrip("/")
        self.id = team_id

    def _post(self, path: str, payload: Dict) -> Dict:
        url = f"{self.base_url}{path}"
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        # print(f'req: {payload}')
        # print(f'res: {r.json()}')
        return r.json()

    def register(self, name: str, pl: str, email: str) -> str:
        data = {"name": name, "pl": pl, "email": email}
        j = self._post("/register", data)
        self.id = j["id"]
        return self.id

    def select_problem(self, problem_name: str) -> str:
        assert self.id, "team id is required. call register() or set client.id"
        j = self._post("/select", {"id": self.id, "problemName": problem_name})
        return j["problemName"]

    def explore(self, plans: List[str]) -> Tuple[List[List[int]], int]:
        assert self.id, "team id is required"
        j = self._post("/explore", {"id": self.id, "plans": plans})
        return j["results"], j["queryCount"]

    def guess(self, guess_map: Dict) -> bool:
        assert self.id, "team id is required"
        j = self._post("/guess", {"id": self.id, "map": guess_map})
        return bool(j.get("correct", False))
    

class CppSolverProcess:
    def __init__(self, exe_path: str = CPP_EXE):
        # Windows/MSVC 前提。標準入出力で対話
        self.proc = subprocess.Popen(
            [exe_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,              # 文字列で扱う（改行は \n を自動変換）
            encoding="utf-8",
            bufsize=1               # 行バッファ
        )
        # 非同期で stderr を吸う（デッドロック回避用）
        self._stderr_buf = []
        self._stderr_thread = threading.Thread(target=self._pump_stderr, daemon=True)
        self._stderr_thread.start()

    def _pump_stderr(self):
        try:
            for line in self.proc.stderr:
                self._stderr_buf.append(line)
        except Exception:
            pass

    def write_line(self, s: str):
        assert self.proc.stdin is not None
        self.proc.stdin.write(s.rstrip("\r\n") + "\n")
        self.proc.stdin.flush()

    def read_line(self, timeout: Optional[float] = None) -> str:
        """1行読み取り。timeout なし（None）だとブロッキング。"""
        assert self.proc.stdout is not None
        if timeout is None:
            line = self.proc.stdout.readline()
            if line == "":
                raise RuntimeError("CPP solver closed stdout unexpectedly.\n" + "".join(self._stderr_buf))
            return line.rstrip("\r\n")
        # 簡易タイムアウト
        buf = {"line": None}
        def _t():
            buf["line"] = self.proc.stdout.readline()
        th = threading.Thread(target=_t)
        th.start()
        th.join(timeout)
        if th.is_alive():
            raise TimeoutError("Timed out waiting for line from CPP solver.")
        line = buf["line"]
        if line == "":
            raise RuntimeError("CPP solver closed stdout unexpectedly.\n" + "".join(self._stderr_buf))
        return line.rstrip("\r\n")

    def read_json_object(self, timeout: Optional[float] = None) -> Dict:
        """中括弧の対応で JSON オブジェクト終端まで読む。複数行でもOK。"""
        assert self.proc.stdout is not None
        chunks = []
        depth = 0
        started = False

        def _feed_line() -> str:
            return self.read_line(timeout=timeout)

        while True:
            line = _feed_line()
            chunks.append(line)
            # 行内の { と } をカウント（文字列・エスケープの精密処理は簡略化）
            for ch in line:
                if ch == '{':
                    depth += 1
                    started = True
                elif ch == '}':
                    depth -= 1
            if started and depth <= 0:
                txt = "\n".join(chunks)
                # 余計な前後テキストを削る（最初と最後のオブジェクトを抽出）
                first = txt.find('{')
                last = txt.rfind('}')
                if first == -1 or last == -1 or last < first:
                    raise ValueError("Malformed JSON block from CPP solver:\n" + txt)
                obj_txt = txt[first:last+1]
                return json.loads(obj_txt)

    def close(self):
        try:
            if self.proc and self.proc.poll() is None:
                self.proc.stdin.close()
                self.proc.terminate()
        except Exception:
            pass


if __name__ == '__main__':

    client = AedificiumClient()
    problem_name = client.select_problem('probatio')

    # launch cpp solver
    cpp = CppSolverProcess(CPP_EXE)

    # send problem name
    cpp.write_line(problem_name)

    # receive one-shot plans
    try:
        n_plans_line = cpp.read_line()
        n_plans = int(n_plans_line.strip())
    except Exception as e:
        cpp.close()
        raise RuntimeError(f"Failed to read number of plans from C++: {e}")
    
    plans: List[str] = []
    for _ in range(n_plans):
        plan = cpp.read_line().strip()
        # 軽いバリデーション（0..5 のみ）
        if any(ch not in ALPHABET for ch in plan):
            cpp.close()
            raise ValueError(f"Plan contains non-door char: {plan}")
        plans.append(plan)

    results, qctr = client.explore(plans)
    # print(results, qctr) # ([[0, 3, 1, 1, 3, 1, 3], [0, 1, 1, 3, 1, 0, 3]], 3)

    # send label-path list
    for rec in results:
        line = "".join(str(x) for x in rec)
        cpp.write_line(line)

    # receive solution (json)
    sol = cpp.read_json_object()
    
    cpp.close()

    client.guess(sol)