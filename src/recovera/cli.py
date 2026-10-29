"""Command line interface."""

import argparse
import json
import logging
import sys
from pathlib import Path

_SRC_DIR = str(Path(__file__).resolve().parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Recovera's Panda pick-and-place baseline")
    parser.add_argument("--model", type=Path, default=Path("assets/franka_emika_panda/panda.xml"))
    parser.add_argument("--viewer", action="store_true", help="show the simulation in MuJoCo's GUI")
    parser.add_argument(
        "--scenario",
        choices=("baseline", "recovery-demo", "drop-demo", "normal", "mid-air-drop", "obstacle-replan", "blocked-route", "beyond-reach"),
        help="run a named scenario headlessly or with --viewer; omit for the interactive menu",
    )
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO"
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="directory for JSON trial records")
    guidance = parser.add_mutually_exclusive_group()
    guidance.add_argument("--llm", dest="llm_enabled", action="store_true")
    guidance.add_argument("--no-llm", dest="llm_enabled", action="store_false")
    parser.set_defaults(llm_enabled=None)
    parser.add_argument("--llm-provider", choices=("ollama", "mock"), default="ollama")
    parser.add_argument("--llm-model", default="qwen3:4b")
    parser.add_argument("--llm-endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--llm-timeout", type=float, default=8.0)
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--llm-num-predict", type=int, default=192)
    parser.add_argument(
        "--llm-fallback", choices=("deterministic", "stop"), default="deterministic"
    )
    parser.add_argument("--llm-budget", type=float, default=16.0)
    parser.add_argument(
        "--llm-mock-strategy",
        choices=(
            "reapproach_and_regrasp",
            "centered_reapproach_and_regrasp",
            "refresh_and_replan",
            "safe_stop",
        ),
        help="strategy selected by the explicit mock provider; never used by Ollama",
    )
    parser.add_argument(
        "--check-llm",
        action="store_true",
        help="check the configured local Ollama model; does not download it or run a trial",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    from recovera.config import RecoveraConfig
    from recovera.llm import OllamaProvider
    from recovera.runner import run_trial
    from recovera.scenarios import get_scenario

    if args.check_llm:
        if args.llm_provider != "ollama":
            parser.error("--check-llm checks real Ollama only; use --llm-provider ollama")
        provider = OllamaProvider(
            endpoint=args.llm_endpoint,
            model=args.llm_model,
            timeout_seconds=args.llm_timeout,
            temperature=args.llm_temperature,
            num_predict=args.llm_num_predict,
        )
        available, message = provider.check_model_available()
        logging.getLogger("recovera.LLM").info("%s", message)
        if not available:
            raise SystemExit(1)
        return

    scenario = args.scenario or _interactive_scenario()
    if scenario is None:
        return
    scenario_aliases = {"normal": "baseline", "mid-air-drop": "drop-demo"}
    selected = scenario_aliases.get(scenario, scenario)
    result = run_trial(
        RecoveraConfig(
            model_path=args.model,
            scenario_id=selected,
            viewer=args.viewer,
            demo_fail_first_grasp=selected == "recovery-demo",
            demo_drop_after_grasp=selected == "drop-demo",
            llm_enabled=(
                args.llm_enabled
                if args.llm_enabled is not None
                else get_scenario(selected).llm_default
            ),
            llm_provider=args.llm_provider,
            llm_mock_strategy=args.llm_mock_strategy,
            llm_model=args.llm_model,
            llm_endpoint=args.llm_endpoint,
            llm_timeout_seconds=args.llm_timeout,
            llm_temperature=args.llm_temperature,
            llm_num_predict=args.llm_num_predict,
            llm_fallback=args.llm_fallback,
            llm_total_budget_seconds=args.llm_budget,
        )
    )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    record = args.results_dir / f"{result.trial_id or 'trial'}.json"
    record.write_text(json.dumps(result.model_dump(mode="json"), indent=2) + "\n")
    logging.getLogger("recovera.SIMULATION").info("trial record saved: %s", record.resolve())
    print(json.dumps(result.model_dump(mode="json"), indent=2))


def _interactive_scenario() -> str | None:
    from recovera.scenarios import MENU_SCENARIO_IDS, get_scenario

    while True:
        print("\n========================================\n          RECOVERA SIMULATION\n========================================\n\nSelect a scenario:\n")
        for number, scenario_id in enumerate(MENU_SCENARIO_IDS, start=1):
            print(f"{number}. {get_scenario(scenario_id).label}")
        print("0. Exit")
        try:
            selected = input("\nEnter scenario number: ").strip()
        except EOFError:
            return None
        if selected == "0":
            return None
        if selected.isdigit() and 1 <= int(selected) <= len(MENU_SCENARIO_IDS):
            return MENU_SCENARIO_IDS[int(selected) - 1]
        print("Please enter a number from 0 to 5.")


if __name__ == "__main__":
    main()
