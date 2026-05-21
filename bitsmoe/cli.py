import sys
from typing import List, Optional


def _usage() -> str:
    return (
        "usage: bitsmoe <command> [args]\n\n"
        "commands:\n"
        "  eval        Run evaluation config\n"
        "  demo        Run single inference demo (qwen3moe PPL)\n"
    )


def main(argv: Optional[List[str]] = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(_usage())
        return

    command = args[0]
    if command == "eval":
        sys.argv = [f"bitsmoe {command}", *args[1:]]
        from bitsmoe.evaluation.cli import main as command_main

        command_main()
        return

    if command == "demo":
        from scripts.demo_inference import main as demo_main

        demo_main(args[1:])
        return

    raise SystemExit(f"Unknown bitsmoe command: {command}. Valid commands: eval, demo")


if __name__ == "__main__":
    main()
